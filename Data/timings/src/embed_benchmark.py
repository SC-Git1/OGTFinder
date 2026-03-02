import os
import time
import csv
import argparse
import glob
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
import esm

UNIFIED_HEADER = [
    "model_name",
    "type",
    "model_type",
    "cores",
    "batch_size",
    "proteome",
    "time_seconds",
    "time_hours",
    "number_proteins",
    "total_aa",
]

def normalize_model_name(model_name: str) -> str:
    """Map common aliases to canonical model IDs."""
    if model_name.strip().lower() == "prostt5":
        return "Rostlab/ProstT5"
    return model_name

def is_t5_family_model(model_name: str) -> bool:
    lower = model_name.lower()
    return ("prot_t5" in lower) or ("prostt5" in lower)

def load_prottrans_model(model_name, num_gpus):
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
    model.eval()
    return model, tokenizer

def load_esm_model(model_name, num_gpus):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
    model.eval()
    return model, batch_converter, alphabet

def parse_fasta(proteome_path):
    seqs = []
    with open(proteome_path) as f:
        name = None
        seq = ''
        for line in f:
            if line.startswith('>'):
                if name and seq:
                    seqs.append(seq)
                name = line.strip()[1:]
                seq = ''
            else:
                seq += line.strip()
        if name and seq:
            seqs.append(seq)
    return seqs

def embed_proteome_prottrans(model, tokenizer, seqs, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="Batch", unit="batch"):
        batch = seqs[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)

def embed_proteome_esm(model, batch_converter, alphabet, data, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(data), batch_size), desc="Batch", unit="batch"):
        batch = data[i:i+batch_size]  # list of (idx, seq)
        batch_ids = [(str(idx), seq) for idx, seq in batch]
        # converter returns (labels, seq_strs, tokens)
        _, _, tokens = batch_converter(batch_ids)
        tokens = tokens.cuda()
        with torch.no_grad():
            results = model(tokens,
                            repr_layers=[model.module.num_layers if hasattr(model, 'module') else model.num_layers],
                            return_contacts=False)
        layer = model.module.num_layers if hasattr(model, 'module') else model.num_layers
        reps = results['representations'][layer]
        for j, (orig_idx, seq) in enumerate(batch):
            # count non-padding tokens to get sequence length including special tokens
            seq_len = (tokens[j] != alphabet.padding_idx).sum().item()
            emb = reps[j, 1:seq_len-1].mean(dim=0)
            embeddings.append(emb.cpu())
    return torch.stack(embeddings, dim=0)

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark ProtTrans and ESM2 embedding times for proteomes.'
    )
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name: e.g. Rostlab/prot_t5_xl_uniref50, Rostlab/ProstT5, or esm2_t36_3B_UR50D')
    parser.add_argument('--hardware', type=str, required=True,
                        help='Hardware identifier, e.g. h100, a100, p100, v100, xeon-36, xeon-76')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding')
    parser.add_argument('--proteome_dir', type=str, required=True,
                        help='Directory containing FASTA proteome files')
    parser.add_argument('--output_csv', type=str, default='src/out/timings.csv',
                        help='CSV file to append timing results')
    args = parser.parse_args()
    args.model_name = normalize_model_name(args.model_name)

    if not os.path.exists(args.output_csv):
        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(UNIFIED_HEADER)

    if is_t5_family_model(args.model_name):
        model, tokenizer = load_prottrans_model(args.model_name, args.num_gpus)
        is_prottrans = True
    else:
        model, batch_converter, alphabet = load_esm_model(args.model_name, args.num_gpus)
        is_prottrans = False

    files = sorted(glob.glob(os.path.join(args.proteome_dir, '*.faa')))
    total_time = 0.0
    total_seqs = 0
    total_aa = 0

    for idx, proteome_file in enumerate(tqdm(files, desc="Proteomes", unit="file"), start=1):
        seqs = parse_fasta(proteome_file)
        num_seqs = len(seqs)
        aa_count = sum(len(s) for s in seqs)
        tqdm.write(
            f"Starting {idx}/{len(files)}: {os.path.basename(proteome_file)} "
            f"with {num_seqs} seqs ({aa_count} aa)"
        )

        start = time.time()
        if is_prottrans:
            _ = embed_proteome_prottrans(model, tokenizer, seqs, args.batch_size)
        else:
            data = list(enumerate(seqs))
            _ = embed_proteome_esm(model, batch_converter, alphabet, data, args.batch_size)
        elapsed = time.time() - start

        total_time += elapsed
        total_seqs += num_seqs
        total_aa += aa_count
        avg_time = total_time / idx
        time_hours = elapsed / 3600.0

        with open(args.output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                args.model_name, 'gpu', args.hardware, '', args.batch_size,
                os.path.basename(proteome_file), f'{elapsed:.4f}', f'{time_hours:.6f}',
                num_seqs, aa_count
            ])
        tqdm.write(f"Finished {idx}/{len(files)} in {elapsed:.2f}s (avg {avg_time:.2f}s)")

    overall_avg = total_time / len(files) if files else 0
    print(
        f"All proteomes processed: hardware={args.hardware}, total_seqs={total_seqs}, "
        f"total_aa={total_aa}, total_time={total_time:.2f}s, avg_time={overall_avg:.2f}s"
    )

if __name__ == '__main__':
    main()
