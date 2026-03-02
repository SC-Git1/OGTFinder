## Paths and layout
- Bench scripts live in `src/` (GPU: `src/embed_benchmark.py`, CPU: `src/embed_benchmark_cpu.py`).
- SLURM job files live in `src/slurm/` and write logs to `src/slurm/logfiles/`.
- All results append to `src/out/timings.csv` (unified schema).
- Aggregated averages per model/type/model_type/cores/batch are written to `src/out/proteome_timings.csv`.
- Figures: `src/out/figures/` via `python src/make_figures.py`
- ProtT5/ProstT5: SLURM files per node/model exist under `src/slurm/` (GPU: a100/h100/p100/v100; CPU: wice/wice_rapids/genius/genius16) using env `ml_env` and archive-derived batch sizes.

## Quick usage
GPU (standalone):
```bash
python src/embed_benchmark.py \
  --model_name esm2_t36_3B_UR50D \
  --hardware a100 \
  --num_gpus 4 \
  --batch_size 4 \
  --proteome_dir ./benchmark_proteomes \
  --output_csv src/out/timings.csv
```

CPU (standalone):
```bash
CPU_MODEL=$(lscpu | awk -F: '/Model name/ {print $2}' | sed 's/^ *//')
python src/embed_benchmark_cpu.py \
  --model_name esm2_t36_3B_UR50D \
  --cpu_model "$CPU_MODEL" \
  --num_cores 72 \
  --batch_size 1 \
  --proteome_dir ./benchmark_proteomes \
  --output_csv src/out/timings.csv
```

SLURM:
- Submit any `src/slurm/*.slurm`; they already point to `python src/...` and log to `src/slurm/logfiles/`.

## Unified `timings.csv` schema
- model_name: model identifier (e.g., esm2_t36_3B_UR50D)
- type: `gpu` or `cpu`
- model_type: GPU hardware (e.g., a100) or CPU model string
- cores: CPU cores used (blank for GPU)
- batch_size: batch size used
- proteome: FASTA file name (no `ALL` rows are written)
- time_seconds: runtime per proteome
- time_hours: time_seconds / 3600
- number_proteins: sequences processed
- total_aa: amino acids processed

## Aggregated averages
- Generated via `python src/proteome_timings.py`
- Output: `src/out/proteome_timings.csv`
- Columns: model_name, type, model_type, cores, batch_size, avg_time_seconds, avg_time_hours, avg_number_proteins, avg_total_aa, run_count

## Figures
- `python src/make_figures.py --agg src/out/proteome_timings.csv --outdir src/out/figures`
- Output SVGs (18 total — each plot in hours + minutes):
  - `cpu_figures/` — CPU barplots:
    - `cpu_bar_avg_time_all_models.svg` / `…_minutes.svg` — all models
    - `cpu_bar_avg_time_esm2_only.svg` / `…_minutes.svg` — ESM2 only
    - `cpu_bar_avg_time_rostlab_only.svg` / `…_minutes.svg` — Rostlab only
  - `gpu_figures/` — GPU barplots:
    - `gpu_bar_avg_time_all_models.svg` / `…_minutes.svg` — all models
    - `gpu_bar_avg_time_esm2_only.svg` / `…_minutes.svg` — ESM2 only
    - `gpu_bar_avg_time_rostlab_only.svg` / `…_minutes.svg` — Rostlab only
  - Root `figures/` — paired CPU/GPU dot plots:
    - `dot_cpu_vs_gpu_all_models.svg` / `…_minutes.svg` — all models
    - `dot_cpu_vs_gpu_esm2_only.svg` / `…_minutes.svg` — ESM2 only
    - `dot_cpu_vs_gpu_rostlab_only.svg` / `…_minutes.svg` — Rostlab only

## ProtT5 / ProstT5 SLURM matrix (env: ml_env)
- GPU nodes (per model): `a100`, `h100`, `p100`, `v100`
  - Files: `{node}_prot_t5_xl_uniref50.slurm`, `{node}_prostt5.slurm`
  - Batch sizes from archive defaults: a100=4, h100=8, p100=2, v100=2
- CPU nodes (per model): `wice`, `wice_rapids`, `genius`, `genius16`
  - Files: `{node}_prot_t5_xl_uniref50.slurm`, `{node}_prostt5.slurm`
  - Cores: wice=72, wice_rapids=96, genius=36, genius16=16; batch_size=1
