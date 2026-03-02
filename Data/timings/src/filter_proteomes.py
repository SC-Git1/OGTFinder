#!/usr/bin/env python3
"""
Filter proteome FASTA files to remove sequences longer than 4096 amino acids.
Updates files in place.

Usage:
    python filter_proteomes.py ./benchmark_proteomes
"""

import argparse
from pathlib import Path


MAX_LENGTH = 4096


def parse_fasta(filepath: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file and return list of (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq_parts = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                # Save previous sequence if exists
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq_parts)))
                current_header = line
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        # Don't forget the last sequence
        if current_header is not None:
            sequences.append((current_header, "".join(current_seq_parts)))

    return sequences


def write_fasta(filepath: Path, sequences: list[tuple[str, str]], line_width: int = 60):
    """Write sequences to a FASTA file."""
    with open(filepath, "w") as f:
        for header, seq in sequences:
            f.write(f"{header}\n")
            # Write sequence in lines of line_width characters
            for i in range(0, len(seq), line_width):
                f.write(f"{seq[i:i+line_width]}\n")


def filter_proteome(filepath: Path) -> tuple[int, int]:
    """
    Filter a proteome file to remove sequences > MAX_LENGTH.
    Returns (original_count, filtered_count).
    """
    sequences = parse_fasta(filepath)
    original_count = len(sequences)

    filtered_sequences = [
        (header, seq) for header, seq in sequences if len(seq) <= MAX_LENGTH
    ]
    filtered_count = len(filtered_sequences)

    # Write back to the same file
    write_fasta(filepath, filtered_sequences)

    return original_count, filtered_count


def main():
    parser = argparse.ArgumentParser(
        description=f"Filter proteome FASTA files to remove sequences > {MAX_LENGTH} aa."
    )
    parser.add_argument(
        "proteome_dir",
        type=Path,
        help="Directory containing proteome FASTA files",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".fasta", ".fa", ".faa"],
        help="File extensions to process (default: .fasta .fa .faa)",
    )
    args = parser.parse_args()

    if not args.proteome_dir.is_dir():
        print(f"Error: {args.proteome_dir} is not a directory")
        return 1

    # Find all FASTA files
    fasta_files = []
    for ext in args.extensions:
        fasta_files.extend(args.proteome_dir.glob(f"*{ext}"))

    if not fasta_files:
        print(f"No FASTA files found in {args.proteome_dir}")
        return 1

    print(f"Found {len(fasta_files)} proteome file(s) in {args.proteome_dir}")
    print(f"Filtering sequences longer than {MAX_LENGTH} amino acids...\n")

    total_original = 0
    total_removed = 0

    for fasta_file in sorted(fasta_files):
        original, remaining = filter_proteome(fasta_file)
        removed = original - remaining
        total_original += original
        total_removed += removed

        if removed > 0:
            print(f"{fasta_file.name}: {original} -> {remaining} sequences ({removed} removed)")
        else:
            print(f"{fasta_file.name}: {original} sequences (none removed)")

    print(f"\nSummary:")
    print(f"  Total sequences processed: {total_original}")
    print(f"  Total sequences removed:   {total_removed}")
    print(f"  Total sequences remaining: {total_original - total_removed}")


if __name__ == "__main__":
    main()