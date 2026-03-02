#!/usr/bin/env python3
"""
Aggregate per-proteome timings into compact averages.

Input:
  - src/out/timings.csv (unified schema)

Grouping key:
  - model_name, type, model_type, cores, batch_size

Output:
  - src/out/proteome_timings.csv with:
      model_name, type, model_type, cores, batch_size,
      avg_time_seconds, avg_time_hours,
      avg_number_proteins, avg_total_aa,
      run_count
"""

from pathlib import Path
import argparse
import pandas as pd


OUTPUT_DIR = Path("src/out")
DEFAULT_INPUT = OUTPUT_DIR / "timings.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "proteome_timings.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate unified timings.csv into proteome_timings.csv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to unified timings.csv (default: src/out/timings.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write proteome_timings.csv (default: src/out/proteome_timings.csv)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    group_cols = ["model_name", "type", "model_type", "cores", "batch_size"]
    agg_rows = []

    for key, g in df.groupby(group_cols, dropna=False):
        model_name, typ, model_type, cores, batch = key
        avg_time_seconds = g["time_seconds"].mean()
        avg_time_hours = g["time_hours"].mean()
        avg_number_proteins = g["number_proteins"].mean()
        avg_total_aa = g["total_aa"].mean()
        run_count = len(g)

        agg_rows.append(
            {
                "model_name": model_name,
                "type": typ,
                "model_type": model_type,
                "cores": cores,
                "batch_size": batch,
                "avg_time_seconds": avg_time_seconds,
                "avg_time_hours": avg_time_hours,
                "avg_number_proteins": avg_number_proteins,
                "avg_total_aa": avg_total_aa,
                "run_count": run_count,
            }
        )

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(args.output, index=False)
    print(f"Wrote {len(agg)} rows to {args.output}")


if __name__ == "__main__":
    main()
