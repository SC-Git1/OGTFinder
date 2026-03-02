#!/usr/bin/env python3
"""
Unified SVG figure generation for protein-LLM benchmark timings.

Reads the aggregated proteome_timings.csv (containing both CPU and GPU rows)
and produces 18 SVG figures: each plot in both hours and minutes variants.

Output directory layout (--outdir, default src/out/figures):
  <outdir>/
    cpu_figures/                    -- 6 CPU barplots (hours + minutes)
      cpu_bar_avg_time_all_models.svg / _minutes.svg
      cpu_bar_avg_time_esm2_only.svg  / _minutes.svg
      cpu_bar_avg_time_rostlab_only.svg / _minutes.svg
    gpu_figures/                    -- 6 GPU barplots (hours + minutes)
      gpu_bar_avg_time_all_models.svg / _minutes.svg
      gpu_bar_avg_time_esm2_only.svg  / _minutes.svg
      gpu_bar_avg_time_rostlab_only.svg / _minutes.svg
    dot_cpu_vs_gpu_all_models.svg / _minutes.svg   -- 6 combined dot plots
    dot_cpu_vs_gpu_esm2_only.svg  / _minutes.svg
    dot_cpu_vs_gpu_rostlab_only.svg / _minutes.svg
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

PALETTE = [
    "#e1b067",  # gold
    "#577967",  # dark teal
    "#d86d43",  # burnt orange
    "#517cc0",  # steel blue
    "#a855a0",  # mauve
    "#c94c4c",  # crimson
    "#2a9d8f",  # teal (spare)
]


def assign_colors(categories: List[str]) -> Dict[str, str]:
    unique = list(dict.fromkeys(categories))
    return {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(unique)}


def create_figure(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def require_columns(df: pd.DataFrame, required_cols, dataset_name: str):
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {', '.join(missing)}. "
            f"Available: {', '.join(df.columns)}. "
            "Regenerate with: python proteome_timings.py"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hw_label(row) -> str:
    """Build a short human-readable hardware label for the x-axis."""
    if row["type"] == "gpu":
        return row["model_type"].upper()
    short = (
        row["model_type"]
        .replace("Intel(R) ", "")
        .replace("Xeon(R) ", "Xeon ")
        .replace("CPU @ ", "")
        .strip()
    )
    cores = int(row["cores"]) if pd.notna(row["cores"]) else "?"
    return f"{short}\n{cores} cores"


def _hw_sort_key(label: str):
    """Sort hardware labels: CPUs first (by core count), then GPUs alphabetically."""
    if "\n" in label:
        try:
            cores = int(label.split("\n")[1].replace(" cores", ""))
        except ValueError:
            cores = 0
        return (0, cores, label)
    return (1, 0, label)


def _time_col(unit: str) -> str:
    return "avg_time_per_proteome_min" if unit == "min" else "avg_time_per_proteome_h"


def _time_ylabel(unit: str) -> str:
    return "Avg time per proteome (min)" if unit == "min" else "Avg time per proteome (h)"


def _prepare_agg(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to the aggregated dataframe."""
    df = df_agg.copy()
    df["avg_time_per_proteome_h"] = df["avg_time_hours"]
    df["avg_time_per_proteome_min"] = df["avg_time_hours"] * 60.0
    df["hw_label"] = df.apply(_hw_label, axis=1)
    return df


def _is_esm2(name: str) -> bool:
    return name.lower().startswith("esm2")


def _is_rostlab(name: str) -> bool:
    return name.lower().startswith("rostlab")


# ---------------------------------------------------------------------------
# Barplots
# ---------------------------------------------------------------------------

def _plot_barplot(df: pd.DataFrame, outdir: Path, filename: str,
                  title: str, model_filter: Optional[str] = None,
                  hw_type_filter: Optional[str] = None,
                  time_unit: str = "h"):
    """
    Grouped barplot of avg time per proteome.

    X-axis = hardware slots, bars grouped by model_name.
    hw_type_filter: "cpu" or "gpu" to restrict to one hardware type.
    time_unit: "h" for hours, "min" for minutes.
    """
    if hw_type_filter:
        df = df[df["type"] == hw_type_filter].copy()

    if model_filter == "esm2":
        df = df[df["model_name"].apply(_is_esm2)].copy()
    elif model_filter == "rostlab":
        df = df[df["model_name"].apply(_is_rostlab)].copy()

    if df.empty:
        return

    col = _time_col(time_unit)
    hw_labels = sorted(df["hw_label"].unique(), key=_hw_sort_key)
    model_order = sorted(df["model_name"].unique())
    color_map = assign_colors(model_order)

    n_hw = len(hw_labels)
    n_models = len(model_order)
    width = min(0.8 / n_models, 0.18)

    fig, ax = create_figure(figsize=(max(8, n_hw * 1.2), 5))

    x = np.arange(n_hw)
    for i, model in enumerate(model_order):
        subset = df[df["model_name"] == model]
        y = []
        for hw in hw_labels:
            match = subset[subset["hw_label"] == hw]
            y.append(match[col].iloc[0] if len(match) else np.nan)

        offsets = x + (i - (n_models - 1) / 2) * width
        ax.bar(
            offsets, y, width=width,
            label=model, color=color_map[model],
            edgecolor="black", linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(hw_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(_time_ylabel(time_unit))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(outdir / filename, format="svg")
    plt.close(fig)


def _barplot_pair(df, outdir, base_name, title_base, **kwargs):
    """Generate both hours and minutes variants of a barplot."""
    _plot_barplot(df, outdir, f"{base_name}.svg",
                  f"{title_base} (h)", time_unit="h", **kwargs)
    _plot_barplot(df, outdir, f"{base_name}_minutes.svg",
                  f"{title_base} (min)", time_unit="min", **kwargs)


def plot_cpu_bar_all(df, outdir):
    _barplot_pair(df, outdir, "cpu_bar_avg_time_all_models",
                  "CPU — avg time per proteome — all models",
                  hw_type_filter="cpu")


def plot_cpu_bar_esm2(df, outdir):
    _barplot_pair(df, outdir, "cpu_bar_avg_time_esm2_only",
                  "CPU — avg time per proteome — ESM2 models",
                  model_filter="esm2", hw_type_filter="cpu")


def plot_cpu_bar_rostlab(df, outdir):
    _barplot_pair(df, outdir, "cpu_bar_avg_time_rostlab_only",
                  "CPU — avg time per proteome — Rostlab models",
                  model_filter="rostlab", hw_type_filter="cpu")


def plot_gpu_bar_all(df, outdir):
    _barplot_pair(df, outdir, "gpu_bar_avg_time_all_models",
                  "GPU — avg time per proteome — all models",
                  hw_type_filter="gpu")


def plot_gpu_bar_esm2(df, outdir):
    _barplot_pair(df, outdir, "gpu_bar_avg_time_esm2_only",
                  "GPU — avg time per proteome — ESM2 models",
                  model_filter="esm2", hw_type_filter="gpu")


def plot_gpu_bar_rostlab(df, outdir):
    _barplot_pair(df, outdir, "gpu_bar_avg_time_rostlab_only",
                  "GPU — avg time per proteome — Rostlab models",
                  model_filter="rostlab", hw_type_filter="gpu")


# ---------------------------------------------------------------------------
# CPU-vs-GPU dot plots
# ---------------------------------------------------------------------------

CPU_DOT_LABEL = "Xeon Gold 6240 (36 cores)"
GPU_DOT_LABEL = "V100"


def _plot_dot_cpu_gpu(df: pd.DataFrame, outdir: Path, filename: str,
                      title: str, model_filter: Optional[str] = None,
                      time_unit: str = "h"):
    """
    Paired dot plot: for each model show a CPU dot (Gold 6240 / 36 cores)
    and a GPU dot (V100) at the same x-position.
    time_unit: "h" for hours, "min" for minutes.
    """
    cpu = df[
        (df["type"] == "cpu") &
        (df["model_type"].str.contains("Gold 6240", na=False)) &
        (df["cores"] == 36.0)
    ].copy()
    gpu = df[
        (df["type"] == "gpu") &
        (df["model_type"] == "v100")
    ].copy()

    if model_filter == "esm2":
        cpu = cpu[cpu["model_name"].apply(_is_esm2)]
        gpu = gpu[gpu["model_name"].apply(_is_esm2)]
    elif model_filter == "rostlab":
        cpu = cpu[cpu["model_name"].apply(_is_rostlab)]
        gpu = gpu[gpu["model_name"].apply(_is_rostlab)]

    models_both = sorted(set(cpu["model_name"]) & set(gpu["model_name"]))
    if not models_both:
        return

    col = _time_col(time_unit)
    fig, ax = create_figure(figsize=(max(6, len(models_both) * 1.4), 5))
    x = np.arange(len(models_both))

    cpu_vals = []
    gpu_vals = []
    for m in models_both:
        cpu_vals.append(cpu.loc[cpu["model_name"] == m, col].iloc[0])
        gpu_vals.append(gpu.loc[gpu["model_name"] == m, col].iloc[0])

    ax.scatter(x, cpu_vals, marker="o", s=90, zorder=3,
               color="#517cc0", edgecolor="black", linewidth=0.8,
               label=CPU_DOT_LABEL)
    ax.scatter(x, gpu_vals, marker="s", s=90, zorder=3,
               color="#d86d43", edgecolor="black", linewidth=0.8,
               label=GPU_DOT_LABEL)

    for xi, cv, gv in zip(x, cpu_vals, gpu_vals):
        ax.plot([xi, xi], [cv, gv], color="gray", linewidth=0.7, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(models_both, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(_time_ylabel(time_unit))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / filename, format="svg")
    plt.close(fig)


def _dot_pair(df, outdir, base_name, title_base, **kwargs):
    """Generate both hours and minutes variants of a dot plot."""
    _plot_dot_cpu_gpu(df, outdir, f"{base_name}.svg",
                      f"{title_base} (h)", time_unit="h", **kwargs)
    _plot_dot_cpu_gpu(df, outdir, f"{base_name}_minutes.svg",
                      f"{title_base} (min)", time_unit="min", **kwargs)


def plot_dot_all(df, outdir):
    _dot_pair(df, outdir, "dot_cpu_vs_gpu_all_models",
              "CPU vs GPU — all models")


def plot_dot_esm2(df, outdir):
    _dot_pair(df, outdir, "dot_cpu_vs_gpu_esm2_only",
              "CPU vs GPU — ESM2 models",
              model_filter="esm2")


def plot_dot_rostlab(df, outdir):
    _dot_pair(df, outdir, "dot_cpu_vs_gpu_rostlab_only",
              "CPU vs GPU — Rostlab models",
              model_filter="rostlab")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate unified benchmark SVG figures."
    )
    parser.add_argument(
        "--agg", type=Path,
        default=Path("src/out/proteome_timings.csv"),
        help="Path to aggregated proteome_timings.csv",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("src/out/figures"),
        help="Output directory for SVG figures",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cpu_dir = args.outdir / "cpu_figures"
    gpu_dir = args.outdir / "gpu_figures"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    gpu_dir.mkdir(parents=True, exist_ok=True)

    if not args.agg.is_file():
        raise FileNotFoundError(f"Aggregated CSV not found: {args.agg}")

    df_agg = pd.read_csv(args.agg)
    require_columns(
        df_agg,
        ["model_name", "type", "model_type", "cores", "batch_size",
         "avg_time_seconds", "avg_time_hours"],
        "Aggregated timings CSV",
    )

    df = _prepare_agg(df_agg)

    plot_cpu_bar_all(df, cpu_dir)
    plot_cpu_bar_esm2(df, cpu_dir)
    plot_cpu_bar_rostlab(df, cpu_dir)
    plot_gpu_bar_all(df, gpu_dir)
    plot_gpu_bar_esm2(df, gpu_dir)
    plot_gpu_bar_rostlab(df, gpu_dir)
    plot_dot_all(df, args.outdir)
    plot_dot_esm2(df, args.outdir)
    plot_dot_rostlab(df, args.outdir)

    print(f"18 SVG figures written to: {args.outdir.resolve()}")
    print(f"  CPU barplots: {cpu_dir.resolve()}")
    print(f"  GPU barplots: {gpu_dir.resolve()}")
    print(f"  Dot plots:    {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
