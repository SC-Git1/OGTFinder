#!/usr/bin/env python3
"""
paired_bootstrap_test.py
========================
Paired bootstrap resampling test (Koehn 2004) for comparing models
on a shared held-out test set.

Performs all-vs-all pairwise comparisons among a set of models
(both DL and sklearn).  For each (model_a, model_b, metric) triple it:

  1. Loads the N test-set predictions from both models.
  2. Draws B bootstrap samples of size N (with replacement, same indices
     for both systems — the "paired" property).
  3. Computes the corpus-level metric on each bootstrap sample for both
     systems and counts wins.
  4. Reports a two-sided p-value and applies Benjamini-Hochberg FDR
     correction across all comparisons within each metric.

Data sources
------------
DL  : extract_deep_learning/{model}/{model}_test_predictions.csv
        columns: actual, predicted
sklearn : extract_scikitlearn/{model}/testPE.csv
        columns: expected, predicted

All files contain the same N=685 test samples in the same row order.

Usage
-----
  python paired_bootstrap_test.py
  python paired_bootstrap_test.py --models danets,wide_deep,mlp,svr
  python paired_bootstrap_test.py --metrics rmse rmse_top10
  python paired_bootstrap_test.py --B 100000 --seed 2024 --alpha 0.05
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (defaults — override via CLI)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DL_DIR_DEFAULT = SCRIPT_DIR / "extract_deep_learning"
SK_DIR_DEFAULT = SCRIPT_DIR / "extract_scikitlearn"
OUTPUT_CSV_DEFAULT = SCRIPT_DIR / "paired_bootstrap_results.csv"

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_B = 100000
DEFAULT_SEED = 2024
DEFAULT_ALPHA = 0.05
DEFAULT_METRICS = ["rmse", "rmse_top10"]
DEFAULT_MODELS = "danets,attention_mlp,wide_deep,baseline_mlp,mlp,svr,linear"

# ---------------------------------------------------------------------------
# Alias map (user-friendly name → canonical directory name)
# ---------------------------------------------------------------------------

ALIAS_MAP: dict[str, str] = {
    # DL aliases
    "sparse_mlp": "snn",
    # sklearn aliases
    "mlp": "mlp_rmse",
    "svr": "svr_rmse",
    "gaussian_process": "gaussian_process_rmse",
    "knn": "knn_rmse",
    "linear": "ridge_rmse",
    "linear_regression": "ridge_rmse",
}

# ---------------------------------------------------------------------------
# Output CSV fields
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "metric",
    "model_a",
    "model_b",
    "n_test",
    "B",
    "score_a",
    "score_b",
    "observed_diff_b_minus_a",
    "direction",
    "wins_a",
    "wins_b",
    "wins_tie",
    "p_hat_a_better",
    "p_two_sided",
    "p_bh",
    "significant_bh",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr)


def parse_model_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test predictions from a CSV file.

    Handles both DL format (actual, predicted) and sklearn format
    (expected, predicted).

    Returns (y_true, y_pred) as float64 arrays.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        actual_col = next(
            (c for c in fieldnames if c.lower() in ("actual", "expected")), None
        )
        pred_col = next(
            (c for c in fieldnames if c.lower() == "predicted"), None
        )

        if actual_col is None or pred_col is None:
            raise ValueError(
                f"{csv_path}: could not find actual/expected + predicted columns "
                f"in {fieldnames}"
            )

        rows = list(reader)

    y_true = np.array([float(r[actual_col]) for r in rows], dtype=np.float64)
    y_pred = np.array([float(r[pred_col]) for r in rows], dtype=np.float64)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Metric functions (corpus-level, computed on index arrays)
# ---------------------------------------------------------------------------


def corpus_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray
) -> float:
    """RMSE on the samples selected by `idx`."""
    residuals = y_true[idx] - y_pred[idx]
    return float(np.sqrt(np.mean(residuals ** 2)))


def corpus_rmse_top10(
    y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray
) -> float:
    """
    RMSE on the top-10% of samples (by y_true) in the bootstrap sample.

    Uses percentile-based thresholding (y_true >= 90th percentile) to
    match the DL pipeline's calculate_binned_metrics() method.
    """
    yt = y_true[idx]
    yp = y_pred[idx]
    threshold = np.percentile(yt, 90)
    mask = yt >= threshold
    if mask.sum() < 2:
        return float("nan")
    residuals = yt[mask] - yp[mask]
    return float(np.sqrt(np.mean(residuals ** 2)))


# Lookup for metric name → function
METRIC_FN = {
    "rmse": corpus_rmse,
    "rmse_top10": corpus_rmse_top10,
}


# ---------------------------------------------------------------------------
# Core bootstrap procedure
# ---------------------------------------------------------------------------


def paired_bootstrap(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn,
    B: int,
    seed: int,
) -> dict[str, float]:
    """
    Paired bootstrap resampling test (Koehn 2004).

    Parameters
    ----------
    y_true : reference values (length N)
    y_pred_a : system A predictions (length N)
    y_pred_b : system B predictions (length N)
    metric_fn : callable(y_true, y_pred, idx) -> float
    B : number of bootstrap replicates
    seed : RNG seed for reproducibility

    Returns
    -------
    dict with observed scores, win counts, p-values.
    """
    N = len(y_true)
    rng = np.random.RandomState(seed)

    # Observed scores on full test set
    full_idx = np.arange(N)
    obs_a = metric_fn(y_true, y_pred_a, full_idx)
    obs_b = metric_fn(y_true, y_pred_b, full_idx)

    wins_a = 0.0
    wins_b = 0.0
    wins_tie = 0.0

    for _ in range(B):
        idx_b = rng.randint(0, N, size=N)

        score_a = metric_fn(y_true, y_pred_a, idx_b)
        score_b = metric_fn(y_true, y_pred_b, idx_b)

        # Lower RMSE is better, so delta > 0 means A is better
        delta = score_b - score_a

        if delta > 0:
            wins_a += 1.0
        elif delta < 0:
            wins_b += 1.0
        else:
            wins_a += 0.5
            wins_b += 0.5
            wins_tie += 1.0

    p_hat = wins_a / B
    p_two_sided = 2.0 * min(p_hat, 1.0 - p_hat)

    return {
        "obs_a": obs_a,
        "obs_b": obs_b,
        "obs_diff": obs_b - obs_a,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "wins_tie": wins_tie,
        "p_hat_a_better": p_hat,
        "p_two_sided": p_two_sided,
    }


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction (identical to legacy implementation)
# ---------------------------------------------------------------------------


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted = np.empty(n, dtype=float)

    for i, p_val in enumerate(sorted_p):
        rank = i + 1
        adjusted[i] = p_val * n / rank

    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def apply_bh_by_metric(
    results: list[dict[str, object]], alpha: float
) -> None:
    """Apply BH correction separately within each metric family."""
    metrics = sorted({str(row["metric"]) for row in results})
    for metric in metrics:
        idxs = [i for i, row in enumerate(results) if row["metric"] == metric]
        pvals = np.array(
            [float(results[i]["p_two_sided"]) for i in idxs], dtype=float
        )
        adjusted = benjamini_hochberg(pvals)
        for local_idx, global_idx in enumerate(idxs):
            p_adj = float(adjusted[local_idx])
            results[global_idx]["p_bh"] = p_adj
            results[global_idx]["significant_bh"] = p_adj < alpha


# ---------------------------------------------------------------------------
# Model discovery and resolution
# ---------------------------------------------------------------------------


def discover_models(
    dl_dir: Path, sk_dir: Path
) -> dict[str, Path]:
    """
    Discover all models with test prediction files from both directories.

    Returns {canonical_name: csv_path} mapping.
    """
    found: dict[str, Path] = {}

    # DL models: {model}/{model}_test_predictions.csv
    if dl_dir.exists():
        for d in dl_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                test_csv = d / f"{d.name}_test_predictions.csv"
                if test_csv.exists():
                    found[d.name] = test_csv

    # sklearn models: {model}/testPE.csv
    if sk_dir.exists():
        for d in sk_dir.iterdir():
            if d.is_dir() and (d / "testPE.csv").exists():
                found[d.name] = d / "testPE.csv"

    return found


def resolve_models(
    requested: list[str], available: dict[str, Path]
) -> list[str]:
    """Resolve user-supplied model tokens to canonical names."""
    available_names = set(available.keys())
    resolved: list[str] = []
    unknown: list[str] = []

    for token in requested:
        # Check alias map first
        canonical = ALIAS_MAP.get(token.lower(), token)
        if canonical in available_names:
            if canonical not in resolved:
                resolved.append(canonical)
            continue
        # Try appending _rmse (sklearn convention)
        if f"{canonical}_rmse" in available_names:
            canonical = f"{canonical}_rmse"
            if canonical not in resolved:
                resolved.append(canonical)
            continue
        # Direct match
        if token in available_names:
            if token not in resolved:
                resolved.append(token)
            continue
        unknown.append(token)

    if unknown:
        raise ValueError(
            "Unknown models: "
            + ", ".join(unknown)
            + f". Available: {sorted(available_names)}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired bootstrap resampling test (Koehn 2004) for "
            "all-vs-all model comparison on held-out test set."
        )
    )
    parser.add_argument(
        "--dl-dir", type=Path, default=DL_DIR_DEFAULT,
        help=f"Path to extract_deep_learning/ (default: {DL_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--sk-dir", type=Path, default=SK_DIR_DEFAULT,
        help=f"Path to extract_scikitlearn/ (default: {SK_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--models", type=str, default=DEFAULT_MODELS,
        help=f"Comma-separated model names (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help=f"Metrics to test (default: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--B", type=int, default=DEFAULT_B,
        help=f"Number of bootstrap replicates (default: {DEFAULT_B})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Significance level for BH correction (default: {DEFAULT_ALPHA})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_CSV_DEFAULT,
        help=f"Path for output CSV (default: {OUTPUT_CSV_DEFAULT})",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Discover available models
    # ------------------------------------------------------------------
    print("\n=== Step 1: Discovering models ===")
    available = discover_models(args.dl_dir, args.sk_dir)
    print(f"  Total models with test predictions: {len(available)}")

    try:
        models = resolve_models(parse_model_arg(args.models), available)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Models resolved ({len(models)}): {models}")

    # Validate requested metrics
    for m in args.metrics:
        if m not in METRIC_FN:
            print(
                f"ERROR: unknown metric '{m}'. "
                f"Available: {sorted(METRIC_FN.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load predictions
    # ------------------------------------------------------------------
    print("\n=== Step 2: Loading test predictions ===")
    all_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_actual: np.ndarray | None = None

    for model in models:
        csv_path = available[model]
        y_true, y_pred = load_predictions(csv_path)
        all_preds[model] = (y_true, y_pred)
        if ref_actual is None:
            ref_actual = y_true
        else:
            if not np.allclose(ref_actual, y_true, atol=1e-6):
                warn(f"Model '{model}' actual values differ from reference!")
        print(f"  {model:<25s}  N={len(y_true)}  ({csv_path.parent.parent.name})")

    assert ref_actual is not None, "No predictions loaded."
    N = len(ref_actual)
    print(f"\n  Shared test set: N={N}")

    # ------------------------------------------------------------------
    # 3. Generate all-vs-all pairs
    # ------------------------------------------------------------------
    pairs = list(itertools.combinations(models, 2))
    n_comparisons = len(args.metrics) * len(pairs)
    print(f"\n=== Step 3: Running paired bootstrap ===")
    print(f"  {len(models)} models → {len(pairs)} pairs × {len(args.metrics)} metrics = {n_comparisons} comparisons")
    print(f"  B={args.B}, seed={args.seed}")

    results: list[dict[str, object]] = []
    done = 0

    for metric_name in args.metrics:
        metric_fn = METRIC_FN[metric_name]
        for model_a, model_b in pairs:
            _, y_pred_a = all_preds[model_a]
            _, y_pred_b = all_preds[model_b]

            done += 1
            print(
                f"  [{done}/{n_comparisons}] {metric_name}: "
                f"{model_a} vs {model_b} ...",
                end="",
                flush=True,
            )

            bt = paired_bootstrap(
                y_true=ref_actual,
                y_pred_a=y_pred_a,
                y_pred_b=y_pred_b,
                metric_fn=metric_fn,
                B=args.B,
                seed=args.seed,
            )

            direction = "A < B" if bt["obs_diff"] > 0 else ("A > B" if bt["obs_diff"] < 0 else "A = B")

            row = {
                "metric": metric_name,
                "model_a": model_a,
                "model_b": model_b,
                "n_test": N,
                "B": args.B,
                "score_a": bt["obs_a"],
                "score_b": bt["obs_b"],
                "observed_diff_b_minus_a": bt["obs_diff"],
                "direction": direction,
                "wins_a": bt["wins_a"],
                "wins_b": bt["wins_b"],
                "wins_tie": bt["wins_tie"],
                "p_hat_a_better": bt["p_hat_a_better"],
                "p_two_sided": bt["p_two_sided"],
                "p_bh": "",
                "significant_bh": "",
            }
            results.append(row)

            direction = "A<B" if bt["obs_diff"] > 0 else "A>B"
            print(
                f"  A={bt['obs_a']:.4f}  B={bt['obs_b']:.4f}  "
                f"({direction})  p={bt['p_two_sided']:.4f}"
            )

    if not results:
        print("ERROR: no valid comparisons produced.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Apply BH correction
    # ------------------------------------------------------------------
    print(f"\n=== Step 4: Benjamini-Hochberg correction (alpha={args.alpha}) ===")
    apply_bh_by_metric(results, alpha=args.alpha)

    # ------------------------------------------------------------------
    # 5. Write output
    # ------------------------------------------------------------------
    print(f"\n=== Step 5: Writing output ===")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"  Wrote {len(results)} rows to: {args.output}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(
        f"  {'Metric':<12s} {'Model A':<25s} {'Model B':<25s} "
        f"{'Score_A':>8s} {'Score_B':>8s} {'Diff':>8s} "
        f"{'p_raw':>8s} {'p_BH':>8s} {'Sig':>5s}"
    )
    print("  " + "-" * 130)
    for row in results:
        sig_str = "YES" if row["significant_bh"] else "no"
        print(
            f"  {row['metric']:<12s} {row['model_a']:<25s} {row['model_b']:<25s} "
            f"{row['score_a']:>8.4f} {row['score_b']:>8.4f} "
            f"{row['observed_diff_b_minus_a']:>8.4f} "
            f"{row['p_two_sided']:>8.4f} {row['p_bh']:>8.4f} {sig_str:>5s}"
        )

    for metric_name in sorted({str(row["metric"]) for row in results}):
        subset = [r for r in results if r["metric"] == metric_name]
        n_sig = sum(bool(r["significant_bh"]) for r in subset)
        print(
            f"\n  Metric={metric_name}: "
            f"{n_sig}/{len(subset)} significant at alpha={args.alpha} (BH)."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
