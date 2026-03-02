"""I/O and logging helpers."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import DEEP_LEARNING_PERF_COLUMNS

TRACKING_METRIC_COLUMNS = DEEP_LEARNING_PERF_COLUMNS[3:]
PERFOLD_PERFORMANCE_COLUMNS = ["model_name", "experiment_name", "fold", "n_train", "n_val"] + TRACKING_METRIC_COLUMNS


def setup_logging(run_dir: Path | str, run_name: str) -> logging.Logger:
    """Configure file+console logger."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{run_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(run_name)


def save_json(path: Path | str, obj: dict[str, Any]) -> None:
    """Save dictionary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(obj, outfile, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    """JSON serializer fallback for NumPy and Path-like objects."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_predictions_csv(path: Path | str, y_true, y_pred) -> None:
    """Save prediction CSV with actual/predicted columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual": y_true, "predicted": y_pred}).to_csv(path, index=False)


def save_cv_summary(path: Path | str, fold_rows: list[dict[str, Any]], aggregate_row: dict[str, Any]) -> None:
    """Save fold-level CV rows plus aggregate row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(fold_rows) + [aggregate_row]
    pd.DataFrame(rows).to_csv(path, index=False)


def append_model_performance_csv(path: Path | str, row_dict: dict[str, Any]) -> None:
    """Append deep_learning-style row to model performance CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=DEEP_LEARNING_PERF_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k) for k in DEEP_LEARNING_PERF_COLUMNS})


def save_perfold_performance_csv(
    path: Path | str,
    model_name: str,
    experiment_name: str,
    fold_rows: list[dict[str, Any]],
) -> None:
    """Append one row per fold to global per-fold performance CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fold_rows:
        return

    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=PERFOLD_PERFORMANCE_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for fold_row in fold_rows:
            row = {
                "model_name": model_name,
                "experiment_name": experiment_name,
                "fold": fold_row.get("fold"),
                "n_train": fold_row.get("n_train"),
                "n_val": fold_row.get("n_val"),
            }
            for metric_key in TRACKING_METRIC_COLUMNS:
                row[metric_key] = fold_row.get(metric_key)
            writer.writerow(row)


def save_cv_performance_csv(
    path: Path | str,
    model_name: str,
    experiment_name: str,
    mean_metrics: dict[str, Any],
    std_metrics: dict[str, Any],
) -> None:
    """Append CV mean/std row to global CV performance CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model_name", "experiment_name"]
    for metric_key in TRACKING_METRIC_COLUMNS:
        fieldnames.extend([f"{metric_key}_mean", f"{metric_key}_std"])

    row = {"model_name": model_name, "experiment_name": experiment_name}
    for metric_key in TRACKING_METRIC_COLUMNS:
        row[f"{metric_key}_mean"] = mean_metrics.get(metric_key)
        row[f"{metric_key}_std"] = std_metrics.get(metric_key)

    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
