"""Metrics utilities with deep_learning-style output schema."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DEEP_LEARNING_PERF_COLUMNS


def compute_weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, sample_weights: np.ndarray) -> float:
    """Compute weighted RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=w)))


def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Compute adjusted R²."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n <= n_features + 1:
        return float(r2)
    return float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))


def _bin_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, n_features: int) -> dict[str, Any]:
    """Compute metrics for a masked bin."""
    count = int(mask.sum())
    if count <= 1:
        return {"rmse": None, "mae": None, "r2": None, "adj_r2": None, "n": count}

    y_t = y_true[mask]
    y_p = y_pred[mask]
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "mae": float(mean_absolute_error(y_t, y_p)),
        "r2": float(r2_score(y_t, y_p)),
        "adj_r2": float(adjusted_r2_score(y_t, y_p, n_features)),
        "n": count,
    }


def compute_binned_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict[str, Any]:
    """Compute top/bottom percentile metrics for 5, 10, 20."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out: dict[str, Any] = {}
    for pct in (5, 10, 20):
        bottom_threshold = np.percentile(y_true, pct)
        top_threshold = np.percentile(y_true, 100 - pct)
        bottom_mask = y_true <= bottom_threshold
        top_mask = y_true >= top_threshold

        bottom = _bin_metrics(y_true, y_pred, bottom_mask, n_features)
        top = _bin_metrics(y_true, y_pred, top_mask, n_features)

        out[f"rmse_bottom_{pct}"] = bottom["rmse"]
        out[f"mae_bottom_{pct}"] = bottom["mae"]
        out[f"r2_bottom_{pct}"] = bottom["r2"]
        out[f"adj_r2_bottom_{pct}"] = bottom["adj_r2"]
        out[f"n_bottom_{pct}"] = bottom["n"]

        out[f"rmse_top_{pct}"] = top["rmse"]
        out[f"mae_top_{pct}"] = top["mae"]
        out[f"r2_top_{pct}"] = top["r2"]
        out[f"adj_r2_top_{pct}"] = top["adj_r2"]
        out[f"n_top_{pct}"] = top["n"]

    return out


def compute_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute aggregate plus binned metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result: dict[str, Any] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "adj_r2": float(adjusted_r2_score(y_true, y_pred, n_features)),
        "weighted_rmse": None,
    }

    if sample_weights is not None:
        result["weighted_rmse"] = compute_weighted_rmse(y_true, y_pred, sample_weights)

    result.update(compute_binned_metrics(y_true, y_pred, n_features))
    return result


def assemble_deep_learning_style_row(
    *,
    model_name: str,
    experiment_name: str,
    dataset_type: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a deep_learning-style performance row with fixed column order."""
    row = {
        "model_name": model_name,
        "experiment_name": experiment_name,
        "dataset_type": dataset_type,
    }
    row.update(metrics)

    ordered = {}
    for key in DEEP_LEARNING_PERF_COLUMNS:
        ordered[key] = row.get(key)
    return ordered
