"""Data loading and preprocessing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS


def load_train_test(train_path: Path | str, test_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test data from CSV files."""
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing file not found: {test_path}")

    return pd.read_csv(train_path), pd.read_csv(test_path)


def validate_and_select_features(df: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Validate strict feature list and return selected features in fixed order."""
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    selected = df[feature_columns]
    non_numeric = [c for c in selected.columns if not pd.api.types.is_numeric_dtype(selected[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

    if selected.isna().any().any():
        nan_cols = selected.columns[selected.isna().any()].tolist()
        raise ValueError(f"NaN values found in feature columns: {nan_cols}")

    return selected


def extract_X_y_groups(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str,
    group_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features, target, and groups as NumPy arrays."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column '{group_col}'")

    X = validate_and_select_features(df, feature_columns).to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    groups_series = df[group_col].fillna(-1)
    groups = groups_series.to_numpy()

    if np.isnan(y).any():
        raise ValueError(f"NaN values found in target column '{target_col}'")
    return X, y, groups


def load_weights_exact(weights_path: Path | str) -> dict[float, float]:
    """Load exact-value sample weights from JSON."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    with open(weights_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    return {float(k): float(v) for k, v in data.items()}


def map_weights_exact(y_values: Sequence[float], weights_dict: dict[float, float]) -> np.ndarray:
    """
    Map target values to sample weights by exact float key match.

    Raises:
        ValueError: if any target value has no exact corresponding weight key.
    """
    y_arr = np.asarray(y_values, dtype=float)
    mapped = np.empty_like(y_arr, dtype=float)
    missing = []

    for idx, value in enumerate(y_arr):
        key = float(value)
        if key not in weights_dict:
            missing.append(key)
            mapped[idx] = np.nan
        else:
            mapped[idx] = weights_dict[key]

    if missing:
        missing_unique = sorted(set(missing))
        raise ValueError(
            f"Found {len(missing_unique)} unmatched target values in weight mapping. "
            f"First values: {missing_unique[:20]}"
        )

    return mapped


def map_weights_exact_or_nearest(
    y_values: Sequence[float],
    weights_dict: dict[float, float],
    *,
    allow_nearest: bool,
) -> np.ndarray:
    """
    Map target values to sample weights with exact-first behavior.

    Args:
        y_values: target values to map.
        weights_dict: dictionary of target->weight mappings.
        allow_nearest: if True, use nearest available target key when exact key is missing.

    Returns:
        Sample-weight array.

    Raises:
        ValueError: when exact key is missing and allow_nearest is False.
    """
    y_arr = np.asarray(y_values, dtype=float)
    mapped = np.empty_like(y_arr, dtype=float)
    keys = np.array(sorted(weights_dict.keys()), dtype=float)

    missing = []
    for idx, value in enumerate(y_arr):
        key = float(value)
        if key in weights_dict:
            mapped[idx] = weights_dict[key]
            continue

        if not allow_nearest:
            missing.append(key)
            mapped[idx] = np.nan
            continue

        nearest_idx = int(np.abs(keys - key).argmin())
        mapped[idx] = weights_dict[float(keys[nearest_idx])]

    if missing:
        missing_unique = sorted(set(missing))
        raise ValueError(
            f"Found {len(missing_unique)} unmatched target values in exact weight mapping. "
            f"First values: {missing_unique[:20]}"
        )

    return mapped


def ensure_dir(path: Path | str) -> Path:
    """Create a directory and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
