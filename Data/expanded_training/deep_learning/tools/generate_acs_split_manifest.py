#!/usr/bin/env python3
"""Generate ACS GroupKFold manifest for exact fold parity in deep_learning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEEP_LEARNING_DIR = Path(__file__).resolve().parents[1]
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(DEEP_LEARNING_DIR))

from config import FEATURE_COLUMNS, TARGET, GROUP_COLUMN, N_CV, RANDOM_STATE  # noqa: E402


def _compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as infile:
        while True:
            chunk = infile.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_naish(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        return s in {"", "nan", "none", "null"}
    return False


def _canonical_scalar(value) -> str:
    if _is_naish(value):
        return "__NA__"

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")

    if isinstance(value, str):
        if value.lstrip("-").isdigit():
            return str(int(value))
        try:
            as_float = float(value)
            if np.isfinite(as_float):
                return format(float(as_float), ".17g")
        except ValueError:
            pass
        return value

    return str(value)


def _parse_group_value(raw_value):
    if _is_naish(raw_value):
        return -1

    if isinstance(raw_value, str):
        raw = raw_value.strip()
        if raw.lstrip("-").isdigit():
            return int(raw)
        try:
            as_float = float(raw)
            if np.isfinite(as_float):
                if as_float.is_integer():
                    return int(as_float)
                return float(as_float)
        except ValueError:
            return raw

    return raw_value


def _build_row_identity(X_rows: list[list], y_values: list, groups: list) -> list[str]:
    row_hashes = []
    for idx in range(len(X_rows)):
        row_values = [_canonical_scalar(v) for v in X_rows[idx]]
        payload = "|".join(
            [
                f"x={','.join(row_values)}",
                f"y={_canonical_scalar(y_values[idx])}",
                f"g={_canonical_scalar(groups[idx])}",
            ]
        )
        row_hashes.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())

    occurrence = {}
    row_ids = []
    for row_hash in row_hashes:
        occ = occurrence.get(row_hash, 0)
        row_ids.append(f"{row_hash}:{occ}")
        occurrence[row_hash] = occ + 1
    return row_ids


def _groupkfold_indices(groups: list, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups_array = np.asarray(groups, dtype=object)
    unique_groups, groups_inverse = np.unique(groups_array, return_inverse=True)

    if n_splits > len(unique_groups):
        raise ValueError(
            f"Cannot have n_splits={n_splits} greater than number of unique groups={len(unique_groups)}"
        )

    n_samples_per_group = np.bincount(groups_inverse)
    group_order = np.argsort(n_samples_per_group)[::-1]

    n_samples_per_fold = np.zeros(n_splits)
    group_to_fold = np.zeros(len(unique_groups), dtype=int)

    for group_idx in group_order:
        fold_idx = int(np.argmin(n_samples_per_fold))
        n_samples_per_fold[fold_idx] += n_samples_per_group[group_idx]
        group_to_fold[group_idx] = fold_idx

    fold_of_sample = group_to_fold[groups_inverse]

    folds = []
    for fold_idx in range(n_splits):
        val_idx = np.where(fold_of_sample == fold_idx)[0]
        train_idx = np.where(fold_of_sample != fold_idx)[0]
        folds.append((train_idx.astype(np.int64), val_idx.astype(np.int64)))
    return folds


def _fold_row_ids(row_ids: list[str], train_idx: np.ndarray, val_idx: np.ndarray, fold: int) -> dict:
    return {
        "fold": int(fold),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_row_ids": [row_ids[int(i)] for i in train_idx],
        "val_row_ids": [row_ids[int(i)] for i in val_idx],
    }


def _shuffle_indices(n_rows: int, random_state: int) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    return rng.permutation(n_rows)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    return columns, rows


def generate_manifest(
    train_path: Path,
    manifest_path: Path,
    feature_columns: list[str],
    target_col: str,
    group_col: str,
    n_splits: int,
    random_state: int,
) -> dict:
    columns, rows = _read_csv_rows(train_path)

    missing_features = [c for c in feature_columns if c not in columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in reference dataset: {missing_features}")
    if target_col not in columns:
        raise ValueError(f"Target column '{target_col}' not found in {train_path}")
    if group_col not in columns:
        raise ValueError(f"Group column '{group_col}' not found in {train_path}")

    X_rows = []
    y_values = []
    groups = []
    for row in rows:
        X_rows.append([row[col] for col in feature_columns])
        y_values.append(row[target_col])
        groups.append(_parse_group_value(row[group_col]))

    row_ids = _build_row_identity(X_rows, y_values, groups)
    if len(row_ids) != len(rows):
        raise ValueError("Row identity length mismatch while building manifest")

    objective_splits = _groupkfold_indices(groups=groups, n_splits=n_splits)
    objective_folds = [
        _fold_row_ids(row_ids=row_ids, train_idx=train_idx, val_idx=val_idx, fold=fold_idx)
        for fold_idx, (train_idx, val_idx) in enumerate(objective_splits)
    ]

    shuffled_order = _shuffle_indices(len(rows), random_state=random_state)
    shuffled_groups = [groups[int(i)] for i in shuffled_order]
    report_splits = _groupkfold_indices(groups=shuffled_groups, n_splits=n_splits)

    cv_report_folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(report_splits):
        train_orig_idx = shuffled_order[train_idx]
        val_orig_idx = shuffled_order[val_idx]
        cv_report_folds.append(
            _fold_row_ids(row_ids=row_ids, train_idx=train_orig_idx, val_idx=val_orig_idx, fold=fold_idx)
        )

    manifest = {
        "schema_version": 1,
        "split_protocol": "acs_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_train_path": str(train_path),
        "group_nan_policy": "fill_minus_one",
        "random_state": int(random_state),
        "n_splits": int(n_splits),
        "target_col": target_col,
        "group_col": group_col,
        "feature_columns": list(feature_columns),
        "dataset_fingerprint": {
            "sha256": _compute_file_sha256(train_path),
            "row_count": int(len(rows)),
            "columns": columns,
            "target_col": target_col,
            "group_col": group_col,
        },
        "row_ids": row_ids,
        "optuna_objective_folds": objective_folds,
        "cv_report_folds": cv_report_folds,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2)

    return manifest


def parse_args() -> argparse.Namespace:
    dl_root = Path(__file__).resolve().parents[1]
    repo_root = dl_root.parent

    parser = argparse.ArgumentParser(description="Generate ACS exact split manifest")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=repo_root / "ACS_draft" / "ogt_final_runs" / "data" / "train_genus.csv",
        help="Path to ACS reference training CSV",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=dl_root / "splits" / "acs_groupkfold_v1.json",
        help="Output path for split manifest JSON",
    )
    parser.add_argument("--target-col", type=str, default=TARGET)
    parser.add_argument("--group-col", type=str, default=GROUP_COLUMN)
    parser.add_argument("--n-splits", type=int, default=N_CV)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = generate_manifest(
        train_path=args.train_path,
        manifest_path=args.manifest_path,
        feature_columns=FEATURE_COLUMNS,
        target_col=args.target_col,
        group_col=args.group_col,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    print(f"Manifest written: {args.manifest_path}")
    print(f"Rows: {manifest['dataset_fingerprint']['row_count']}")
    print(f"Objective folds: {len(manifest['optuna_objective_folds'])}")
    print(f"CV report folds: {len(manifest['cv_report_folds'])}")


if __name__ == "__main__":
    main()
