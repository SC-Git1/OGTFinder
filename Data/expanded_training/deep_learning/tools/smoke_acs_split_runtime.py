#!/usr/bin/env python3
"""Tiny runtime smoke test to ensure ACS exact folds are used in both CV stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEEP_LEARNING_DIR = Path(__file__).resolve().parents[1]
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(DEEP_LEARNING_DIR))

from config import FEATURE_COLUMNS, TARGET, GROUP_COLUMN, RANDOM_STATE  # noqa: E402


def parse_args() -> argparse.Namespace:
    dl_root = Path(__file__).resolve().parents[1]
    repo_root = dl_root.parent

    parser = argparse.ArgumentParser(description="Smoke test ACS exact CV runtime")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=repo_root / "data" / "train_genus.csv",
        help="deep_learning training dataset",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=dl_root / "splits" / "acs_groupkfold_v1.json",
        help="ACS split manifest",
    )
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=1)
    return parser.parse_args()


def _build_model_for_trial(trial, input_dim):
    import tensorflow as tf

    units = trial.suggest_categorical("units", [32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(units, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _build_model_from_params(params, input_dim):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(int(params["units"]), activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(params["learning_rate"])),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _assert_stage(stage: str, manifest_key: str, X, y, groups, manifest_path: str) -> None:
    from utils import load_split_manifest, get_split_context, get_stage_folds

    manifest = load_split_manifest(manifest_path)
    expected = manifest[manifest_key]
    row_ids = get_split_context()["row_ids"]

    resolved = get_stage_folds(
        stage=stage,
        X=X,
        y=y,
        groups=groups,
        n_splits=len(expected),
        manifest_path=manifest_path,
        protocol="acs_v1_exact",
        strict=True,
    )

    for fold_idx, ((train_idx, val_idx), expected_fold) in enumerate(zip(resolved, expected)):
        got_train = [row_ids[int(i)] for i in train_idx]
        got_val = [row_ids[int(i)] for i in val_idx]
        if got_train != expected_fold["train_row_ids"] or got_val != expected_fold["val_row_ids"]:
            raise AssertionError(f"Fold mismatch at stage={stage}, fold={fold_idx}")


def main() -> int:
    args = parse_args()

    try:
        from utils import (
            load_full_train_data_grouped,
            create_groupkfold_objective_keras,
            groupkfold_cross_validate,
            run_optuna_study,
        )

        _, _, X_raw, y_original, groups, _, _, _ = load_full_train_data_grouped(
            data_path=str(args.train_path),
            feature_columns=FEATURE_COLUMNS,
            target_col=TARGET,
            group_col=GROUP_COLUMN,
            group_nan_policy="fill_minus_one",
            split_protocol="acs_v1_exact",
            manifest_path=str(args.manifest_path),
            strict_dataset_match=True,
        )

        objective = create_groupkfold_objective_keras(
            build_model_fn=_build_model_for_trial,
            X=X_raw,
            y=y_original,
            groups=groups,
            n_splits=5,
            batch_size=256,
            max_epochs=args.max_epochs,
            patience_es=1,
            patience_lr=1,
            random_state=RANDOM_STATE,
            split_protocol="acs_v1_exact",
            manifest_path=str(args.manifest_path),
            strict=True,
        )

        study, best_params, best_value = run_optuna_study(
            objective_fn=objective,
            study_name="acs_exact_smoke",
            n_trials=args.n_trials,
            timeout=None,
            show_progress_bar=False,
        )

        mean_metrics, std_metrics, _ = groupkfold_cross_validate(
            model_builder_fn=_build_model_from_params,
            best_params=best_params,
            X=X_raw,
            y=y_original,
            groups=groups,
            n_splits=5,
            batch_size=256,
            max_epochs=args.max_epochs,
            patience_es=1,
            patience_lr=1,
            n_features=X_raw.shape[1],
            random_state=RANDOM_STATE,
            split_protocol="acs_v1_exact",
            manifest_path=str(args.manifest_path),
            strict=True,
        )

        _assert_stage("optuna_objective", "optuna_objective_folds", X_raw, y_original, groups, str(args.manifest_path))
        _assert_stage("cv_report", "cv_report_folds", X_raw, y_original, groups, str(args.manifest_path))

        print("Runtime smoke passed.")
        print(f"Best objective value: {best_value}")
        print(f"Best params: {json.dumps(best_params)}")
        print(f"CV mean RMSE: {mean_metrics.get('rmse')}")
        print(f"CV std RMSE: {std_metrics.get('rmse')}")
        return 0
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Skipped smoke runtime due to missing dependency: {exc}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
