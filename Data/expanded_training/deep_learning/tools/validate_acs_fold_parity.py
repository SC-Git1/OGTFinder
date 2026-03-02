#!/usr/bin/env python3
"""Validate that manifest folds match ACS split procedure exactly."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

DEEP_LEARNING_DIR = Path(__file__).resolve().parents[1]
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(DEEP_LEARNING_DIR))


def _load_generator_module():
    generator_path = DEEP_LEARNING_DIR / "tools" / "generate_acs_split_manifest.py"
    spec = importlib.util.spec_from_file_location("generate_acs_split_manifest", str(generator_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compare_stage(manifest: dict, regenerated: dict, stage_key: str) -> None:
    expected = manifest[stage_key]
    actual = regenerated[stage_key]

    if len(expected) != len(actual):
        raise AssertionError(f"Stage '{stage_key}' fold count mismatch: expected={len(expected)}, actual={len(actual)}")

    for fold_idx, (exp_fold, got_fold) in enumerate(zip(expected, actual)):
        if exp_fold["train_row_ids"] != got_fold["train_row_ids"]:
            raise AssertionError(f"Stage '{stage_key}' fold {fold_idx} train_row_ids mismatch")
        if exp_fold["val_row_ids"] != got_fold["val_row_ids"]:
            raise AssertionError(f"Stage '{stage_key}' fold {fold_idx} val_row_ids mismatch")


def _run_lightweight_validation(train_path: Path, manifest_path: Path) -> None:
    generator = _load_generator_module()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_manifest_path = Path(tmpdir) / "regen_manifest.json"
        regenerated = generator.generate_manifest(
            train_path=train_path,
            manifest_path=tmp_manifest_path,
            feature_columns=manifest["feature_columns"],
            target_col=manifest["target_col"],
            group_col=manifest["group_col"],
            n_splits=int(manifest["n_splits"]),
            random_state=int(manifest["random_state"]),
        )

    if manifest["row_ids"] != regenerated["row_ids"]:
        raise AssertionError("Row identity list mismatch against manifest")

    _compare_stage(manifest, regenerated, "optuna_objective_folds")
    _compare_stage(manifest, regenerated, "cv_report_folds")


def _run_utils_validation(train_path: Path, manifest_path: Path) -> None:
    from config import FEATURE_COLUMNS, TARGET, GROUP_COLUMN  # noqa: E402
    from utils import (  # noqa: E402
        load_full_train_data_grouped,
        load_split_manifest,
        get_split_context,
        get_stage_folds,
    )

    manifest = load_split_manifest(str(manifest_path))

    _, _, X_raw, y_original, groups, _, _, _ = load_full_train_data_grouped(
        data_path=str(train_path),
        feature_columns=FEATURE_COLUMNS,
        target_col=TARGET,
        group_col=GROUP_COLUMN,
        group_nan_policy="fill_minus_one",
        split_protocol="acs_v1_exact",
        manifest_path=str(manifest_path),
        strict_dataset_match=True,
    )

    row_ids = get_split_context().get("row_ids")
    if row_ids is None:
        raise RuntimeError("Split context row_ids not available")

    for stage, key in (("optuna_objective", "optuna_objective_folds"), ("cv_report", "cv_report_folds")):
        expected = manifest[key]
        resolved = get_stage_folds(
            stage=stage,
            X=X_raw,
            y=y_original,
            groups=groups,
            n_splits=len(expected),
            manifest_path=str(manifest_path),
            protocol="acs_v1_exact",
            strict=True,
        )

        for fold_idx, ((train_idx, val_idx), expected_fold) in enumerate(zip(resolved, expected)):
            got_train = [row_ids[int(i)] for i in train_idx]
            got_val = [row_ids[int(i)] for i in val_idx]
            if got_train != expected_fold["train_row_ids"]:
                raise AssertionError(f"Stage '{stage}' fold {fold_idx} train mismatch")
            if got_val != expected_fold["val_row_ids"]:
                raise AssertionError(f"Stage '{stage}' fold {fold_idx} val mismatch")


def parse_args() -> argparse.Namespace:
    dl_root = Path(__file__).resolve().parents[1]
    repo_root = dl_root.parent

    parser = argparse.ArgumentParser(description="Validate ACS fold parity")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=repo_root / "ACS_draft" / "ogt_final_runs" / "data" / "train_genus.csv",
        help="Dataset path to validate against manifest",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=dl_root / "splits" / "acs_groupkfold_v1.json",
        help="ACS split manifest JSON path",
    )
    parser.add_argument(
        "--force-lightweight",
        action="store_true",
        help="Use CSV/lightweight validation path even if utils deps are available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.force_lightweight:
        _run_lightweight_validation(args.train_path, args.manifest_path)
        mode = "lightweight"
    else:
        try:
            _run_utils_validation(args.train_path, args.manifest_path)
            mode = "utils"
        except ModuleNotFoundError:
            _run_lightweight_validation(args.train_path, args.manifest_path)
            mode = "lightweight"

    print("ACS fold parity check passed for both stages.")
    print(f"Mode: {mode}")
    print(f"Manifest: {args.manifest_path}")
    print(f"Dataset: {args.train_path}")


if __name__ == "__main__":
    main()
