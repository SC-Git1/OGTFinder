"""Shared runner for DL-based stacking ensembles (from-scratch training).

Trains each DL base model from scratch using best_params.json hyperparameters
from deep_learning/out/, generates leak-free OOF predictions via GroupKFold CV,
then tunes a meta-learner with Optuna and evaluates the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    ALL_BEST_PARAMS_FILE,
    BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DL_PARAMS_ROOT,
    EXPERIMENT_NAME_DEFAULT,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    MAX_EPOCHS,
    MODEL_CV_PERFORMANCE_FILE,
    MODEL_PERFOLD_PERFORMANCE_FILE,
    MODEL_PERFORMANCE_FILE,
    N_CV,
    OUTPUT_ROOT,
    PATIENCE_ES,
    PATIENCE_LR,
    RANDOM_STATE,
    SEED,
    TARGET,
    TESTING_FILE,
    TRAINING_FILE,
    WEIGHTS_FILE,
)
from dl_model_registry import MODEL_REGISTRY
from dl_utils import (
    check_gpu,
    create_callbacks,
    get_sample_weights_array,
    get_tqdm_keras_callback,
    groupkfold_cross_validate,
    load_and_preprocess_test,
    load_full_train_data_grouped,
    load_sample_weights,
    set_global_seeds,
)
from utils_data import (
    extract_X_y_groups,
    load_train_test,
    load_weights_exact,
    map_weights_exact,
    map_weights_exact_or_nearest,
)
from utils_io import (
    append_model_performance_csv,
    save_cv_performance_csv,
    save_cv_summary,
    save_json,
    save_perfold_performance_csv,
    save_predictions_csv,
    setup_logging,
)
from utils_metrics import assemble_deep_learning_style_row, compute_overall_metrics
from utils_plots import plot_all_diagnostics, plot_train_diagnostics
from utils_tuning import tune_model

# Meta-learner trial count
META_TRIALS_DEFAULT = 300


@dataclass
class DLStackConfig:
    """Configuration for a deep-learning-based stacking ensemble."""

    stack_name: str
    """Short name used for output directory and file prefixes."""

    base_model_keys: list[str]
    """List of deep_learning model directory names (e.g. 'wide_deep', 'vime')."""

    meta_learner_key: str
    """Key into ``dl_meta_registry.META_REGISTRY``."""

    description: str = ""
    """Human-readable description of this stacking plan."""


def parse_dl_stack_args(description: str) -> argparse.Namespace:
    """Parse CLI arguments for a DL stacking ensemble script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_DATA_DIR / TRAINING_FILE)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_DATA_DIR / TESTING_FILE)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_DATA_DIR / WEIGHTS_FILE)
    parser.add_argument("--dl-params-root", type=Path, default=DL_PARAMS_ROOT,
                        help="Root directory containing deep_learning best_params.json files.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-cv", type=int, default=N_CV)
    parser.add_argument("--meta-trials", type=int, default=META_TRIALS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience-es", type=int, default=PATIENCE_ES)
    parser.add_argument("--patience-lr", type=int, default=PATIENCE_LR)
    parser.add_argument("--experiment-name", type=str, default=EXPERIMENT_NAME_DEFAULT)
    parser.add_argument("--n-jobs", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Base model training helpers
# ---------------------------------------------------------------------------

_all_params_cache: dict | None = None


def _load_best_params(dl_params_root: Path, model_key: str) -> dict:
    """Load best hyperparameters for a base model.

    Lookup order:
    1. Consolidated all_best_params.json (self-contained, no dependency on
       deep_learning/out/ — works on the cluster without syncing).
    2. Per-directory fallback: dl_params_root/{model_key}/{model_key}_best_params.json.
    """
    global _all_params_cache

    # Tier 1: consolidated file (loaded once, cached)
    if _all_params_cache is None and ALL_BEST_PARAMS_FILE.exists():
        with open(ALL_BEST_PARAMS_FILE) as f:
            _all_params_cache = json.load(f)

    if _all_params_cache is not None and model_key in _all_params_cache:
        return _all_params_cache[model_key]

    # Tier 2: per-directory fallback (original behaviour)
    params_path = dl_params_root / model_key / f"{model_key}_best_params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"best_params not found for '{model_key}'.\n"
            f"  Consolidated: {ALL_BEST_PARAMS_FILE} "
            f"({'found' if ALL_BEST_PARAMS_FILE.exists() else 'missing'})\n"
            f"  Per-directory: {params_path}\n"
            "Run collect_best_params.py to generate the consolidated file, "
            "or ensure deep_learning/out/ is accessible."
        )
    with open(params_path) as f:
        return json.load(f)


def _train_base_model_oof(
    model_key: str,
    best_params: dict,
    X_raw: np.ndarray,
    y_original: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict, dict, list]:
    """Train base model with GroupKFold CV and return OOF predictions.

    Returns:
        (oof_predictions_sorted, mean_metrics, std_metrics, all_fold_metrics)
    """
    from tensorflow import keras

    build_fn = MODEL_REGISTRY[model_key]

    mean_metrics, std_metrics, all_fold_metrics, oof_predictions = groupkfold_cross_validate(
        model_builder_fn=build_fn,
        best_params=best_params,
        X=X_raw,
        y=y_original,
        groups=groups,
        sample_weights=sample_weights,
        n_splits=args.n_cv,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience_es=args.patience_es,
        patience_lr=args.patience_lr,
        n_features=X_raw.shape[1],
        random_state=RANDOM_STATE,
    )

    # Sort OOF predictions by original index to align with training data order
    oof_df = pd.DataFrame(oof_predictions)
    oof_df = oof_df.sort_values('indices').reset_index(drop=True)

    logger.info(
        "  OOF CV RMSE: %.4f ± %.4f, R²: %.4f ± %.4f",
        mean_metrics.get('rmse', 0), std_metrics.get('rmse', 0),
        mean_metrics.get('r2', 0), std_metrics.get('r2', 0),
    )

    return oof_df['predicted'].values, mean_metrics, std_metrics, all_fold_metrics


def _train_base_model_full(
    model_key: str,
    best_params: dict,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    target_scaler,
    sample_weights: np.ndarray | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> np.ndarray:
    """Train base model on full training data and predict on test set.

    Returns test predictions in original (unscaled) target space.
    """
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler

    build_fn = MODEL_REGISTRY[model_key]

    keras.backend.clear_session()
    set_global_seeds(RANDOM_STATE)

    model = build_fn(best_params, X_train_scaled.shape[1])
    callbacks = create_callbacks(args.patience_es, args.patience_lr)

    # Use last 10% as validation set for early stopping
    n = len(X_train_scaled)
    val_size = max(1, int(n * 0.1))
    X_tr = X_train_scaled[:-val_size]
    X_val = X_train_scaled[-val_size:]
    y_tr = y_train_scaled[:-val_size]
    y_val = y_train_scaled[-val_size:]
    sw_tr = sample_weights[:-val_size] if sample_weights is not None else None

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        sample_weight=sw_tr,
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    logger.info("  Full-data training complete, test predictions generated")
    return y_pred


# ---------------------------------------------------------------------------
# CV metric helpers
# ---------------------------------------------------------------------------

def _build_cv_aggregate(fold_rows: list[dict], label: str) -> dict:
    weighted = [r["weighted_rmse"] for r in fold_rows if r.get("weighted_rmse") is not None]
    rmses = [r["rmse"] for r in fold_rows if r.get("rmse") is not None]
    return {
        "fold": "mean",
        "model": label,
        "weighted_rmse": float(np.mean(weighted)) if weighted else None,
        "weighted_rmse_std": float(np.std(weighted)) if weighted else None,
        "rmse": float(np.mean(rmses)) if rmses else None,
        "rmse_std": float(np.std(rmses)) if rmses else None,
    }


def _build_cv_metrics_json(fold_rows: list[dict]) -> dict:
    metric_keys = [k for k in fold_rows[0] if k not in ("fold", "model", "n_train", "n_val")]
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for key in metric_keys:
        values = [r[key] for r in fold_rows if r.get(key) is not None]
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))
    return {"mean": mean_metrics, "std": std_metrics, "folds": fold_rows}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(config: DLStackConfig) -> None:
    """Execute a DL stacking ensemble plan end-to-end."""
    args = parse_dl_stack_args(config.description or f"DL stack: {config.stack_name}")

    run_dir = OUTPUT_ROOT / config.stack_name
    logger = setup_logging(run_dir, config.stack_name)
    logger.info("=" * 70)
    logger.info("DL Stacking Ensemble (from-scratch): %s", config.stack_name)
    logger.info("Base models: %s", config.base_model_keys)
    logger.info("Meta-learner: %s", config.meta_learner_key)
    logger.info("DL params root: %s", args.dl_params_root)
    logger.info("=" * 70)

    check_gpu()
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load training data (DL-style: returns both raw and scaled)
    # ------------------------------------------------------------------
    X_scaled, y_scaled, X_raw, y_original, groups, feature_scaler, target_scaler, feature_names = \
        load_full_train_data_grouped(
            str(args.train_path),
            feature_columns=FEATURE_COLUMNS,
            target_col=TARGET,
            group_col=GROUP_COLUMN,
        )

    # Load test data
    X_test_scaled, y_test_scaled, y_test_original = load_and_preprocess_test(
        str(args.test_path),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=FEATURE_COLUMNS,
        target_col=TARGET,
    )

    # Load sample weights
    sample_weights_dl = None
    if args.weights_path and args.weights_path.exists():
        weights_dict_dl = load_sample_weights(str(args.weights_path))
        sample_weights_dl = get_sample_weights_array(y_original, weights_dict_dl)
        logger.info("DL sample weights loaded, range: [%.4f, %.4f]",
                     sample_weights_dl.min(), sample_weights_dl.max())

    # Also load weights for meta-learner (ensemble_models style)
    weights_map = load_weights_exact(args.weights_path)
    sw_train_meta = map_weights_exact(y_original, weights_map)
    sw_test_meta = map_weights_exact_or_nearest(y_test_original, weights_map, allow_nearest=True)

    n_train = len(y_original)
    n_test = len(y_test_original)

    # ------------------------------------------------------------------
    # Train each base model and collect OOF + test predictions
    # ------------------------------------------------------------------
    train_meta_cols: list[np.ndarray] = []
    test_meta_cols: list[np.ndarray] = []
    base_cv_metrics: dict[str, dict] = {}

    for model_key in config.base_model_keys:
        logger.info("-" * 50)
        logger.info("Training base model: %s", model_key)

        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown base model '{model_key}'. "
                f"Available: {sorted(MODEL_REGISTRY.keys())}"
            )

        best_params = _load_best_params(args.dl_params_root, model_key)
        logger.info("  Loaded best_params: %s", best_params)

        # Phase 1: OOF predictions via GroupKFold CV
        logger.info("  Phase 1: GroupKFold CV for OOF predictions...")
        oof_preds, mean_m, std_m, fold_m = _train_base_model_oof(
            model_key, best_params, X_raw, y_original, groups,
            sample_weights_dl, args, logger
        )
        base_cv_metrics[model_key] = {'mean': mean_m, 'std': std_m}

        # Phase 2: Full-data training for test predictions
        logger.info("  Phase 2: Full-data training for test predictions...")
        test_preds = _train_base_model_full(
            model_key, best_params, X_scaled, y_scaled, X_test_scaled,
            target_scaler, sample_weights_dl, args, logger
        )

        train_meta_cols.append(oof_preds)
        test_meta_cols.append(test_preds)
        logger.info("  OOF: %d rows, Test: %d rows", len(oof_preds), len(test_preds))

    train_meta = np.column_stack(train_meta_cols)
    test_meta = np.column_stack(test_meta_cols)
    logger.info("=" * 50)
    logger.info("Meta feature matrix: train=%s, test=%s", train_meta.shape, test_meta.shape)

    # ------------------------------------------------------------------
    # Tune meta-learner with Optuna
    # ------------------------------------------------------------------
    from dl_meta_registry import META_REGISTRY

    meta_entry = META_REGISTRY[config.meta_learner_key]
    build_fn = meta_entry["build"]
    suggest_fn = meta_entry["suggest"]
    fallback_fn = meta_entry["fallback"]

    meta_label = f"{config.stack_name}_meta_{config.meta_learner_key}"
    logger.info("Tuning meta-learner: %s (%d trials)", meta_label, args.meta_trials)

    meta_params, meta_score, meta_folds, meta_backend = tune_model(
        model_name=meta_label,
        build_model_fn=lambda p: build_fn(p, seed=args.seed),
        suggest_params_fn=suggest_fn,
        fallback_param_distributions=fallback_fn(),
        X=train_meta,
        y=y_original,
        groups=groups,
        sample_weights=sw_train_meta,
        n_trials=args.meta_trials,
        n_cv=args.n_cv,
        seed=args.seed,
        logger=logger,
        require_optuna=True,
        n_jobs=args.n_jobs,
    )

    logger.info("Meta tuner backend: %s", meta_backend)
    logger.info("Best meta params: %s", meta_params)
    logger.info("Best CV weighted RMSE: %.6f", meta_score)

    # Fit final meta model on full OOF features
    final_meta = build_fn(meta_params, seed=args.seed)
    final_meta.fit(train_meta, y_original)
    y_pred_train_oof = final_meta.predict(train_meta)
    y_pred_test = final_meta.predict(test_meta)

    # ------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_meta, run_dir / f"{config.stack_name}_meta_model.pkl")
    save_json(run_dir / f"{config.stack_name}_meta_best_params.json", meta_params)

    pipeline_artifact = {
        "stack_name": config.stack_name,
        "base_model_keys": config.base_model_keys,
        "meta_learner_key": config.meta_learner_key,
        "meta_model": final_meta,
        "meta_params": meta_params,
        "seed": args.seed,
        "dl_params_root": str(args.dl_params_root),
        "base_cv_metrics": base_cv_metrics,
    }
    joblib.dump(pipeline_artifact, run_dir / f"{config.stack_name}_pipeline.pkl")
    logger.info("Pipeline saved to %s", run_dir / f"{config.stack_name}_pipeline.pkl")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = compute_overall_metrics(
        y_true=y_test_original,
        y_pred=y_pred_test,
        n_features=len(FEATURE_COLUMNS),
        sample_weights=sw_test_meta,
    )

    # Base model individual test metrics
    base_test_metrics = {}
    for i, model_key in enumerate(config.base_model_keys):
        base_test_metrics[model_key] = compute_overall_metrics(
            y_true=y_test_original,
            y_pred=test_meta[:, i],
            n_features=len(FEATURE_COLUMNS),
            sample_weights=sw_test_meta,
        )

    metrics_payload = {
        "model_name": config.stack_name,
        "experiment_name": args.experiment_name,
        "dataset_type": "test",
        "seed": args.seed,
        "n_cv": args.n_cv,
        "meta_trials": args.meta_trials,
        "feature_count": len(FEATURE_COLUMNS),
        "base_model_count": len(config.base_model_keys),
        "meta_learner": config.meta_learner_key,
        "meta_backend": meta_backend,
        "best_meta_params": meta_params,
        "best_cv_weighted_rmse": meta_score,
        "metrics": metrics,
        "base_models_test": base_test_metrics,
        "base_model_keys": config.base_model_keys,
        "base_cv_metrics": {k: v['mean'] for k, v in base_cv_metrics.items()},
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    save_json(run_dir / f"{config.stack_name}_metrics.json", metrics_payload)
    save_predictions_csv(
        run_dir / f"{config.stack_name}_train_predictions.csv", y_original, y_pred_train_oof
    )
    save_predictions_csv(
        run_dir / f"{config.stack_name}_test_predictions.csv", y_test_original, y_pred_test
    )

    # CV summary
    cv_rows = list(meta_folds)
    for row in cv_rows:
        row["model"] = meta_label
    aggregate_row = _build_cv_aggregate(cv_rows, meta_label)
    save_cv_summary(
        run_dir / f"{config.stack_name}_cv_summary.csv", cv_rows, aggregate_row
    )

    cv_metrics_json = _build_cv_metrics_json(meta_folds)
    save_json(run_dir / f"{config.stack_name}_cv_metrics.json", cv_metrics_json)

    # Global CSV outputs
    cv_perf_row = assemble_deep_learning_style_row(
        model_name=config.stack_name,
        experiment_name=args.experiment_name,
        dataset_type="cv",
        metrics=cv_metrics_json["mean"],
    )
    append_model_performance_csv(OUTPUT_ROOT / MODEL_PERFORMANCE_FILE, cv_perf_row)

    test_perf_row = assemble_deep_learning_style_row(
        model_name=config.stack_name,
        experiment_name=args.experiment_name,
        dataset_type="test",
        metrics=metrics,
    )
    append_model_performance_csv(OUTPUT_ROOT / MODEL_PERFORMANCE_FILE, test_perf_row)

    save_perfold_performance_csv(
        OUTPUT_ROOT / MODEL_PERFOLD_PERFORMANCE_FILE,
        config.stack_name,
        args.experiment_name,
        meta_folds,
    )
    save_cv_performance_csv(
        OUTPUT_ROOT / MODEL_CV_PERFORMANCE_FILE,
        config.stack_name,
        args.experiment_name,
        cv_metrics_json["mean"],
        cv_metrics_json["std"],
    )

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    logger.info("Generating diagnostic plots...")
    plot_all_diagnostics(y_test_original, y_pred_test, str(run_dir), config.stack_name)
    plot_train_diagnostics(y_original, y_pred_train_oof, str(run_dir), config.stack_name)
    logger.info("Diagnostic plots saved to %s", run_dir)

    logger.info("=" * 70)
    logger.info("DL stack '%s' complete.", config.stack_name)
    logger.info("Test RMSE: %.4f, R²: %.4f", metrics["rmse"], metrics["r2"])
    logger.info("Metrics: %s", run_dir / f"{config.stack_name}_metrics.json")
    logger.info("=" * 70)
