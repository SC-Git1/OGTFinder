"""Tuning helpers: Optuna-first, sklearn random-search fallback."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, r2_score

from utils_metrics import adjusted_r2_score, compute_binned_metrics, compute_weighted_rmse


def _get_n_workers() -> int:
    """Detect available CPUs from SLURM or OS, reserve 1 for the main thread."""
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1
    return max(1, n - 1)


def has_optuna() -> bool:
    """Check whether Optuna is available."""
    try:
        import optuna  # noqa: F401

        return True
    except Exception:
        return False


def _fit_model_with_optional_weights(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray | None,
):
    """Fit estimator while passing weights only where supported."""
    # Optional GP safety-mode subsampling controlled by model attributes.
    max_samples = getattr(model, "_gp_max_train_samples", None)
    if max_samples is not None and int(max_samples) > 0 and len(X_train) > int(max_samples):
        rng = np.random.RandomState(int(getattr(model, "_gp_random_state", 42)))
        selected = rng.choice(len(X_train), size=int(max_samples), replace=False)
        X_train = X_train[selected]
        y_train = y_train[selected]
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights)[selected]

    if sample_weights is None:
        model.fit(X_train, y_train)
        return model

    if isinstance(model, Pipeline):
        step_name = model.steps[-1][0]
        kwargs = {f"{step_name}__sample_weight": sample_weights}
        try:
            model.fit(X_train, y_train, **kwargs)
            return model
        except TypeError:
            model.fit(X_train, y_train)
            return model

    try:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    except TypeError:
        model.fit(X_train, y_train)
    return model


def _evaluate_params_groupkfold(
    build_model_fn: Callable[[dict[str, Any]], Any],
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray | None,
    n_cv: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Evaluate parameter set with GroupKFold weighted RMSE."""
    splitter = GroupKFold(n_splits=n_cv)
    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = build_model_fn(params)
        sw_train = sample_weights[train_idx] if sample_weights is not None else None
        sw_val = sample_weights[val_idx] if sample_weights is not None else None

        _fit_model_with_optional_weights(model, X[train_idx], y[train_idx], sw_train)
        preds = model.predict(X[val_idx])
        wrmse = compute_weighted_rmse(y[val_idx], preds, sw_val) if sw_val is not None else np.nan
        rmse = float(np.sqrt(np.mean((y[val_idx] - preds) ** 2)))
        n_features = X.shape[1]
        binned = compute_binned_metrics(y[val_idx], preds, n_features)
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "rmse": rmse,
                "mae": float(mean_absolute_error(y[val_idx], preds)),
                "r2": float(r2_score(y[val_idx], preds)),
                "adj_r2": float(adjusted_r2_score(y[val_idx], preds, n_features)),
                "weighted_rmse": float(wrmse) if not np.isnan(wrmse) else None,
                **binned,
            }
        )

    weighted_values = [r["weighted_rmse"] for r in fold_rows if r["weighted_rmse"] is not None]
    mean_weighted = float(np.mean(weighted_values)) if weighted_values else float("inf")
    return mean_weighted, fold_rows


def tune_with_optuna(
    *,
    model_name: str,
    build_model_fn: Callable[[dict[str, Any]], Any],
    suggest_params_fn: Callable[[Any], dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray | None,
    n_trials: int,
    n_cv: int,
    seed: int,
    logger,
    n_jobs: int | None = None,
) -> tuple[dict[str, Any], float, list[dict[str, Any]], str]:
    """Tune model with Optuna TPE and GroupKFold weighted RMSE.

    Parameters
    ----------
    n_jobs : int or None
        Number of parallel Optuna trials.  ``None`` (default) auto-detects
        from ``SLURM_CPUS_PER_TASK`` or ``os.cpu_count()``.
        Set to 1 for serial execution.
    """
    import optuna

    if n_jobs is None:
        n_jobs = _get_n_workers()

    def objective(trial):
        params = suggest_params_fn(trial)
        score, _ = _evaluate_params_groupkfold(
            build_model_fn=build_model_fn,
            params=params,
            X=X,
            y=y,
            groups=groups,
            sample_weights=sample_weights,
            n_cv=n_cv,
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    logger.info("[%s] Starting Optuna: %d trials, n_jobs=%d", model_name, n_trials, n_jobs)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_score, fold_rows = _evaluate_params_groupkfold(
        build_model_fn=build_model_fn,
        params=best_params,
        X=X,
        y=y,
        groups=groups,
        sample_weights=sample_weights,
        n_cv=n_cv,
    )
    logger.info("[%s] Optuna best weighted RMSE: %.6f", model_name, best_score)
    return best_params, best_score, fold_rows, "optuna"


def tune_with_fallback_random_search(
    *,
    model_name: str,
    build_model_fn: Callable[[dict[str, Any]], Any],
    param_distributions: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray | None,
    n_trials: int,
    n_cv: int,
    seed: int,
    logger,
) -> tuple[dict[str, Any], float, list[dict[str, Any]], str]:
    """Fallback random search tuner with GroupKFold weighted RMSE."""
    sampler = ParameterSampler(
        param_distributions=param_distributions,
        n_iter=max(1, n_trials),
        random_state=seed,
    )

    best_params = None
    best_score = float("inf")
    best_fold_rows: list[dict[str, Any]] = []

    for idx, params in enumerate(sampler, start=1):
        score, fold_rows = _evaluate_params_groupkfold(
            build_model_fn=build_model_fn,
            params=params,
            X=X,
            y=y,
            groups=groups,
            sample_weights=sample_weights,
            n_cv=n_cv,
        )
        logger.info("[%s][fallback] trial %d weighted RMSE=%.6f", model_name, idx, score)
        if score < best_score:
            best_score = score
            best_params = dict(params)
            best_fold_rows = fold_rows

    if best_params is None:
        raise RuntimeError(f"No candidate parameters generated for {model_name}")

    logger.info("[%s] Fallback best weighted RMSE: %.6f", model_name, best_score)
    return best_params, best_score, best_fold_rows, "fallback_random_search"


def tune_model(
    *,
    model_name: str,
    build_model_fn: Callable[[dict[str, Any]], Any],
    suggest_params_fn: Callable[[Any], dict[str, Any]],
    fallback_param_distributions: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray | None,
    n_trials: int,
    n_cv: int,
    seed: int,
    logger,
    require_optuna: bool = False,
    n_jobs: int | None = None,
) -> tuple[dict[str, Any], float, list[dict[str, Any]], str]:
    """Tune with Optuna when available, otherwise fallback random search."""
    if require_optuna and not has_optuna():
        raise ImportError(
            f"Optuna is required for model '{model_name}' but is not installed."
        )

    if require_optuna:
        return tune_with_optuna(
            model_name=model_name,
            build_model_fn=build_model_fn,
            suggest_params_fn=suggest_params_fn,
            X=X,
            y=y,
            groups=groups,
            sample_weights=sample_weights,
            n_trials=n_trials,
            n_cv=n_cv,
            seed=seed,
            logger=logger,
            n_jobs=n_jobs,
        )

    if has_optuna():
        return tune_with_optuna(
            model_name=model_name,
            build_model_fn=build_model_fn,
            suggest_params_fn=suggest_params_fn,
            X=X,
            y=y,
            groups=groups,
            sample_weights=sample_weights,
            n_trials=n_trials,
            n_cv=n_cv,
            seed=seed,
            logger=logger,
            n_jobs=n_jobs,
        )
    return tune_with_fallback_random_search(
        model_name=model_name,
        build_model_fn=build_model_fn,
        param_distributions=fallback_param_distributions,
        X=X,
        y=y,
        groups=groups,
        sample_weights=sample_weights,
        n_trials=n_trials,
        n_cv=n_cv,
        seed=seed,
        logger=logger,
    )



def generate_oof_predictions(
    build_model_fn: Callable[[dict[str, Any]], Any],
    best_params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray | None,
    n_cv: int,
    seed: int,
) -> np.ndarray:
    """Generate out-of-fold (OOF) predictions for stacking.

    For each fold the base model is built from scratch, fit on the training
    fold, and used to predict the held-out validation fold.  The predictions
    are assembled so every sample is predicted exactly once by a model that
    never saw it during training — eliminating meta-learner data leakage.

    Parameters
    ----------
    build_model_fn : callable
        Factory ``(params) -> estimator``.
    best_params : dict
        Best hyperparameters to pass to ``build_model_fn``.
    X, y, groups : array-like
        Full training features, targets, and group labels.
    sample_weights : array or None
        Per-sample weights (indexed in parallel with *X* / *y*).
    n_cv : int
        Number of GroupKFold splits.
    seed : int
        Random seed (set before each fold for reproducibility).

    Returns
    -------
    oof_preds : np.ndarray, shape ``(len(y),)``
        OOF predictions aligned with the original row order.
    """
    splitter = GroupKFold(n_splits=n_cv)
    oof_preds = np.empty(len(y), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
        np.random.seed(seed)

        model = build_model_fn(best_params)
        sw_train = sample_weights[train_idx] if sample_weights is not None else None
        _fit_model_with_optional_weights(model, X[train_idx], y[train_idx], sw_train)
        oof_preds[val_idx] = model.predict(X[val_idx])

    return oof_preds


def build_standardized_model(estimator):
    """Wrap estimator with StandardScaler in pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("regressor", estimator)])

