"""Meta-learner registry for deep-learning-based stacking ensembles.

Provides builder functions and Optuna search-space suggesters for 10
meta-learner types used across the 10 DL stacking plans.
"""

from __future__ import annotations

from typing import Any

from config import SEED

# ---------------------------------------------------------------------------
# 1. Ridge Regression  (Plan 1: Elite Specialists)
# ---------------------------------------------------------------------------

def build_meta_ridge(params: dict[str, Any], seed: int = SEED):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=float(params.get("alpha", 1.0)))


def suggest_meta_ridge(trial) -> dict[str, Any]:
    return {"alpha": trial.suggest_float("alpha", 1e-6, 100.0, log=True)}


def fallback_meta_ridge() -> dict[str, list]:
    return {"alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]}


# ---------------------------------------------------------------------------
# 2. LightGBM  (Plan 2: Architecture Diversity)
# ---------------------------------------------------------------------------

def build_meta_lgbm(params: dict[str, Any], seed: int = SEED):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=int(params.get("n_estimators", 200)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 4)),
        num_leaves=int(params.get("num_leaves", 15)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_alpha=float(params.get("reg_alpha", 0.01)),
        reg_lambda=float(params.get("reg_lambda", 0.01)),
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )


def suggest_meta_lgbm(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "num_leaves": trial.suggest_int("num_leaves", 4, 31),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
    }


def fallback_meta_lgbm() -> dict[str, list]:
    return {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "num_leaves": [7, 15, 31],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.001, 0.01, 0.1],
        "reg_lambda": [0.001, 0.01, 0.1],
    }


# ---------------------------------------------------------------------------
# 3. Huber Regressor  (Plan 3: Extreme Hunter)
# ---------------------------------------------------------------------------

def build_meta_huber(params: dict[str, Any], seed: int = SEED):
    from sklearn.linear_model import HuberRegressor
    return HuberRegressor(
        epsilon=float(params.get("epsilon", 1.35)),
        alpha=float(params.get("alpha", 1e-4)),
        max_iter=5000,
    )


def suggest_meta_huber(trial) -> dict[str, Any]:
    return {
        "epsilon": trial.suggest_float("epsilon", 1.01, 2.0),
        "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
    }


def fallback_meta_huber() -> dict[str, list]:
    return {
        "epsilon": [1.1, 1.2, 1.35, 1.5, 1.8],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
    }


# ---------------------------------------------------------------------------
# 4. Lasso Regression  (Plan 4: Deep MLP Feature Extractor)
# ---------------------------------------------------------------------------

def build_meta_lasso(params: dict[str, Any], seed: int = SEED):
    from sklearn.linear_model import Lasso
    return Lasso(
        alpha=float(params.get("alpha", 0.01)),
        max_iter=10000,
        random_state=seed,
    )


def suggest_meta_lasso(trial) -> dict[str, Any]:
    return {"alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True)}


def fallback_meta_lasso() -> dict[str, list]:
    return {"alpha": [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]}


# ---------------------------------------------------------------------------
# 5. XGBoost  (Plan 5: Kitchen Sink)
# ---------------------------------------------------------------------------

def build_meta_xgb(params: dict[str, Any], seed: int = SEED):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=int(params.get("n_estimators", 200)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 3)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_alpha=float(params.get("reg_alpha", 0.01)),
        reg_lambda=float(params.get("reg_lambda", 0.01)),
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def suggest_meta_xgb(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
    }


def fallback_meta_xgb() -> dict[str, list]:
    return {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.001, 0.01, 0.1],
        "reg_lambda": [0.001, 0.01, 0.1],
    }


# ---------------------------------------------------------------------------
# 6. Meta-MLP  (Plan 6: Transformer & Attention Resurgence)
# ---------------------------------------------------------------------------

def build_meta_mlp(params: dict[str, Any], seed: int = SEED):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(
        hidden_layer_sizes=params.get("hidden_layer_sizes", (64, 32)),
        activation=params.get("activation", "relu"),
        alpha=float(params.get("alpha", 1e-4)),
        learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
        max_iter=5000,
        early_stopping=True,
        random_state=seed,
    )


def suggest_meta_mlp(trial) -> dict[str, Any]:
    return {
        "hidden_layer_sizes": trial.suggest_categorical(
            "hidden_layer_sizes",
            [(32,), (64, 32), (128, 64), (64, 32, 16)],
        ),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
    }


def fallback_meta_mlp() -> dict[str, list]:
    return {
        "hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-4, 5e-4, 1e-3],
    }


# ---------------------------------------------------------------------------
# 7. CatBoost  (Plan 7: Balanced Power)
# ---------------------------------------------------------------------------

def build_meta_catboost(params: dict[str, Any], seed: int = SEED):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=int(params.get("iterations", 300)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        depth=int(params.get("depth", 4)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        thread_count=1,
    )


def suggest_meta_catboost(trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "depth": trial.suggest_int("depth", 2, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
    }


def fallback_meta_catboost() -> dict[str, list]:
    return {
        "iterations": [200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "depth": [3, 4, 5],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
    }


# ---------------------------------------------------------------------------
# 8. Random Forest  (Plan 8: Contrarian)
# ---------------------------------------------------------------------------

def build_meta_rf(params: dict[str, Any], seed: int = SEED):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
        max_features=params.get("max_features", "sqrt"),
        random_state=seed,
        n_jobs=1,
    )


def suggest_meta_rf(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
    }


def fallback_meta_rf() -> dict[str, list]:
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 8, 10, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 1.0],
    }


# ---------------------------------------------------------------------------
# 9. ElasticNet  (Plan 9: Hot/Cold Bridge)
# ---------------------------------------------------------------------------

def build_meta_elasticnet(params: dict[str, Any], seed: int = SEED):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(
        alpha=float(params.get("alpha", 0.01)),
        l1_ratio=float(params.get("l1_ratio", 0.5)),
        max_iter=10000,
        random_state=seed,
    )


def suggest_meta_elasticnet(trial) -> dict[str, Any]:
    return {
        "alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
    }


def fallback_meta_elasticnet() -> dict[str, list]:
    return {
        "alpha": [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }


# ---------------------------------------------------------------------------
# 10. SVR with RBF Kernel  (Plan 10: Two-Stage Cascade)
# ---------------------------------------------------------------------------

def build_meta_svr(params: dict[str, Any], seed: int = SEED):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(
            C=float(params.get("C", 1.0)),
            epsilon=float(params.get("epsilon", 0.1)),
            gamma=params.get("gamma", "scale"),
            kernel="rbf",
        )),
    ])


def suggest_meta_svr(trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def fallback_meta_svr() -> dict[str, list]:
    return {
        "C": [1e-2, 1e-1, 1.0, 10.0, 100.0],
        "epsilon": [1e-3, 1e-2, 0.1, 0.5],
        "gamma": ["scale", "auto"],
    }


# ---------------------------------------------------------------------------
# Registry mapping plan names -> (build, suggest, fallback) tuples
# ---------------------------------------------------------------------------

META_REGISTRY: dict[str, dict] = {
    "ridge": {
        "build": build_meta_ridge,
        "suggest": suggest_meta_ridge,
        "fallback": fallback_meta_ridge,
    },
    "lgbm": {
        "build": build_meta_lgbm,
        "suggest": suggest_meta_lgbm,
        "fallback": fallback_meta_lgbm,
    },
    "huber": {
        "build": build_meta_huber,
        "suggest": suggest_meta_huber,
        "fallback": fallback_meta_huber,
    },
    "lasso": {
        "build": build_meta_lasso,
        "suggest": suggest_meta_lasso,
        "fallback": fallback_meta_lasso,
    },
    "xgb": {
        "build": build_meta_xgb,
        "suggest": suggest_meta_xgb,
        "fallback": fallback_meta_xgb,
    },
    "mlp": {
        "build": build_meta_mlp,
        "suggest": suggest_meta_mlp,
        "fallback": fallback_meta_mlp,
    },
    "catboost": {
        "build": build_meta_catboost,
        "suggest": suggest_meta_catboost,
        "fallback": fallback_meta_catboost,
    },
    "rf": {
        "build": build_meta_rf,
        "suggest": suggest_meta_rf,
        "fallback": fallback_meta_rf,
    },
    "elasticnet": {
        "build": build_meta_elasticnet,
        "suggest": suggest_meta_elasticnet,
        "fallback": fallback_meta_elasticnet,
    },
    "svr": {
        "build": build_meta_svr,
        "suggest": suggest_meta_svr,
        "fallback": fallback_meta_svr,
    },
}
