"""
DL Training Utilities for ensemble_models_fixed.

Extracted from deep_learning/utils.py — contains only the functions needed
for training DL base models from scratch and generating OOF predictions.
"""

import os
import json
import logging
import hashlib
import random
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from config import (
    RANDOM_STATE,
    CV_SPLIT_PROTOCOL,
    CV_SPLIT_MANIFEST_PATH,
    GROUP_NAN_POLICY,
    CV_STRICT_DATASET_MATCH,
    CV_MANIFEST_TOLERANCE_MAX_MISSING_ROWS,
    CV_MANIFEST_TOLERANCE_ALLOW_EXTRA_ROWS,
    CV_MANIFEST_TOLERANCE_ON_MISMATCH,
)

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


###############################################################################
# LOGGING SETUP
###############################################################################

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


###############################################################################
# REPRODUCIBILITY / SPLIT CONTEXT
###############################################################################

_CURRENT_SPLIT_CONTEXT = {
    'row_ids': None,
    'dataset_fingerprint': None,
    'dataset_matches_manifest': None,
    'manifest_path': None,
    'split_protocol': None,
    'configured_split_protocol': None,
    'effective_split_protocol': None,
    'manifest_missing_row_ids': None,
    'manifest_extra_row_ids': None,
}


def set_global_seeds(seed: int = RANDOM_STATE):
    """Set Python/NumPy/TensorFlow seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _compute_file_sha256(path: str) -> str | None:
    """Compute sha256 for a file path when available."""
    if not path or not os.path.exists(path):
        return None
    hasher = hashlib.sha256()
    with open(path, 'rb') as infile:
        while True:
            chunk = infile.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_scalar(value) -> str:
    """Canonical scalar serialization used for stable row identities."""
    if value is None:
        return "__NA__"

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "__NA__"
        return format(float(value), '.17g')

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, str):
        if value == "":
            return ""
        lower = value.lower()
        if lower in {"nan", "none", "null"}:
            return "__NA__"
        if value.lstrip("-").isdigit():
            return str(int(value))
        try:
            as_float = float(value)
            if np.isfinite(as_float):
                return format(float(as_float), '.17g')
        except ValueError:
            pass
        return value

    try:
        if pd.isna(value):
            return "__NA__"
    except TypeError:
        pass

    return str(value)


def build_row_identity(df_or_arrays) -> list[str]:
    """Build stable, occurrence-aware row IDs."""
    row_hashes = []

    if isinstance(df_or_arrays, pd.DataFrame):
        columns = list(df_or_arrays.columns)
        for _, row in df_or_arrays.iterrows():
            payload = "|".join(
                f"{col}={_canonical_scalar(row[col])}" for col in columns
            )
            row_hashes.append(hashlib.sha256(payload.encode('utf-8')).hexdigest())
    else:
        if isinstance(df_or_arrays, dict):
            X = np.asarray(df_or_arrays['X'])
            y = np.asarray(df_or_arrays['y'])
            groups = np.asarray(df_or_arrays['groups'])
        elif isinstance(df_or_arrays, (tuple, list)) and len(df_or_arrays) >= 3:
            X = np.asarray(df_or_arrays[0])
            y = np.asarray(df_or_arrays[1])
            groups = np.asarray(df_or_arrays[2])
        else:
            raise TypeError("build_row_identity expects DataFrame, dict, or (X, y, groups)")

        if X.shape[0] != y.shape[0] or X.shape[0] != groups.shape[0]:
            raise ValueError("X, y, and groups must have matching row counts")

        for idx in range(X.shape[0]):
            row_values = [_canonical_scalar(v) for v in X[idx].tolist()]
            payload_parts = [
                f"x={','.join(row_values)}",
                f"y={_canonical_scalar(y[idx])}",
                f"g={_canonical_scalar(groups[idx])}",
            ]
            payload = "|".join(payload_parts)
            row_hashes.append(hashlib.sha256(payload.encode('utf-8')).hexdigest())

    occurrence = {}
    row_ids = []
    for row_hash in row_hashes:
        occ = occurrence.get(row_hash, 0)
        row_ids.append(f"{row_hash}:{occ}")
        occurrence[row_hash] = occ + 1

    return row_ids


def _build_dataset_fingerprint(
    data: pd.DataFrame,
    data_path: str,
    target_col: str,
    group_col: str,
) -> dict:
    """Build dataset fingerprint for strict ACS parity checks."""
    return {
        'sha256': _compute_file_sha256(data_path),
        'row_count': int(len(data)),
        'columns': list(data.columns),
        'target_col': target_col,
        'group_col': group_col,
    }


def load_split_manifest(manifest_path: str) -> dict:
    """Load and minimally validate a split manifest JSON."""
    if not manifest_path:
        raise ValueError("Manifest path is required for exact split protocol")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as infile:
        manifest = json.load(infile)

    required = {
        'schema_version',
        'split_protocol',
        'dataset_fingerprint',
        'row_ids',
        'optuna_objective_folds',
        'cv_report_folds',
    }
    missing = sorted(required - set(manifest.keys()))
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")
    if manifest.get('split_protocol') != 'acs_v1':
        raise ValueError(
            f"Unsupported manifest split_protocol '{manifest.get('split_protocol')}'. Expected 'acs_v1'."
        )
    return manifest


def _compute_manifest_row_id_deltas(
    *,
    manifest_row_ids: list[str],
    current_row_ids: list[str],
) -> tuple[list[str], list[str]]:
    """Compute row-ID set differences between manifest and current dataset."""
    manifest_set = set(manifest_row_ids)
    current_set = set(current_row_ids)

    missing_manifest_rows = sorted(manifest_set - current_set)
    extra_dataset_rows = sorted(current_set - manifest_set)
    return missing_manifest_rows, extra_dataset_rows


def _emit_split_notice(logger: logging.Logger, message: str):
    """Emit split diagnostics to both logs and terminal stdout."""
    logger.warning(message)
    print(message, flush=True)


def _enforce_tolerant_manifest_policy(
    *,
    missing_manifest_rows: list[str],
    extra_dataset_rows: list[str],
    max_missing_rows: int,
    allow_extra_rows: bool,
    on_mismatch: str,
    context_label: str,
):
    """Validate tolerated manifest drift according to configuration."""
    if max_missing_rows < 0:
        raise ValueError(
            f"Manifest tolerance max_missing_rows must be >= 0, got {max_missing_rows}"
        )
    if on_mismatch not in {'warn', 'error'}:
        raise ValueError(
            f"Unsupported CV_MANIFEST_TOLERANCE_ON_MISMATCH value '{on_mismatch}'. "
            "Expected one of ['warn', 'error']."
        )

    missing_count = len(missing_manifest_rows)
    extra_count = len(extra_dataset_rows)

    if missing_count > max_missing_rows:
        raise ValueError(
            f"{context_label}: manifest tolerance exceeded missing-row threshold "
            f"(missing={missing_count}, allowed={max_missing_rows})."
        )
    if extra_count > 0 and not allow_extra_rows:
        raise ValueError(
            f"{context_label}: dataset contains {extra_count} row IDs not present in manifest, "
            "and extra rows are disallowed."
        )

    if (missing_count > 0 or extra_count > 0) and on_mismatch == 'error':
        raise ValueError(
            f"{context_label}: tolerated mismatch detected (missing={missing_count}, extra={extra_count}) "
            "and CV_MANIFEST_TOLERANCE_ON_MISMATCH='error'."
        )


def _filter_fold_ids_to_current_dataset(
    fold_row_ids: list[str],
    current_row_id_set: set[str],
) -> list[str]:
    """Filter fold row IDs to rows available in the current dataset."""
    return [row_id for row_id in fold_row_ids if row_id in current_row_id_set]


def _set_split_context(
    *,
    row_ids: list[str],
    dataset_fingerprint: dict,
    dataset_matches_manifest: bool | None,
    manifest_path: str | None,
    split_protocol: str,
    configured_split_protocol: str | None = None,
    effective_split_protocol: str | None = None,
    manifest_missing_row_ids: list[str] | None = None,
    manifest_extra_row_ids: list[str] | None = None,
):
    _CURRENT_SPLIT_CONTEXT['row_ids'] = row_ids
    _CURRENT_SPLIT_CONTEXT['dataset_fingerprint'] = dataset_fingerprint
    _CURRENT_SPLIT_CONTEXT['dataset_matches_manifest'] = dataset_matches_manifest
    _CURRENT_SPLIT_CONTEXT['manifest_path'] = manifest_path
    _CURRENT_SPLIT_CONTEXT['split_protocol'] = split_protocol
    _CURRENT_SPLIT_CONTEXT['configured_split_protocol'] = configured_split_protocol
    _CURRENT_SPLIT_CONTEXT['effective_split_protocol'] = effective_split_protocol
    _CURRENT_SPLIT_CONTEXT['manifest_missing_row_ids'] = manifest_missing_row_ids
    _CURRENT_SPLIT_CONTEXT['manifest_extra_row_ids'] = manifest_extra_row_ids


def get_split_context() -> dict:
    """Expose current split context for tooling and diagnostics."""
    return dict(_CURRENT_SPLIT_CONTEXT)


def resolve_exact_folds_from_manifest(
    stage: str,
    row_ids: list[str],
    manifest: dict,
    strict: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Resolve fold indices from manifest row IDs for a given stage.

    Stages:
    - "optuna_objective" -> manifest["optuna_objective_folds"]
    - "cv_report" -> manifest["cv_report_folds"]
    """
    stage_to_key = {
        'optuna_objective': 'optuna_objective_folds',
        'cv_report': 'cv_report_folds',
    }
    if stage not in stage_to_key:
        raise ValueError(f"Unsupported stage '{stage}'. Expected one of {list(stage_to_key.keys())}")

    fold_key = stage_to_key[stage]
    manifest_folds = manifest.get(fold_key, [])
    if not manifest_folds:
        raise ValueError(f"Manifest contains no folds for stage '{stage}'")

    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Row IDs are not unique; occurrence-aware IDs are required for exact mapping")
    row_id_to_idx = {row_id: idx for idx, row_id in enumerate(row_ids)}

    resolved = []
    val_ids_all = []
    for fold in manifest_folds:
        train_ids = fold.get('train_row_ids', [])
        val_ids = fold.get('val_row_ids', [])

        try:
            train_idx = np.array([row_id_to_idx[row_id] for row_id in train_ids], dtype=np.int64)
            val_idx = np.array([row_id_to_idx[row_id] for row_id in val_ids], dtype=np.int64)
        except KeyError as exc:
            missing_id = exc.args[0]
            raise ValueError(
                f"Manifest row ID '{missing_id}' is not present in current dataset context"
            ) from exc

        if strict and np.intersect1d(train_idx, val_idx).size > 0:
            raise ValueError(f"Fold '{fold.get('fold')}' has train/val overlap in manifest")

        resolved.append((train_idx, val_idx))
        val_ids_all.extend(val_ids)

    if strict:
        if sorted(val_ids_all) != sorted(row_ids):
            raise ValueError(
                f"Stage '{stage}' manifest validation rows do not match dataset row IDs exactly"
            )

    return resolved


def resolve_tolerant_folds_from_manifest(
    stage: str,
    row_ids: list[str],
    manifest: dict,
    strict: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Resolve fold indices from manifest row IDs using manifest intersection.

    Rows missing from the current dataset are ignored. Extra rows should be
    filtered by policy before calling this resolver.
    """
    stage_to_key = {
        'optuna_objective': 'optuna_objective_folds',
        'cv_report': 'cv_report_folds',
    }
    if stage not in stage_to_key:
        raise ValueError(f"Unsupported stage '{stage}'. Expected one of {list(stage_to_key.keys())}")

    fold_key = stage_to_key[stage]
    manifest_folds = manifest.get(fold_key, [])
    if not manifest_folds:
        raise ValueError(f"Manifest contains no folds for stage '{stage}'")

    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Row IDs are not unique; occurrence-aware IDs are required for tolerant mapping")

    row_id_set = set(row_ids)
    row_id_to_idx = {row_id: idx for idx, row_id in enumerate(row_ids)}

    resolved = []
    val_ids_all = []
    for fold in manifest_folds:
        train_ids = _filter_fold_ids_to_current_dataset(fold.get('train_row_ids', []), row_id_set)
        val_ids = _filter_fold_ids_to_current_dataset(fold.get('val_row_ids', []), row_id_set)

        train_idx = np.array([row_id_to_idx[row_id] for row_id in train_ids], dtype=np.int64)
        val_idx = np.array([row_id_to_idx[row_id] for row_id in val_ids], dtype=np.int64)

        if strict and np.intersect1d(train_idx, val_idx).size > 0:
            raise ValueError(f"Fold '{fold.get('fold')}' has train/val overlap after tolerant intersection")

        resolved.append((train_idx, val_idx))
        val_ids_all.extend(val_ids)

    if strict:
        if sorted(val_ids_all) != sorted(row_ids):
            raise ValueError(
                f"Stage '{stage}' tolerant manifest validation rows do not cover current dataset row IDs exactly"
            )

    return resolved


def get_stage_folds(
    stage: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int = RANDOM_STATE,
    n_splits: int = 5,
    manifest_path: str = CV_SPLIT_MANIFEST_PATH,
    protocol: str = CV_SPLIT_PROTOCOL,
    strict: bool = CV_STRICT_DATASET_MATCH,
    manifest_tolerance_max_missing_rows: int = CV_MANIFEST_TOLERANCE_MAX_MISSING_ROWS,
    manifest_tolerance_allow_extra_rows: bool = CV_MANIFEST_TOLERANCE_ALLOW_EXTRA_ROWS,
    manifest_tolerance_on_mismatch: str = CV_MANIFEST_TOLERANCE_ON_MISMATCH,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Resolve fold indices for a training stage under the configured split protocol."""
    logger = logging.getLogger(__name__)

    if protocol == 'native_groupkfold':
        gkf = GroupKFold(n_splits=n_splits)
        folds = [(train_idx, val_idx) for train_idx, val_idx in gkf.split(X, y, groups)]
        logger.info(
            "Split stage '%s': protocol=%s, n_splits=%d (native GroupKFold)",
            stage, protocol, len(folds)
        )
        return folds

    if protocol not in {'acs_v1_exact', 'acs_v1_tolerant'}:
        raise ValueError(f"Unsupported CV split protocol '{protocol}'")

    manifest = load_split_manifest(manifest_path)
    row_ids = _CURRENT_SPLIT_CONTEXT.get('row_ids')
    if row_ids is None:
        row_ids = build_row_identity({'X': X, 'y': y, 'groups': groups})
    if len(row_ids) != X.shape[0]:
        raise ValueError(
            f"Row identity length mismatch: row_ids={len(row_ids)} vs X rows={X.shape[0]}"
        )
    manifest_row_ids = manifest.get('row_ids', [])
    missing_manifest_rows, extra_dataset_rows = _compute_manifest_row_id_deltas(
        manifest_row_ids=manifest_row_ids,
        current_row_ids=row_ids,
    )

    effective_protocol = _CURRENT_SPLIT_CONTEXT.get('effective_split_protocol')
    if effective_protocol not in {'acs_v1_exact', 'acs_v1_tolerant', 'native_groupkfold'}:
        if not missing_manifest_rows and not extra_dataset_rows:
            effective_protocol = 'acs_v1_exact'
        elif extra_dataset_rows:
            effective_protocol = 'native_groupkfold'
        else:
            effective_protocol = 'acs_v1_tolerant'

    if missing_manifest_rows or extra_dataset_rows:
        _emit_split_notice(
            logger,
            (
                f"Split stage '{stage}': manifest/data mismatch detected "
                f"(missing={len(missing_manifest_rows)}, extra={len(extra_dataset_rows)}). "
                f"Configured protocol={protocol}, effective protocol={effective_protocol}. "
                f"Continuing without failing exact-match checks."
            ),
        )

    if effective_protocol == 'native_groupkfold':
        gkf = GroupKFold(n_splits=n_splits)
        folds = [(train_idx, val_idx) for train_idx, val_idx in gkf.split(X, y, groups)]
        logger.info(
            "Split stage '%s': protocol=%s, effective=%s, n_splits=%d (native fallback)",
            stage, protocol, effective_protocol, len(folds)
        )
        return folds

    if effective_protocol == 'acs_v1_exact':
        try:
            folds = resolve_exact_folds_from_manifest(stage, row_ids, manifest, strict=strict)
        except ValueError as exc:
            if (
                "not present in current dataset context" in str(exc)
                or "validation rows do not match dataset row IDs exactly" in str(exc)
            ):
                _emit_split_notice(
                    logger,
                    (
                        f"Split stage '{stage}': exact manifest mapping failed ({exc}). "
                        "Falling back to tolerant manifest intersection."
                    ),
                )
                folds = resolve_tolerant_folds_from_manifest(stage, row_ids, manifest, strict=strict)
                effective_protocol = 'acs_v1_tolerant'
            else:
                raise
    else:
        folds = resolve_tolerant_folds_from_manifest(stage, row_ids, manifest, strict=strict)

    if n_splits != len(folds):
        raise ValueError(
            f"Requested n_splits={n_splits}, but manifest provides {len(folds)} folds for stage '{stage}'"
        )

    logger.info(
        "Split stage '%s': protocol=%s, effective=%s, manifest=%s, n_splits=%d",
        stage, protocol, effective_protocol, manifest_path, len(folds)
    )
    return folds


###############################################################################
# PROGRESS TRACKING
###############################################################################

class TqdmKerasCallback(keras.callbacks.Callback):
    """Custom Keras callback for tqdm progress bar during training."""

    def __init__(self, total_epochs: int = None, desc: str = "Epoch"):
        super().__init__()
        self.total_epochs = total_epochs
        self.desc = desc
        self.pbar = None

    def on_train_begin(self, logs=None):
        if not TQDM_AVAILABLE:
            return
        total = self.total_epochs or self.params.get('epochs', 100)
        self.pbar = tqdm(total=total, desc=self.desc, unit="epoch",
                         leave=True, dynamic_ncols=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.pbar is None:
            return
        logs = logs or {}
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        postfix = f"loss={loss:.4f}"
        if val_loss:
            postfix += f" val={val_loss:.4f}"
        self.pbar.set_postfix_str(postfix)
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def get_tqdm_keras_callback(total_epochs: int = None, desc: str = "Epoch"):
    """Get a tqdm callback for Keras training if tqdm is available."""
    if not TQDM_AVAILABLE:
        return None
    return TqdmKerasCallback(total_epochs=total_epochs, desc=desc)


def print_phase(model_name: str, phase, total_phases: int, description: str):
    """Print a phase indicator for progress tracking."""
    print(f"\n[{model_name}] Phase {phase}/{total_phases}: {description}")


###############################################################################
# GPU & KERAS METRICS
###############################################################################

def check_gpu() -> bool:
    """Check and log GPU availability."""
    logger = logging.getLogger(__name__)
    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"GPU available: {gpus[0]}")
        return True
    logger.info("No GPU available. Using CPU.")
    return False


def r2_keras(y_true, y_pred):
    """Custom R² metric for Keras."""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


r2_keras.__name__ = 'r2'


###############################################################################
# DATA LOADING
###############################################################################

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from various file formats (.feather, .csv, .parquet)."""
    if data_path.endswith('.feather'):
        return feather.read_feather(data_path)
    elif data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


def load_and_preprocess_test(
    data_path: str,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    feature_columns: list = None,
    target_col: str = 'topt',
    feature_regex: str = None
) -> tuple:
    """Load and preprocess test data using pre-fitted scalers.

    Returns:
        Tuple of (X_test, y_test, y_test_original)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading test data from {data_path}...")

    data = load_data(data_path)
    logger.info(f"Test data shape: {data.shape}")

    if feature_columns is not None:
        missing_cols = [c for c in feature_columns if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in test data: {missing_cols}")
        features = data[feature_columns]
    elif feature_regex is not None:
        features = data.filter(regex=feature_regex)
    else:
        raise ValueError("Either feature_columns or feature_regex must be provided")

    target = data[target_col]

    X_test = feature_scaler.transform(features)
    y_test_original = target.values
    y_test = target_scaler.transform(target.values.reshape(-1, 1)).flatten()

    logger.info(f"Test samples: {X_test.shape[0]}")
    return X_test, y_test, y_test_original


###############################################################################
# CALLBACKS
###############################################################################

def create_callbacks(
    patience_es: int = 30,
    patience_lr: int = 15,
    min_lr: float = 1e-6,
    monitor: str = 'val_loss'
) -> list:
    """Create standard training callbacks (EarlyStopping + ReduceLROnPlateau)."""
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1
        )
    ]


###############################################################################
# SAMPLE WEIGHTS
###############################################################################

def load_sample_weights(weights_path: str) -> dict:
    """Load sample weights from JSON file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading sample weights from {weights_path}...")
    with open(weights_path, 'r') as f:
        weights_dict = json.load(f)
    logger.info(f"Loaded weights for {len(weights_dict)} target value bins")
    return weights_dict


def get_sample_weights_array(y: np.ndarray, weights_dict: dict) -> np.ndarray:
    """Convert target values to sample weights array for Keras training."""
    bin_values = np.array(sorted([float(k) for k in weights_dict.keys()]))
    bin_weights = np.array([weights_dict[str(k) if str(k) in weights_dict else str(int(k))]
                           for k in bin_values])
    sample_weights = np.zeros(len(y))
    for i, val in enumerate(y):
        idx = np.abs(bin_values - val).argmin()
        sample_weights[i] = bin_weights[idx]
    return sample_weights


def extreme_rmse_sampled(y_true: np.ndarray, y_pred: np.ndarray,
                         sample_weights: np.ndarray) -> float:
    """Compute weighted RMSE using pre-computed sample weights."""
    squared_errors = (y_true - y_pred) ** 2
    weighted_mse = np.average(squared_errors, weights=sample_weights)
    return np.sqrt(weighted_mse)


###############################################################################
# ADJUSTED METRICS
###############################################################################

def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Compute adjusted R² accounting for feature count."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n <= n_features + 1:
        return r2
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2


def calculate_binned_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             n_features: int) -> dict:
    """Calculate metrics for top/bottom 5%, 10%, 20% percentile bins."""
    results = {}
    for pct in [5, 10, 20]:
        bottom_threshold = np.percentile(y_true, pct)
        bottom_mask = y_true <= bottom_threshold
        if bottom_mask.sum() > 1:
            results[f'rmse_bottom_{pct}'] = np.sqrt(
                mean_squared_error(y_true[bottom_mask], y_pred[bottom_mask]))
            results[f'mae_bottom_{pct}'] = mean_absolute_error(
                y_true[bottom_mask], y_pred[bottom_mask])
            results[f'r2_bottom_{pct}'] = r2_score(
                y_true[bottom_mask], y_pred[bottom_mask])
            results[f'adj_r2_bottom_{pct}'] = adjusted_r2_score(
                y_true[bottom_mask], y_pred[bottom_mask], n_features)
            results[f'n_bottom_{pct}'] = int(bottom_mask.sum())
        else:
            results[f'rmse_bottom_{pct}'] = None
            results[f'mae_bottom_{pct}'] = None
            results[f'r2_bottom_{pct}'] = None
            results[f'adj_r2_bottom_{pct}'] = None
            results[f'n_bottom_{pct}'] = 0

        top_threshold = np.percentile(y_true, 100 - pct)
        top_mask = y_true >= top_threshold
        if top_mask.sum() > 1:
            results[f'rmse_top_{pct}'] = np.sqrt(
                mean_squared_error(y_true[top_mask], y_pred[top_mask]))
            results[f'mae_top_{pct}'] = mean_absolute_error(
                y_true[top_mask], y_pred[top_mask])
            results[f'r2_top_{pct}'] = r2_score(
                y_true[top_mask], y_pred[top_mask])
            results[f'adj_r2_top_{pct}'] = adjusted_r2_score(
                y_true[top_mask], y_pred[top_mask], n_features)
            results[f'n_top_{pct}'] = int(top_mask.sum())
        else:
            results[f'rmse_top_{pct}'] = None
            results[f'mae_top_{pct}'] = None
            results[f'r2_top_{pct}'] = None
            results[f'adj_r2_top_{pct}'] = None
            results[f'n_top_{pct}'] = 0
    return results


###############################################################################
# GROUPKFOLD CV FOR KERAS (OOF generation)
###############################################################################

def groupkfold_cross_validate(
    model_builder_fn,
    best_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray = None,
    n_splits: int = 5,
    batch_size: int = 32,
    max_epochs: int = 500,
    patience_es: int = 30,
    patience_lr: int = 15,
    n_features: int = None,
    random_state: int = RANDOM_STATE,
    split_protocol: str = CV_SPLIT_PROTOCOL,
    manifest_path: str = CV_SPLIT_MANIFEST_PATH,
    strict: bool = CV_STRICT_DATASET_MATCH,
) -> tuple:
    """
    Perform GroupKFold CV with optimal params for final analysis.

    Scaling is performed per-fold to avoid information leakage.

    Args:
        model_builder_fn: Function(params, input_dim) -> compiled Keras model
        best_params: Best hyperparameters
        X: Feature array (raw, unscaled)
        y: Target array (raw, unscaled)
        groups: Group labels for GroupKFold
        sample_weights: Pre-computed sample weights
        n_splits: Number of CV folds

    Returns:
        Tuple of (mean_metrics, std_metrics, all_fold_metrics, oof_predictions)
    """
    logger = logging.getLogger(__name__)
    n_features = n_features or X.shape[1]
    fold_splits = get_stage_folds(
        stage='cv_report',
        X=X,
        y=y,
        groups=groups,
        random_state=random_state,
        n_splits=n_splits,
        manifest_path=manifest_path,
        protocol=split_protocol,
        strict=strict,
    )

    all_fold_metrics = []
    oof_indices = []
    oof_actual = []
    oof_predicted = []
    oof_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        logger.info(f"Training fold {fold_idx + 1}/{n_splits}...")

        keras.backend.clear_session()
        set_global_seeds(random_state)

        # Per-fold scaling to avoid information leakage
        fold_feature_scaler = StandardScaler()
        X_train_fold = fold_feature_scaler.fit_transform(X[train_idx])
        X_val_fold = fold_feature_scaler.transform(X[val_idx])

        fold_target_scaler = StandardScaler()
        y_train_fold = fold_target_scaler.fit_transform(y[train_idx].reshape(-1, 1)).flatten()
        y_val_fold = fold_target_scaler.transform(y[val_idx].reshape(-1, 1)).flatten()
        y_val_original = y[val_idx]

        sw_train = sample_weights[train_idx] if sample_weights is not None else None
        sw_val = sample_weights[val_idx] if sample_weights is not None else None

        # Build and train model
        input_dim = X.shape[1]
        model = model_builder_fn(best_params, input_dim)

        callbacks = create_callbacks(patience_es, patience_lr)

        model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            sample_weight=sw_train,
            verbose=0
        )

        # Evaluate using fold-specific scaler for inverse transform
        y_pred_scaled = model.predict(X_val_fold, verbose=0).flatten()
        y_pred = fold_target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        fold_metrics = {
            'fold': fold_idx,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'rmse': np.sqrt(mean_squared_error(y_val_original, y_pred)),
            'mae': mean_absolute_error(y_val_original, y_pred),
            'r2': r2_score(y_val_original, y_pred),
            'adj_r2': adjusted_r2_score(y_val_original, y_pred, n_features)
        }

        if sw_val is not None:
            fold_metrics['weighted_rmse'] = extreme_rmse_sampled(y_val_original, y_pred, sw_val)

        binned = calculate_binned_metrics(y_val_original, y_pred, n_features)
        fold_metrics.update(binned)

        all_fold_metrics.append(fold_metrics)

        # Collect OOF predictions (original scale)
        oof_indices.extend(val_idx.tolist())
        oof_actual.extend(y_val_original.tolist())
        oof_predicted.extend(y_pred.tolist())
        oof_folds.extend([fold_idx] * len(val_idx))

        logger.info(f"Fold {fold_idx + 1} RMSE: {fold_metrics['rmse']:.4f}, R²: {fold_metrics['r2']:.4f}")

    # Compute mean and std across folds
    metric_keys = [k for k in all_fold_metrics[0].keys() if all_fold_metrics[0][k] is not None]
    mean_metrics = {}
    std_metrics = {}

    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics if m.get(key) is not None]
        if values:
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)

    oof_predictions = {
        'indices': oof_indices,
        'actual': oof_actual,
        'predicted': oof_predicted,
        'fold': oof_folds,
    }

    return mean_metrics, std_metrics, all_fold_metrics, oof_predictions


###############################################################################
# FULL TRAINING DATA LOADING (with ACS split context setup)
###############################################################################

def load_full_train_data_grouped(
    data_path: str,
    feature_columns: list = None,
    target_col: str = 'median_temp',
    group_col: str = 'genus_id',
    feature_regex: str = None,
    group_nan_policy: str = GROUP_NAN_POLICY,
    split_protocol: str = CV_SPLIT_PROTOCOL,
    manifest_path: str = CV_SPLIT_MANIFEST_PATH,
    strict_dataset_match: bool = CV_STRICT_DATASET_MATCH,
    manifest_tolerance_max_missing_rows: int = CV_MANIFEST_TOLERANCE_MAX_MISSING_ROWS,
    manifest_tolerance_allow_extra_rows: bool = CV_MANIFEST_TOLERANCE_ALLOW_EXTRA_ROWS,
    manifest_tolerance_on_mismatch: str = CV_MANIFEST_TOLERANCE_ON_MISMATCH,
) -> tuple:
    """
    Load full training data without split for GroupKFold CV.

    Args:
        data_path: Path to training data file
        feature_columns: List of feature column names (preferred)
        target_col: Name of target column
        group_col: Name of group column for GroupKFold
        feature_regex: Deprecated - regex pattern to match feature columns

    Returns:
        Tuple of (X, y, X_raw, y_original, groups, feature_scaler, target_scaler, feature_names)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading full training data from {data_path}...")

    data = load_data(data_path)
    logger.info(f"Data shape: {data.shape}")

    # Use explicit column list if provided, otherwise fall back to regex
    if feature_columns is not None:
        missing_cols = [c for c in feature_columns if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        features = data[feature_columns]
    elif feature_regex is not None:
        features = data.filter(regex=feature_regex)
    else:
        raise ValueError("Either feature_columns or feature_regex must be provided")

    target = data[target_col]
    feature_names = features.columns.tolist()

    if group_col not in data.columns:
        raise ValueError(
            f"Group column '{group_col}' is required for grouped CV but was not found"
        )

    # Extract groups
    groups = data[group_col].values

    # Group NaN handling
    nan_mask = pd.isna(groups)
    if nan_mask.any():
        if group_nan_policy == 'fill_minus_one':
            logger.warning(
                "Found %d NaN values in '%s'; filling with -1 to match ACS behavior",
                int(nan_mask.sum()), group_col
            )
            groups = pd.Series(groups).fillna(-1).values
        elif group_nan_policy == 'error':
            nan_indices = np.where(nan_mask)[0].tolist()
            raise ValueError(
                f"Found {nan_mask.sum()} NaN values in group column '{group_col}' "
                f"at row indices: {nan_indices[:20]}{'...' if len(nan_indices) > 20 else ''}."
            )
        else:
            raise ValueError(
                f"Unsupported group_nan_policy '{group_nan_policy}'. "
                "Expected one of ['fill_minus_one', 'error']"
            )

    logger.info(f"Features shape: {features.shape}, Target shape: {target.shape}")
    if groups is not None:
        logger.info(f"Unique groups: {len(np.unique(groups))}")

    # Store original values
    X_raw = features.values.copy()
    y_original = target.values.copy()

    dataset_fingerprint = _build_dataset_fingerprint(
        data=data,
        data_path=data_path,
        target_col=target_col,
        group_col=group_col,
    )
    row_ids = build_row_identity({'X': X_raw, 'y': y_original, 'groups': groups})
    dataset_matches_manifest = None
    fingerprint_match = None
    manifest_missing_row_ids = None
    manifest_extra_row_ids = None
    effective_split_protocol = split_protocol

    if split_protocol in {'acs_v1_exact', 'acs_v1_tolerant'}:
        manifest = load_split_manifest(manifest_path)
        manifest_fingerprint = manifest.get('dataset_fingerprint', {})
        manifest_row_ids = manifest.get('row_ids', [])

        fingerprint_match = (
            int(manifest_fingerprint.get('row_count', -1)) == int(dataset_fingerprint['row_count']) and
            manifest_fingerprint.get('columns') == dataset_fingerprint['columns'] and
            manifest_fingerprint.get('target_col') == dataset_fingerprint['target_col'] and
            manifest_fingerprint.get('group_col') == dataset_fingerprint['group_col']
        )

        manifest_sha = manifest_fingerprint.get('sha256')
        current_sha = dataset_fingerprint.get('sha256')
        if manifest_sha and current_sha:
            fingerprint_match = fingerprint_match and (manifest_sha == current_sha)

        row_id_match = sorted(manifest_row_ids) == sorted(row_ids)
        dataset_matches_manifest = bool(row_id_match)

        manifest_missing_row_ids, manifest_extra_row_ids = _compute_manifest_row_id_deltas(
            manifest_row_ids=manifest_row_ids,
            current_row_ids=row_ids,
        )

        if dataset_matches_manifest:
            effective_split_protocol = 'acs_v1_exact'
        elif manifest_extra_row_ids:
            effective_split_protocol = 'native_groupkfold'
            _emit_split_notice(
                logger,
                (
                    "Manifest mismatch: dataset contains rows not present in manifest "
                    f"(extra={len(manifest_extra_row_ids)}, missing={len(manifest_missing_row_ids)}). "
                    "Falling back to native GroupKFold and continuing."
                ),
            )
        else:
            effective_split_protocol = 'acs_v1_tolerant'
            _emit_split_notice(
                logger,
                (
                    "Manifest mismatch: dataset is missing manifest rows "
                    f"(missing={len(manifest_missing_row_ids)}, extra={len(manifest_extra_row_ids)}). "
                    "Using manifest-intersection folds and continuing."
                ),
            )

        if split_protocol == 'acs_v1_tolerant':
            mismatch_detected = bool(manifest_missing_row_ids or manifest_extra_row_ids)
            if mismatch_detected:
                if len(manifest_missing_row_ids) > int(manifest_tolerance_max_missing_rows):
                    _emit_split_notice(
                        logger,
                        (
                            "Tolerant mismatch exceeds configured missing-row threshold "
                            f"(missing={len(manifest_missing_row_ids)}, threshold={manifest_tolerance_max_missing_rows}). "
                            "Continuing with native GroupKFold fallback."
                        ),
                    )
                    effective_split_protocol = 'native_groupkfold'
                elif manifest_extra_row_ids and not bool(manifest_tolerance_allow_extra_rows):
                    _emit_split_notice(
                        logger,
                        (
                            "Tolerant mode is configured to disallow extra dataset rows "
                            f"(extra={len(manifest_extra_row_ids)}). "
                            "Continuing with native GroupKFold fallback."
                        ),
                    )
                    effective_split_protocol = 'native_groupkfold'
                elif manifest_tolerance_on_mismatch == 'error':
                    _emit_split_notice(
                        logger,
                        "Tolerant mismatch policy is set to 'error', but continuing by request with current split mode.",
                    )

        if not dataset_matches_manifest or not fingerprint_match:
            _emit_split_notice(
                logger,
                (
                    "Dataset does not fully match manifest parity checks "
                    f"(fingerprint_match={fingerprint_match}, row_id_match={row_id_match}). "
                    "Run will continue."
                ),
            )
    elif split_protocol != 'native_groupkfold':
        raise ValueError(f"Unsupported split_protocol '{split_protocol}'")

    _set_split_context(
        row_ids=row_ids,
        dataset_fingerprint=dataset_fingerprint,
        dataset_matches_manifest=dataset_matches_manifest,
        manifest_path=manifest_path if split_protocol in {'acs_v1_exact', 'acs_v1_tolerant'} else None,
        split_protocol=effective_split_protocol,
        configured_split_protocol=split_protocol,
        effective_split_protocol=effective_split_protocol,
        manifest_missing_row_ids=manifest_missing_row_ids,
        manifest_extra_row_ids=manifest_extra_row_ids,
    )

    logger.info(
        "CV split setup: configured_protocol=%s, effective_protocol=%s, manifest=%s, dataset_match=%s, fingerprint_match=%s, row_ids=%d",
        split_protocol,
        effective_split_protocol,
        manifest_path if split_protocol in {'acs_v1_exact', 'acs_v1_tolerant'} else 'n/a',
        dataset_matches_manifest,
        fingerprint_match,
        len(row_ids),
    )

    # Scale features (global scaler for final training and test evaluation)
    feature_scaler = StandardScaler()
    X = feature_scaler.fit_transform(features)

    # Scale target (global scaler for final training and test evaluation)
    target_scaler = StandardScaler()
    y = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()

    return X, y, X_raw, y_original, groups, feature_scaler, target_scaler, feature_names


###############################################################################
# EXTENDED MODEL EVALUATION
###############################################################################

def evaluate_model_extended(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler,
    n_features: int = None,
    sample_weights: np.ndarray = None
) -> dict:
    """Evaluate model and return extended metrics including binned metrics."""
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    n_features = n_features or X_test.shape[1]

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    adj_r2 = adjusted_r2_score(y_actual, y_pred, n_features)

    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'adj_r2': adj_r2,
        'y_pred': y_pred,
        'y_actual': y_actual
    }

    if sample_weights is not None:
        metrics['weighted_rmse'] = extreme_rmse_sampled(y_actual, y_pred, sample_weights)

    binned = calculate_binned_metrics(y_actual, y_pred, n_features)
    metrics['binned'] = binned

    mask_lt_30 = y_actual < 30
    mask_gt_80 = y_actual > 80
    metrics['rmse_lt_30'] = np.sqrt(mean_squared_error(
        y_actual[mask_lt_30], y_pred[mask_lt_30])) if mask_lt_30.any() else None
    metrics['rmse_gt_80'] = np.sqrt(mean_squared_error(
        y_actual[mask_gt_80], y_pred[mask_gt_80])) if mask_gt_80.any() else None

    return metrics
