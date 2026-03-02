"""
Shared Utilities for Tabular Deep Learning Training Scripts

This module provides common functionality used across all training scripts:
- Data loading and preprocessing
- Model evaluation
- Plotting utilities
- Optuna integration helpers
- Keras callbacks
"""

import os
import json
import logging
import hashlib
import random
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import csv
import scipy.stats as stats
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

# Filter Pydantic warnings from dependencies (wandb, optuna)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

# Optional imports
OPTUNA_AVAILABLE = False
TFKerasPruningCallback = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None

# TFKerasPruningCallback may be in different locations depending on optuna version
if OPTUNA_AVAILABLE:
    try:
        from optuna.integration import TFKerasPruningCallback
    except ImportError:
        try:
            from optuna.integration.keras import TFKerasPruningCallback
        except ImportError:
            TFKerasPruningCallback = None

# Flag for explicit optuna-integration availability check
OPTUNA_INTEGRATION_AVAILABLE = TFKerasPruningCallback is not None

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

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
    """
    Set up logging configuration.

    Args:
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Configured logger
    """
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


def setup_logging_for_progress(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging to file only, keeping terminal clean for progress bars.
    
    Args:
        log_file: Path to log file (required)
        level: Logging level
        
    Returns:
        Configured logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    
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
        # Keep integer-like strings exact.
        if value.lstrip("-").isdigit():
            return str(int(value))
        # Normalize float-like strings when possible.
        try:
            as_float = float(value)
            if np.isfinite(as_float):
                return format(float(as_float), '.17g')
        except ValueError:
            pass
        return value

    # pandas NA / numpy NA handling
    try:
        if pd.isna(value):
            return "__NA__"
    except TypeError:
        pass

    return str(value)


def build_row_identity(df_or_arrays) -> list[str]:
    """
    Build stable, occurrence-aware row IDs.

    Accepts:
    - pandas.DataFrame
    - dict with keys {'X', 'y', 'groups'}
    - tuple/list in the form (X, y, groups)
    """
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
        raise ValueError("Manifest path is required for manifest-based split protocols")
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
# PROGRESS TRACKING (tqdm)
###############################################################################

class TqdmKerasCallback(keras.callbacks.Callback):
    """
    Custom Keras callback for tqdm progress bar during training.
    
    Shows epoch progress with loss, val_loss, and ETA.
    """
    
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
    """
    Get a tqdm callback for Keras training if tqdm is available.
    
    Args:
        total_epochs: Total number of epochs (for progress bar)
        desc: Description prefix for the progress bar
        
    Returns:
        TqdmKerasCallback or None if tqdm not available
    """
    if not TQDM_AVAILABLE:
        return None
    return TqdmKerasCallback(total_epochs=total_epochs, desc=desc)


def print_phase(model_name: str, phase: int, total_phases: int, description: str):
    """Print a phase indicator for progress tracking."""
    print(f"\n[{model_name}] Phase {phase}/{total_phases}: {description}")


###############################################################################
# W&B UTILITIES
###############################################################################

def add_wandb_args(parser):
    """Add common Weights & Biases CLI arguments."""
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_group', type=str, default=None, help='W&B group name')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--wandb_tags', type=str, default=None, help='Comma-separated W&B tags')


def _parse_wandb_tags(tags: str | None) -> list | None:
    if not tags:
        return None
    return [t.strip() for t in tags.split(',') if t.strip()]


def init_wandb_run(
    *,
    enabled: bool,
    project: str = 'ogtfinder-dl',
    group: str | None = None,
    name: str | None = None,
    config: dict | None = None,
    tags: list | None = None,
    output_dir: str | None = None,
    job_type: str | None = None
):
    """Initialize a W&B run if enabled and available."""
    logger = logging.getLogger(__name__)
    if not enabled:
        return None
    if not WANDB_AVAILABLE:
        logger.warning("wandb not installed; skipping W&B logging")
        return None

    api_key = os.getenv('KEY_WB_API')
    if api_key:
        try:
            wandb.login(key=api_key, relogin=True)
        except Exception as exc:
            logger.warning(f"wandb login failed: {exc}")
    else:
        logger.warning("KEY_WB_API not set; relying on existing wandb auth")

    run = wandb.init(
        project=project,
        group=group,
        name=name,
        config=config,
        tags=tags,
        dir=output_dir,
        job_type=job_type,
        reinit="finish_previous"
    )
    return run


def init_wandb_from_args(args, model_name: str, output_dir: str | None = None):
    """Initialize W&B from argparse args and defaults."""
    enabled = bool(getattr(args, 'wandb', False))
    group = getattr(args, 'wandb_group', None) or os.getenv('WANDB_GROUP')
    name = getattr(args, 'wandb_name', None) or model_name
    tags = _parse_wandb_tags(getattr(args, 'wandb_tags', None))
    config = vars(args) if hasattr(args, '__dict__') else None
    return init_wandb_run(
        enabled=enabled,
        project=os.getenv('WANDB_PROJECT', 'ogtfinder-dl'),
        group=group,
        name=name,
        config=config,
        tags=tags,
        output_dir=output_dir
    )


def log_wandb_images(run, image_paths: dict):
    """Log a dict of images to W&B using file paths (skips SVG - PIL can't read them)."""
    if run is None or not WANDB_AVAILABLE:
        return
    images = {name: wandb.Image(path) for name, path in image_paths.items() 
              if path and not path.lower().endswith('.svg')}
    if images:
        run.log(images)


def get_wandb_epoch_callback(wandb_run):
    """
    Get a W&B callback for per-epoch metric logging.
    
    Tries wandb.keras.WandbMetricsLogger first (with explicit run binding),
    falls back to custom callback.
    
    Args:
        wandb_run: Active W&B run object
        
    Returns:
        Keras callback for W&B logging, or None if W&B not available
    """
    if wandb_run is None or not WANDB_AVAILABLE:
        return None
    
    # Try built-in WandbMetricsLogger first with explicit run binding
    try:
        from wandb.integration.keras import WandbMetricsLogger
        return WandbMetricsLogger(log_freq='epoch', run=wandb_run)
    except (ImportError, AttributeError, TypeError):
        pass
    
    # Fall back to custom callback
    return WandbEpochLogger(wandb_run)


class WandbEpochLogger(keras.callbacks.Callback):
    """
    Custom callback to log train/val metrics each epoch to W&B.
    
    Logs: loss, mae, rmse, r2 for both train and validation sets.
    """
    
    def __init__(self, wandb_run):
        super().__init__()
        self.wandb_run = wandb_run
    
    def on_epoch_end(self, epoch, logs=None):
        if self.wandb_run is None or not WANDB_AVAILABLE:
            return
        
        logs = logs or {}
        
        # Map metric names to W&B-friendly names with train/val prefixes
        metric_mapping = {
            'loss': 'train/loss',
            'val_loss': 'val/loss',
            'mae': 'train/mae',
            'val_mae': 'val/mae',
            'mean_absolute_error': 'train/mae',
            'val_mean_absolute_error': 'val/mae',
            'root_mean_squared_error': 'train/rmse',
            'val_root_mean_squared_error': 'val/rmse',
            'r2': 'train/r2',
            'val_r2': 'val/r2',
            'r2_keras': 'train/r2',
            'val_r2_keras': 'val/r2',
        }
        
        payload = {'epoch': epoch + 1}
        for key, value in logs.items():
            if isinstance(value, (float, int, np.floating, np.integer)):
                wandb_key = metric_mapping.get(key, key)
                payload[wandb_key] = float(value)
        
        self.wandb_run.log(payload, step=epoch + 1)


def compute_weighted_unweighted_metrics(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray = None
) -> dict:
    """
    Compute both weighted and unweighted RMSE/MAE.
    
    Args:
        y_actual: True target values
        y_pred: Predicted values
        sample_weights: Optional sample weights array
        
    Returns:
        Dictionary with unweighted_rmse, unweighted_mae, and optionally
        weighted_rmse, weighted_mae
    """
    # Unweighted metrics (always computed)
    unweighted_rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    unweighted_mae = mean_absolute_error(y_actual, y_pred)
    
    result = {
        'unweighted_rmse': unweighted_rmse,
        'unweighted_mae': unweighted_mae,
    }
    
    # Weighted metrics (if weights provided)
    if sample_weights is not None:
        squared_errors = (y_actual - y_pred) ** 2
        weighted_mse = np.average(squared_errors, weights=sample_weights)
        weighted_rmse = np.sqrt(weighted_mse)
        
        abs_errors = np.abs(y_actual - y_pred)
        weighted_mae = np.average(abs_errors, weights=sample_weights)
        
        result['weighted_rmse'] = weighted_rmse
        result['weighted_mae'] = weighted_mae
    
    return result


def log_wandb_final_metrics(
    wandb_run,
    metrics: dict,
    sample_weights: np.ndarray = None,
    prefix: str = 'test'
):
    """
    Log final evaluation metrics to W&B as summary values.
    
    Includes both weighted and unweighted metrics when sample weights provided.
    
    Args:
        wandb_run: Active W&B run object
        metrics: Evaluation metrics dict (must include 'y_actual', 'y_pred')
        sample_weights: Optional sample weights for weighted metrics
        prefix: Metric prefix (e.g., 'test', 'train')
    """
    if wandb_run is None or not WANDB_AVAILABLE:
        return
    
    # Core metrics
    payload = {
        f'{prefix}/rmse': metrics.get('rmse'),
        f'{prefix}/mae': metrics.get('mae'),
        f'{prefix}/r2': metrics.get('r2'),
        f'{prefix}/adj_r2': metrics.get('adj_r2'),
    }
    
    # Weighted/unweighted comparison
    if 'y_actual' in metrics and 'y_pred' in metrics:
        weight_metrics = compute_weighted_unweighted_metrics(
            metrics['y_actual'], metrics['y_pred'], sample_weights
        )
        payload[f'{prefix}/unweighted_rmse'] = weight_metrics['unweighted_rmse']
        payload[f'{prefix}/unweighted_mae'] = weight_metrics['unweighted_mae']
        if 'weighted_rmse' in weight_metrics:
            payload[f'{prefix}/weighted_rmse'] = weight_metrics['weighted_rmse']
            payload[f'{prefix}/weighted_mae'] = weight_metrics['weighted_mae']
    
    # Binned metrics
    binned = metrics.get('binned', {})
    for key, value in binned.items():
        if value is not None:
            payload[f'{prefix}/{key}'] = value
    
    # Filter None values and log
    payload = {k: v for k, v in payload.items() if v is not None}
    wandb_run.log(payload)
    
    # Also log as summary for easy comparison across runs
    for key, value in payload.items():
        wandb_run.summary[key] = value


def log_wandb_diagnostics(
    wandb_run,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    prefix: str,
    suffix: str = ''
) -> dict:
    """
    Generate diagnostic plots and log to W&B as images.
    
    Args:
        wandb_run: Active W&B run object
        y_actual: True target values
        y_pred: Predicted values
        output_dir: Directory to save plot files
        prefix: Model name prefix for filenames
        suffix: Optional suffix (e.g., '_train', '_test')
        
    Returns:
        Dictionary of plot paths
    """
    if wandb_run is None or not WANDB_AVAILABLE:
        return {}
    
    paths = {}
    
    # Generate all diagnostic plots
    paths['predictions_enhanced'] = plot_predictions_enhanced(
        y_actual, y_pred, output_dir, prefix, suffix=suffix
    )
    paths['residuals_vs_predicted'] = plot_residuals_vs_predicted(
        y_actual, y_pred, output_dir, prefix
    )
    paths['residual_distribution'] = plot_residual_distribution(
        y_actual, y_pred, output_dir, prefix
    )
    paths['qq_plot'] = plot_residuals_qq(
        y_actual, y_pred, output_dir, prefix
    )
    paths['error_by_decile'] = plot_error_by_decile(
        y_actual, y_pred, output_dir, prefix, suffix=suffix
    )
    paths['calibration_curve'] = plot_calibration_curve(
        y_actual, y_pred, output_dir, prefix, suffix=suffix
    )
    
    # Log all plots to W&B (skip SVG - PIL can't read them)
    images = {}
    for name, path in paths.items():
        if path and os.path.exists(path) and not path.lower().endswith('.svg'):
            display_name = f'{prefix}{suffix}/{name}'
            images[display_name] = wandb.Image(path)
    
    if images:
        wandb_run.log(images)
    
    return paths


###############################################################################
# GPU UTILITIES
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


###############################################################################
# KERAS METRICS
###############################################################################

def r2_keras(y_true, y_pred):
    """Custom R² metric for Keras."""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# Ensure consistent metric naming across scripts
r2_keras.__name__ = 'r2'


###############################################################################
# DATA LOADING
###############################################################################

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from various file formats.

    Args:
        data_path: Path to data file (.feather, .csv, .parquet)

    Returns:
        Loaded DataFrame
    """
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
    """
    Load and preprocess test data using pre-fitted scalers.

    Args:
        data_path: Path to test data file
        feature_scaler: Pre-fitted feature scaler
        target_scaler: Pre-fitted target scaler
        feature_columns: List of feature column names (preferred)
        target_col: Name of target column
        feature_regex: Deprecated - regex pattern to match feature columns

    Returns:
        Tuple of (X_test, y_test, y_test_original)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading test data from {data_path}...")

    data = load_data(data_path)
    logger.info(f"Test data shape: {data.shape}")

    # Use explicit column list if provided, otherwise fall back to regex
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

    # Scale using pre-fitted scalers
    X_test = feature_scaler.transform(features)
    y_test_original = target.values
    y_test = target_scaler.transform(target.values.reshape(-1, 1)).flatten()

    logger.info(f"Test samples: {X_test.shape[0]}")

    return X_test, y_test, y_test_original


###############################################################################
# MODEL EVALUATION
###############################################################################


###############################################################################
# CALLBACKS
###############################################################################

def create_callbacks(
    patience_es: int = 30,
    patience_lr: int = 15,
    min_lr: float = 1e-6,
    monitor: str = 'val_loss'
) -> list:
    """
    Create standard training callbacks.

    Args:
        patience_es: Patience for early stopping
        patience_lr: Patience for learning rate reduction
        min_lr: Minimum learning rate
        monitor: Metric to monitor

    Returns:
        List of Keras callbacks
    """
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


def create_optuna_callbacks(
    trial,
    patience_es: int = 30,
    patience_lr: int = 15,
    min_lr: float = 1e-6,
    monitor: str = 'val_loss',
    use_pruning: bool = True
) -> list:
    """
    Create callbacks for Optuna hyperparameter tuning.

    Args:
        trial: Optuna trial object
        patience_es: Early stopping patience
        patience_lr: LR reduction patience
        min_lr: Minimum learning rate
        monitor: Metric to monitor
        use_pruning: Whether to include pruning callback (disable for GroupKFold CV)

    Includes pruning callback to stop unpromising trials early.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=0
        )
    ]

    if use_pruning and OPTUNA_AVAILABLE and trial is not None and TFKerasPruningCallback is not None:
        callbacks.append(TFKerasPruningCallback(trial, monitor))
    elif use_pruning and trial is not None and TFKerasPruningCallback is None:
        logger = logging.getLogger(__name__)
        logger.warning("TFKerasPruningCallback not available; pruning disabled. "
                       "Install optuna-integration for pruning support.")

    return callbacks


###############################################################################
# PLOTTING
###############################################################################

def plot_training_history(
    history,
    output_dir: str,
    prefix: str = 'model'
):
    """Plot and save training history."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{prefix}: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{prefix}_loss_curve.svg'), dpi=150)
    plt.close()


def plot_train_val_test_metric(history, output_dir: str, prefix: str, metric: str, title: str):
    """Plot train/val/test curves for a single metric on one graph. Test is optional."""
    os.makedirs(output_dir, exist_ok=True)
    metric_key = metric
    if metric == 'r2' and metric not in history.history:
        metric_key = 'r2_keras'
    if metric == 'mae' and metric not in history.history:
        metric_key = 'mean_absolute_error'

    train = history.history.get(metric_key)
    val = history.history.get(f'val_{metric_key}')
    test = history.history.get(f'test_{metric_key}')

    if train is None and val is None and test is None:
        return None

    max_len = max(
        len(train) if train is not None else 0,
        len(val) if val is not None else 0,
        len(test) if test is not None else 0
    )
    epochs = np.arange(1, max_len + 1)

    plt.figure(figsize=(10, 5))
    if train is not None:
        plt.plot(epochs[:len(train)], train, label='Train', linewidth=2)
    if val is not None:
        plt.plot(epochs[:len(val)], val, label='Validation', linewidth=2)
    # Only plot test if available (backward compatibility)
    if test is not None:
        plt.plot(epochs[:len(test)], test, label='Test', linewidth=2)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f'{prefix}_{metric}_train_val.svg')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_train_val_test_metrics(history, output_dir: str, prefix: str):
    """Plot loss, r2, mae curves with train/val on one graph each. Test included if available."""
    paths = {}
    paths['loss'] = plot_train_val_test_metric(
        history, output_dir, prefix, 'loss', f'{prefix}: Loss (Train/Val)'
    )
    paths['r2'] = plot_train_val_test_metric(
        history, output_dir, prefix, 'r2', f'{prefix}: R² (Train/Val)'
    )
    paths['mae'] = plot_train_val_test_metric(
        history, output_dir, prefix, 'mae', f'{prefix}: MAE (Train/Val)'
    )
    return paths


def plot_predictions(
    metrics: dict,
    output_dir: str,
    prefix: str = 'model'
):
    """Plot and save prediction scatter plot."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.scatter(metrics['y_actual'], metrics['y_pred'], alpha=0.5, s=10)
    min_val = min(metrics['y_actual'].min(), metrics['y_pred'].min())
    max_val = max(metrics['y_actual'].max(), metrics['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(f"{prefix}: Actual vs Predicted (R²={metrics['r2']:.4f})")
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(os.path.join(output_dir, f'{prefix}_predictions.svg'), dpi=150)
    plt.close()


def plot_residuals(
    metrics: dict,
    output_dir: str,
    prefix: str = 'model'
):
    """Plot and save residuals distribution."""
    os.makedirs(output_dir, exist_ok=True)

    residuals = metrics['y_actual'] - metrics['y_pred']
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True)
    plt.title(f'{prefix}: Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f'{prefix}_residuals.svg'), dpi=150)
    plt.close()


def plot_predictions_enhanced(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    prefix: str = 'model',
    suffix: str = ''
):
    """
    Enhanced predictions scatter with y=x line, fitted regression line, and stats.
    
    Args:
        y_actual: Actual target values
        y_pred: Predicted values
        output_dir: Output directory
        prefix: Model name prefix
        suffix: Optional suffix for filename (e.g., '_train')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Fit regression line
    slope, intercept, r_value, _, _ = stats.linregress(y_actual, y_pred)
    r2 = r_value ** 2
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_pred, alpha=0.5, s=10, label='Predictions')
    
    # Plot range
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    
    # y=x perfect fit line (dotted)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect fit (y=x)')
    
    # Fitted regression line
    x_line = np.linspace(min_val, max_val, 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r-', lw=2, label=f'Fit: y={slope:.3f}x+{intercept:.2f}')
    
    plt.title(f'{prefix}{suffix}: Actual vs Predicted\nR²={r2:.4f}, Slope={slope:.3f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'{prefix}{suffix}_predictions_enhanced.svg')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_residuals_vs_predicted(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    prefix: str = 'model'
):
    """
    Residuals vs predicted scatter to show heteroscedasticity and bias.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    residuals = y_actual - y_pred
    
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='red', linestyle='--', lw=2, label='Zero residual')
    plt.title(f'{prefix}: Residuals vs Predicted')
    plt.xlabel('Predicted')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'{prefix}_residuals_vs_predicted.svg')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_residual_distribution(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    prefix: str = 'model'
):
    """
    Residual distribution histogram with KDE and mean/median markers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    residuals = y_actual - y_pred
    mean_res = np.mean(residuals)
    median_res = np.median(residuals)
    
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True, alpha=0.7)
    plt.axvline(x=mean_res, color='red', linestyle='-', lw=2, label=f'Mean={mean_res:.2f}')
    plt.axvline(x=median_res, color='green', linestyle='--', lw=2, label=f'Median={median_res:.2f}')
    plt.title(f'{prefix}: Residual Distribution')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'{prefix}_residual_distribution.svg')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_residuals_qq(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    prefix: str = 'model'
):
    """
    QQ-plot of residuals to check normality and heavy tails.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    residuals = y_actual - y_pred
    
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{prefix}: QQ-Plot of Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'{prefix}_qq_plot.svg')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_error_by_decile(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    prefix: str = 'model',
    suffix: str = ''
):
    """
    Violin plot of errors per target decile to show model performance across range.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    errors = y_actual - y_pred
    
    # Create deciles with handling for duplicates
    try:
        deciles = pd.qcut(y_actual, 10, labels=False, duplicates='drop')
    except ValueError:
        # Fallback to equal-width bins if qcut fails
        deciles = pd.cut(y_actual, 10, labels=False)
    
    df = pd.DataFrame({
        'Decile': deciles,
        'Error': errors,
        'Actual': y_actual
    })
    
    # Get decile labels for x-axis
    decile_ranges = df.groupby('Decile')['Actual'].agg(['min', 'max'])
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Decile', y='Error', data=df, inner='box')
    plt.axhline(y=0, color='red', linestyle='--', lw=1, alpha=0.7)
    plt.title(f'{prefix}{suffix}: Error Distribution by Target Decile')
    plt.xlabel('Target Decile (0=lowest, 9=highest)')
    plt.ylabel('Error (Actual - Predicted)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'{prefix}{suffix}_error_by_decile.svg')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_calibration_curve(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    prefix: str = 'model',
    suffix: str = '',
    n_bins: int = 10
):
    """
    Calibration curve: binned average predicted vs average actual with error bars.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create bins based on predicted values
    try:
        bins = pd.qcut(y_pred, n_bins, labels=False, duplicates='drop')
    except ValueError:
        bins = pd.cut(y_pred, n_bins, labels=False)
    
    df = pd.DataFrame({
        'Bin': bins,
        'Actual': y_actual,
        'Predicted': y_pred
    })
    
    # Calculate stats per bin
    bin_stats = df.groupby('Bin').agg({
        'Actual': ['mean', 'std', 'count'],
        'Predicted': 'mean'
    }).reset_index()
    bin_stats.columns = ['Bin', 'Actual_Mean', 'Actual_Std', 'Count', 'Predicted_Mean']
    
    # Standard error for error bars
    bin_stats['SE'] = bin_stats['Actual_Std'] / np.sqrt(bin_stats['Count'])
    
    plt.figure(figsize=(8, 8))
    
    # Perfect calibration line
    min_val = min(bin_stats['Predicted_Mean'].min(), bin_stats['Actual_Mean'].min())
    max_val = max(bin_stats['Predicted_Mean'].max(), bin_stats['Actual_Mean'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect calibration')
    
    # Calibration curve with error bars
    plt.errorbar(
        bin_stats['Predicted_Mean'], 
        bin_stats['Actual_Mean'],
        yerr=bin_stats['SE'],
        fmt='o-', 
        capsize=4, 
        capthick=2,
        markersize=8,
        label='Model calibration'
    )
    
    plt.title(f'{prefix}{suffix}: Calibration Curve ({n_bins} bins)')
    plt.xlabel('Mean Predicted')
    plt.ylabel('Mean Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'{prefix}{suffix}_calibration_curve.svg')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_train_diagnostics(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    prefix: str = 'model'
):
    """
    Generate subset of diagnostic plots for training data.
    Only includes: predictions enhanced, error by decile, calibration curve.
    """
    plot_predictions_enhanced(y_actual, y_pred, output_dir, prefix, suffix='_train')
    plot_error_by_decile(y_actual, y_pred, output_dir, prefix, suffix='_train')
    plot_calibration_curve(y_actual, y_pred, output_dir, prefix, suffix='_train')


def plot_all(
    history,
    metrics: dict,
    output_dir: str,
    prefix: str = 'model'
):
    """Generate all standard plots and return paths for metric curves."""
    if history is not None:
        plot_training_history(history, output_dir, prefix)
    plot_predictions(metrics, output_dir, prefix)
    plot_residuals(metrics, output_dir, prefix)
    
    # New diagnostic plots for test data
    y_actual = metrics['y_actual']
    y_pred = metrics['y_pred']
    plot_predictions_enhanced(y_actual, y_pred, output_dir, prefix)
    plot_residuals_vs_predicted(y_actual, y_pred, output_dir, prefix)
    plot_residual_distribution(y_actual, y_pred, output_dir, prefix)
    plot_residuals_qq(y_actual, y_pred, output_dir, prefix)
    plot_error_by_decile(y_actual, y_pred, output_dir, prefix)
    plot_calibration_curve(y_actual, y_pred, output_dir, prefix)

    if history is None:
        return {}

    return plot_train_val_test_metrics(history, output_dir, prefix)


###############################################################################
# SAVING UTILITIES
###############################################################################

def save_results(
    model,
    metrics: dict,
    best_params: dict,
    output_dir: str,
    model_name: str = 'model',
    train_predictions: tuple = None
):
    """
    Save model, metrics, best hyperparameters, and predictions.

    Args:
        model: Trained Keras model
        metrics: Evaluation metrics dictionary (must include 'y_actual', 'y_pred')
        best_params: Best hyperparameters from Optuna
        output_dir: Output directory
        model_name: Name prefix for saved files
        train_predictions: Optional tuple of (y_actual, y_pred) for training data
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(__name__)

    # Save model
    model_path = os.path.join(output_dir, f'{model_name}_model.keras')
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics (exclude arrays)
    metrics_save = {
        k: float(v) if isinstance(v, (np.floating, float)) else v
        for k, v in metrics.items()
        if k not in ['y_pred', 'y_actual']
    }
    metrics_path = os.path.join(output_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_save, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Save best hyperparameters
    params_path = os.path.join(output_dir, f'{model_name}_best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Best parameters saved to {params_path}")

    # Save TEST predictions
    test_predictions_path = os.path.join(output_dir, f'{model_name}_test_predictions.csv')
    pd.DataFrame({
        'actual': metrics['y_actual'],
        'predicted': metrics['y_pred']
    }).to_csv(test_predictions_path, index=False)
    logger.info(f"Test predictions saved to {test_predictions_path}")

    # Save TRAINING predictions if provided
    if train_predictions is not None:
        train_predictions_path = os.path.join(output_dir, f'{model_name}_train_predictions.csv')
        pd.DataFrame({
            'actual': train_predictions[0],
            'predicted': train_predictions[1]
        }).to_csv(train_predictions_path, index=False)
        logger.info(f"Train predictions saved to {train_predictions_path}")


###############################################################################
# OPTUNA UTILITIES
###############################################################################

def create_study(
    study_name: str,
    direction: str = 'minimize',
    storage: str = None,
    load_if_exists: bool = True
):
    """
    Create or load an Optuna study.

    Args:
        study_name: Name of the study
        direction: 'minimize' or 'maximize'
        storage: Optional database URL for persistence
        load_if_exists: Whether to load existing study

    Returns:
        Optuna study object
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Run: pip install optuna")

    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists
    )


def run_optuna_study(
    objective_fn,
    study_name: str,
    n_trials: int = 300,
    timeout: int = None,
    n_jobs: int = 1,
    show_progress_bar: bool = True
) -> tuple:
    """
    Run Optuna hyperparameter optimization study.

    Args:
        objective_fn: Objective function for optimization
        study_name: Name of the study
        n_trials: Number of trials to run
        timeout: Optional timeout in seconds
        n_jobs: Number of parallel jobs
        show_progress_bar: Whether to show progress bar

    Returns:
        Tuple of (study, best_params, best_value)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Run: pip install optuna")

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Optuna study: {study_name} with {n_trials} trials")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    )

    # Optimize
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
        gc_after_trial=True
    )

    logger.info(f"Best trial value: {study.best_trial.value:.6f}")
    logger.info(f"Best parameters: {study.best_params}")

    return study, study.best_params, study.best_trial.value


###############################################################################
# COMMON HYPERPARAMETER SUGGESTIONS
###############################################################################

def suggest_common_hyperparams(trial) -> dict:
    """
    Suggest common hyperparameters that apply to most models.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of suggested hyperparameters
    """
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
    }


def suggest_mlp_architecture(trial, prefix: str = '') -> list:
    """
    Suggest MLP hidden layer architecture.

    Args:
        trial: Optuna trial object
        prefix: Optional prefix for parameter names

    Returns:
        List of hidden layer sizes
    """
    n_layers = trial.suggest_int(f'{prefix}n_layers', 2, 4)
    hidden_units = []

    for i in range(n_layers):
        units = trial.suggest_categorical(
            f'{prefix}hidden_units_{i}',
            [64, 128, 256, 512]
        )
        hidden_units.append(units)

    return hidden_units


###############################################################################
# SAMPLE WEIGHTS FUNCTIONS
###############################################################################

def load_sample_weights(weights_path: str) -> dict:
    """
    Load sample weights from JSON file.

    Args:
        weights_path: Path to JSON file containing sample weights

    Returns:
        Dictionary mapping target values (as strings) to weights
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading sample weights from {weights_path}...")

    with open(weights_path, 'r') as f:
        weights_dict = json.load(f)

    logger.info(f"Loaded weights for {len(weights_dict)} target value bins")
    return weights_dict


def get_sample_weights_array(y: np.ndarray, weights_dict: dict) -> np.ndarray:
    """
    Convert target values to sample weights array for Keras training.

    Maps each target value to its corresponding weight based on binning.
    Uses nearest bin matching for continuous target values.

    Args:
        y: Target values array (original scale, not scaled)
        weights_dict: Dictionary mapping bin values to weights

    Returns:
        Array of sample weights matching y shape
    """
    # Convert dict keys to floats and sort
    bin_values = np.array(sorted([float(k) for k in weights_dict.keys()]))
    bin_weights = np.array([weights_dict[str(k) if str(k) in weights_dict else str(int(k))]
                           for k in bin_values])

    # For each y value, find the nearest bin
    sample_weights = np.zeros(len(y))
    for i, val in enumerate(y):
        # Find nearest bin index
        idx = np.abs(bin_values - val).argmin()
        sample_weights[i] = bin_weights[idx]

    return sample_weights


def extreme_rmse_sampled(y_true: np.ndarray, y_pred: np.ndarray,
                         sample_weights: np.ndarray) -> float:
    """
    Compute weighted RMSE using pre-computed sample weights.

    Args:
        y_true: True target values
        y_pred: Predicted values
        sample_weights: Pre-computed sample weights array

    Returns:
        Weighted RMSE value
    """
    squared_errors = (y_true - y_pred) ** 2
    weighted_mse = np.average(squared_errors, weights=sample_weights)
    return np.sqrt(weighted_mse)


###############################################################################
# ADJUSTED METRICS
###############################################################################

def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Compute adjusted R² accounting for feature count.

    Args:
        y_true: True target values
        y_pred: Predicted values
        n_features: Number of features used in the model

    Returns:
        Adjusted R² value
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)

    if n <= n_features + 1:
        return r2  # Cannot compute adjusted R² with insufficient samples

    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2


def calculate_binned_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             n_features: int) -> dict:
    """
    Calculate metrics for top/bottom 5%, 10%, 20% percentile bins.

    Args:
        y_true: True target values
        y_pred: Predicted values
        n_features: Number of features for adjusted R²

    Returns:
        Dictionary with binned metrics for each percentile
    """
    results = {}

    for pct in [5, 10, 20]:
        # Bottom percentile
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

        # Top percentile
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
# GROUPKFOLD FOR KERAS
###############################################################################

def create_groupkfold_objective_keras(
    build_model_fn,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray = None,
    n_splits: int = 5,
    batch_size: int = 32,
    max_epochs: int = 500,
    patience_es: int = 30,
    patience_lr: int = 15,
    compile_fn=None,
    random_state: int = RANDOM_STATE,
    split_protocol: str = CV_SPLIT_PROTOCOL,
    manifest_path: str = CV_SPLIT_MANIFEST_PATH,
    strict: bool = CV_STRICT_DATASET_MATCH,
):
    """
    Create Optuna objective using GroupKFold CV with weighted RMSE.

    Scaling is performed per-fold to avoid information leakage: the scaler
    is fit on the training fold only and applied to the validation fold.

    Args:
        build_model_fn: Function(trial, input_dim) -> compiled Keras model
        X: Feature array (raw, unscaled)
        y: Target array (raw, unscaled)
        groups: Group labels for GroupKFold
        sample_weights: Pre-computed sample weights (optional)
        n_splits: Number of CV folds
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        patience_es: Early stopping patience
        patience_lr: LR reduction patience
        compile_fn: Optional function(model, trial) to compile model

    Returns:
        Objective function for Optuna
    """
    fold_splits = get_stage_folds(
        stage='optuna_objective',
        X=X,
        y=y,
        groups=groups,
        random_state=random_state,
        n_splits=n_splits,
        manifest_path=manifest_path,
        protocol=split_protocol,
        strict=strict,
    )

    def objective(trial):
        keras.backend.clear_session()
        set_global_seeds(random_state)

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            # Per-fold scaling to avoid information leakage
            fold_feature_scaler = StandardScaler()
            X_train_fold = fold_feature_scaler.fit_transform(X[train_idx])
            X_val_fold = fold_feature_scaler.transform(X[val_idx])

            fold_target_scaler = StandardScaler()
            y_train_fold = fold_target_scaler.fit_transform(y[train_idx].reshape(-1, 1)).flatten()
            y_val_fold = fold_target_scaler.transform(y[val_idx].reshape(-1, 1)).flatten()

            # Get sample weights for this fold
            sw_train = sample_weights[train_idx] if sample_weights is not None else None

            # Build model
            input_dim = X.shape[1]
            model = build_model_fn(trial, input_dim)

            if compile_fn is not None:
                compile_fn(model, trial)

            # Callbacks - disable pruning for GroupKFold CV to avoid duplicate step warnings
            callbacks = create_optuna_callbacks(
                trial,
                patience_es=patience_es,
                patience_lr=patience_lr,
                use_pruning=False
            )

            # Train
            model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=max_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                sample_weight=sw_train,
                verbose=0
            )

            # Evaluate - use validation loss
            val_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            if isinstance(val_loss, list):
                val_loss = val_loss[0]  # First element is the loss

            fold_scores.append(val_loss)

            # Clear memory
            keras.backend.clear_session()

        # Return mean validation loss across folds
        return np.mean(fold_scores)

    return objective


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

    Scaling is performed per-fold to avoid information leakage: the scaler
    is fit on the training fold only and applied to the validation fold.

    Args:
        model_builder_fn: Function(params, input_dim) -> compiled Keras model
        best_params: Best hyperparameters from Optuna
        X: Feature array (raw, unscaled)
        y: Target array (raw, unscaled)
        groups: Group labels for GroupKFold
        sample_weights: Pre-computed sample weights
        n_splits: Number of CV folds
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        patience_es: Early stopping patience
        patience_lr: LR reduction patience
        n_features: Number of features for adjusted R²

    Returns:
        Tuple of (mean_metrics, std_metrics, all_fold_metrics)
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

        # Weighted RMSE if weights available
        if sw_val is not None:
            fold_metrics['weighted_rmse'] = extreme_rmse_sampled(y_val_original, y_pred, sw_val)

        # Binned metrics
        binned = calculate_binned_metrics(y_val_original, y_pred, n_features)
        fold_metrics.update(binned)

        all_fold_metrics.append(fold_metrics)

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

    return mean_metrics, std_metrics, all_fold_metrics


###############################################################################
# CSV EXPORT
###############################################################################

TRACKING_METRIC_COLUMNS = [
    'rmse', 'mae', 'r2', 'adj_r2', 'weighted_rmse',
    'rmse_bottom_5', 'mae_bottom_5', 'r2_bottom_5', 'adj_r2_bottom_5', 'n_bottom_5',
    'rmse_top_5', 'mae_top_5', 'r2_top_5', 'adj_r2_top_5', 'n_top_5',
    'rmse_bottom_10', 'mae_bottom_10', 'r2_bottom_10', 'adj_r2_bottom_10', 'n_bottom_10',
    'rmse_top_10', 'mae_top_10', 'r2_top_10', 'adj_r2_top_10', 'n_top_10',
    'rmse_bottom_20', 'mae_bottom_20', 'r2_bottom_20', 'adj_r2_bottom_20', 'n_bottom_20',
    'rmse_top_20', 'mae_top_20', 'r2_top_20', 'adj_r2_top_20', 'n_top_20'
]


def save_performance_to_csv(
    file_path: str,
    model_name: str,
    experiment_name: str,
    dataset_type: str,
    overall_metrics: dict,
    binned_metrics: dict = None,
    append: bool = True
):
    """
    Save metrics to CSV with binned columns.

    Args:
        file_path: Path to CSV file
        model_name: Name of the model
        experiment_name: Name of the experiment
        dataset_type: 'train', 'val', or 'test'
        overall_metrics: Dictionary with overall metrics (rmse, mae, r2, adj_r2)
        binned_metrics: Optional dictionary with binned metrics
        append: If True, append to existing file; if False, overwrite
    """
    logger = logging.getLogger(__name__)

    # Build row data
    row = {
        'model_name': model_name,
        'experiment_name': experiment_name,
        'dataset_type': dataset_type,
        'rmse': overall_metrics.get('rmse'),
        'mae': overall_metrics.get('mae'),
        'r2': overall_metrics.get('r2'),
        'adj_r2': overall_metrics.get('adj_r2'),
        'weighted_rmse': overall_metrics.get('weighted_rmse')
    }

    # Add binned metrics
    if binned_metrics:
        row.update(binned_metrics)

    # Determine if file exists and get headers
    file_exists = os.path.exists(file_path)
    existing_headers = []

    if file_exists and append:
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            existing_headers = reader.fieldnames or []

    # Merge headers
    all_headers = list(row.keys())
    if existing_headers:
        # Add any new columns to the end
        for h in all_headers:
            if h not in existing_headers:
                existing_headers.append(h)
        all_headers = existing_headers

    # Write/append to CSV
    mode = 'a' if (file_exists and append) else 'w'
    write_header = not (file_exists and append)

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)

    with open(file_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_headers, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Saved performance metrics to {file_path}")


def save_perfold_performance_csv(
    file_path: str,
    model_name: str,
    experiment_name: str,
    all_fold_metrics: list
):
    """
    Append one row per CV fold to model_perfold_performance.csv.

    Args:
        file_path: Target CSV path
        model_name: Name of the model
        experiment_name: Name of the experiment
        all_fold_metrics: List of fold metrics from groupkfold_cross_validate
    """
    logger = logging.getLogger(__name__)

    if not all_fold_metrics:
        logger.warning("No fold metrics provided; skipping per-fold CSV write.")
        return

    fieldnames = ['model_name', 'experiment_name', 'fold', 'n_train', 'n_val'] + TRACKING_METRIC_COLUMNS
    rows = []
    for fold_metrics in all_fold_metrics:
        row = {
            'model_name': model_name,
            'experiment_name': experiment_name,
            'fold': fold_metrics.get('fold'),
            'n_train': fold_metrics.get('n_train'),
            'n_val': fold_metrics.get('n_val')
        }
        for metric_key in TRACKING_METRIC_COLUMNS:
            row[metric_key] = fold_metrics.get(metric_key)
        rows.append(row)

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} per-fold rows to {file_path}")


def save_cv_performance_csv(
    file_path: str,
    model_name: str,
    experiment_name: str,
    mean_metrics: dict,
    std_metrics: dict
):
    """
    Append CV mean/std summary row to model_cv_performance.csv.

    Args:
        file_path: Target CSV path
        model_name: Name of the model
        experiment_name: Name of the experiment
        mean_metrics: Mean metrics from groupkfold_cross_validate
        std_metrics: Std metrics from groupkfold_cross_validate
    """
    logger = logging.getLogger(__name__)

    fieldnames = ['model_name', 'experiment_name']
    for metric_key in TRACKING_METRIC_COLUMNS:
        fieldnames.extend([f'{metric_key}_mean', f'{metric_key}_std'])

    row = {
        'model_name': model_name,
        'experiment_name': experiment_name
    }
    for metric_key in TRACKING_METRIC_COLUMNS:
        row[f'{metric_key}_mean'] = mean_metrics.get(metric_key)
        row[f'{metric_key}_std'] = std_metrics.get(metric_key)

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Saved CV mean/std row to {file_path}")


###############################################################################
# EXTENDED DATA LOADING WITH GROUPS
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
    """
    Evaluate model and return extended metrics including binned metrics.

    Args:
        model: Trained Keras model
        X_test: Test features (scaled)
        y_test: Test targets (scaled)
        target_scaler: Scaler for inverse transform
        n_features: Number of features for adjusted R²
        sample_weights: Optional sample weights for weighted RMSE

    Returns:
        Dictionary with evaluation metrics including binned metrics
    """
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    n_features = n_features or X_test.shape[1]

    # Compute overall metrics
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

    # Weighted RMSE
    if sample_weights is not None:
        metrics['weighted_rmse'] = extreme_rmse_sampled(y_actual, y_pred, sample_weights)

    # Binned metrics
    binned = calculate_binned_metrics(y_actual, y_pred, n_features)
    metrics['binned'] = binned

    # Legacy extreme value metrics for backward compatibility
    mask_lt_30 = y_actual < 30
    mask_gt_80 = y_actual > 80
    metrics['rmse_lt_30'] = np.sqrt(mean_squared_error(
        y_actual[mask_lt_30], y_pred[mask_lt_30])) if mask_lt_30.any() else None
    metrics['rmse_gt_80'] = np.sqrt(mean_squared_error(
        y_actual[mask_gt_80], y_pred[mask_gt_80])) if mask_gt_80.any() else None

    return metrics


def log_metrics_extended(metrics: dict, prefix: str = "Test"):
    """Log extended evaluation metrics."""
    logger = logging.getLogger(__name__)
    logger.info(f"{prefix} RMSE: {metrics['rmse']:.4f}")
    logger.info(f"{prefix} MAE: {metrics['mae']:.4f}")
    logger.info(f"{prefix} R²: {metrics['r2']:.4f}")
    logger.info(f"{prefix} Adjusted R²: {metrics['adj_r2']:.4f}")

    if metrics.get('weighted_rmse'):
        logger.info(f"{prefix} Weighted RMSE: {metrics['weighted_rmse']:.4f}")

    if metrics.get('rmse_lt_30'):
        logger.info(f"{prefix} RMSE (<30): {metrics['rmse_lt_30']:.4f}")
    if metrics.get('rmse_gt_80'):
        logger.info(f"{prefix} RMSE (>80): {metrics['rmse_gt_80']:.4f}")

    # Log binned metrics if available
    binned = metrics.get('binned', {})
    for pct in [5, 10, 20]:
        if binned.get(f'rmse_bottom_{pct}') is not None:
            logger.info(f"{prefix} Bottom {pct}% RMSE: {binned[f'rmse_bottom_{pct}']:.4f}")
        if binned.get(f'rmse_top_{pct}') is not None:
            logger.info(f"{prefix} Top {pct}% RMSE: {binned[f'rmse_top_{pct}']:.4f}")
