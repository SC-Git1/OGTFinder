"""
Centralized Configuration for Tabular Deep Learning Training Scripts

This module provides a single source of truth for all paths, settings, and metadata
used across the training scripts, following the ACS_draft approach.
"""

###############################################################################
# DATA PATHS (configure per environment)
###############################################################################

from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]

DATA_PATH = str(_REPO_ROOT / "data")
MODEL_SAVE_PATH = "./models"
MODEL_PERFORMANCE_PATH = "./out"
LOGS_PATH = "./logs"

###############################################################################
# DATA FILES
###############################################################################

TRAINING_FILE = "train_genus.csv"
TESTING_FILE = "test_genus.csv"
WEIGHTS_FILE = "weights_train_genus.json"

###############################################################################
# TARGET AND FEATURES
###############################################################################

TARGET = "median_temp"
GROUP_COLUMN = "genus_id"

# Explicit feature columns (58 features)
FEATURE_COLUMNS = [
    "B1_mean", "B2_mean", "B3_mean", "B4_mean", "B5_mean",
    "B6_mean", "B7_mean", "B8_mean", "B9_mean", "B10_mean",
    "PP1_mean", "PP2_mean", "PP3_mean",
    "F1_mean", "F2_mean", "F3_mean", "F4_mean", "F5_mean", "F6_mean",
    "K1_mean", "K2_mean", "K3_mean", "K4_mean", "K5_mean",
    "K6_mean", "K7_mean", "K8_mean", "K9_mean", "K10_mean",
    "MSWHIM1_mean", "MSWHIM2_mean", "MSWHIM3_mean",
    "ST1_mean", "ST2_mean", "ST3_mean", "ST4_mean",
    "ST5_mean", "ST6_mean", "ST7_mean", "ST8_mean",
    "T1_mean", "T2_mean", "T3_mean", "T4_mean", "T5_mean",
    "VHSE1_mean", "VHSE2_mean", "VHSE3_mean", "VHSE4_mean",
    "VHSE5_mean", "VHSE6_mean", "VHSE7_mean", "VHSE8_mean",
    "Z1_mean", "Z2_mean", "Z3_mean", "Z4_mean", "Z5_mean"
]

###############################################################################
# TRAINING SETTINGS
###############################################################################

BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE_ES = 30
PATIENCE_LR = 15

###############################################################################
# CROSS-VALIDATION
###############################################################################

N_CV = 5
RANDOM_STATE = 2024

# Split protocol configuration
# - "acs_v1_exact": enforce exact ACS fold membership using manifest row IDs
# - "acs_v1_tolerant": allow small manifest row drift and use manifest intersection
# - "native_groupkfold": fallback to in-code GroupKFold splitting
CV_SPLIT_PROTOCOL = "acs_v1_tolerant"
CV_SPLIT_MANIFEST_PATH = str(_Path(__file__).resolve().parent / "splits" / "acs_groupkfold_v1.json")

# Tolerant ACS parity settings:
# - allow up to this many manifest rows to be missing from current dataset
# - reject datasets that contain rows unknown to the manifest
CV_MANIFEST_TOLERANCE_MAX_MISSING_ROWS = 5
CV_MANIFEST_TOLERANCE_ALLOW_EXTRA_ROWS = False

# Behavior when a tolerated mismatch is detected:
# - "warn": continue with manifest-intersection folds
# - "error": fail even when mismatch is within tolerance
CV_MANIFEST_TOLERANCE_ON_MISMATCH = "warn"

# Group NaN policy for grouped CV:
# - "fill_minus_one": ACS-compatible behavior
# - "error": fail when NaN groups are present
GROUP_NAN_POLICY = "fill_minus_one"

# In ACS exact mode, fail fast when dataset fingerprint/row identities mismatch
CV_STRICT_DATASET_MATCH = True

###############################################################################
# OPTUNA SETTINGS
###############################################################################

N_TRIALS = 300
OPTUNA_TIMEOUT = None

###############################################################################
# EXPERIMENT NAMING
###############################################################################

EXPERIMENT_NAME_DEFAULT = "genus_exp"
MODEL_PERFORMANCE_NAME = "model_performance.csv"
