"""Configuration for DL-only stacking ensemble with from-scratch training."""

from __future__ import annotations

from pathlib import Path


# Repository root and default directories
REPO_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = IMPLEMENTATION_ROOT / "out"

# Data lives at project root, shared across all model directories
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Data files
TRAINING_FILE = "train_genus.csv"
TESTING_FILE = "test_genus.csv"
WEIGHTS_FILE = "weights_train_genus.json"

# Core columns
TARGET = "median_temp"
GROUP_COLUMN = "genus_id"

# Deep-learning strict feature list (58 features)
FEATURE_COLUMNS = [
    "B1_mean",
    "B2_mean",
    "B3_mean",
    "B4_mean",
    "B5_mean",
    "B6_mean",
    "B7_mean",
    "B8_mean",
    "B9_mean",
    "B10_mean",
    "PP1_mean",
    "PP2_mean",
    "PP3_mean",
    "F1_mean",
    "F2_mean",
    "F3_mean",
    "F4_mean",
    "F5_mean",
    "F6_mean",
    "K1_mean",
    "K2_mean",
    "K3_mean",
    "K4_mean",
    "K5_mean",
    "K6_mean",
    "K7_mean",
    "K8_mean",
    "K9_mean",
    "K10_mean",
    "MSWHIM1_mean",
    "MSWHIM2_mean",
    "MSWHIM3_mean",
    "ST1_mean",
    "ST2_mean",
    "ST3_mean",
    "ST4_mean",
    "ST5_mean",
    "ST6_mean",
    "ST7_mean",
    "ST8_mean",
    "T1_mean",
    "T2_mean",
    "T3_mean",
    "T4_mean",
    "T5_mean",
    "VHSE1_mean",
    "VHSE2_mean",
    "VHSE3_mean",
    "VHSE4_mean",
    "VHSE5_mean",
    "VHSE6_mean",
    "VHSE7_mean",
    "VHSE8_mean",
    "Z1_mean",
    "Z2_mean",
    "Z3_mean",
    "Z4_mean",
    "Z5_mean",
]

# Ensemble execution defaults
SEED = 2026
N_CV = 5
N_TRIALS = 300
EXPERIMENT_NAME_DEFAULT = "genus_exp"

# ---------------------------------------------------------------------------
# DL training constants (from deep_learning/config.py)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE_ES = 30
PATIENCE_LR = 15
RANDOM_STATE = 2024  # DL seed (distinct from ensemble SEED=2026)

# Split protocol configuration (tolerant ACS mode by default)
CV_SPLIT_PROTOCOL = "acs_v1_tolerant"
CV_SPLIT_MANIFEST_PATH = str(
    REPO_ROOT / "deep_learning" / "splits" / "acs_groupkfold_v1.json"
)

# Tolerant manifest mismatch settings
CV_MANIFEST_TOLERANCE_MAX_MISSING_ROWS = 5
CV_MANIFEST_TOLERANCE_ALLOW_EXTRA_ROWS = False
CV_MANIFEST_TOLERANCE_ON_MISMATCH = "warn"
GROUP_NAN_POLICY = "fill_minus_one"
CV_STRICT_DATASET_MATCH = True

# Default location for best_params.json files from deep_learning
DL_PARAMS_ROOT = REPO_ROOT / "deep_learning" / "out"
ALL_BEST_PARAMS_FILE = IMPLEMENTATION_ROOT / "all_best_params.json"

# ---------------------------------------------------------------------------
# Output file names
# ---------------------------------------------------------------------------
MODEL_PERFORMANCE_FILE = "model_performance.csv"
MODEL_PERFOLD_PERFORMANCE_FILE = "model_perfold_performance.csv"
MODEL_CV_PERFORMANCE_FILE = "model_cv_performance.csv"
PIPELINE_FILE_SUFFIX = "_pipeline.pkl"

# Deep-learning style output columns (order matters)
DEEP_LEARNING_PERF_COLUMNS = [
    "model_name",
    "experiment_name",
    "dataset_type",
    "rmse",
    "mae",
    "r2",
    "adj_r2",
    "weighted_rmse",
    "rmse_bottom_5",
    "mae_bottom_5",
    "r2_bottom_5",
    "adj_r2_bottom_5",
    "n_bottom_5",
    "rmse_top_5",
    "mae_top_5",
    "r2_top_5",
    "adj_r2_top_5",
    "n_top_5",
    "rmse_bottom_10",
    "mae_bottom_10",
    "r2_bottom_10",
    "adj_r2_bottom_10",
    "n_bottom_10",
    "rmse_top_10",
    "mae_top_10",
    "r2_top_10",
    "adj_r2_top_10",
    "n_top_10",
    "rmse_bottom_20",
    "mae_bottom_20",
    "r2_bottom_20",
    "adj_r2_bottom_20",
    "n_bottom_20",
    "rmse_top_20",
    "mae_top_20",
    "r2_top_20",
    "adj_r2_top_20",
    "n_top_20",
]
