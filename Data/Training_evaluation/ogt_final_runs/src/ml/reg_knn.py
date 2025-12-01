"""
REGRESSION WITH KNN MODEL

REQUIRES CONFIG_MODELS, UTILS_ML, CONFIG
"""
import contextlib
import csv
import json
import logging
import os
import sys
import joblib
import traceback
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from datetime import datetime
from optuna.logging import get_logger
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle
from sklearn.model_selection import GroupKFold

# IMPORT CONFIG
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.append(project_root)

from config_models import N_CV, N_JOBS, N_TRIALS, RANDOM_STATE

from utils_ml import (
    adjusted_r2_score,
    adjusted_rmse,
    calculate_binned_metrics,
    save_performance_to_csv,
    extreme_rmse,
    extreme_rmse_sampled,
)
from ogt_final_runs.config import (
    DATA_PATH,
    EXPERIMENT_NAME_DEFAULT,
    HYPERPARAMETERS_JSON_NAME,
    LOGS_PATH,
    META_DATA_COLUMNS,
    MODEL_HYPERPARAMETER_PERFORMANCE_NAME,
    MODEL_PERFORMANCE_NAME,
    MODEL_PERFORMANCE_PATH,
    MODEL_SAVE_PATH,
    TARGET,
    TESTING_FILE,
    TRAINING_FILE,
    WEIGHTS_FILE,
)

################################################################################
################################################################################
################################################################################
# LOG SETUP



os.makedirs(LOGS_PATH, exist_ok=True)
log_file_name = f"knn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_path = os.path.join(LOGS_PATH, log_file_name)
# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)
logging.info("KNN training script started")


################################################################################
################################################################################
################################################################################
# FILE SETUP


# MODEL NAME
MODEL_NAME = "knn_rmse.pkl"
MODEL_FOLDER = "knn_rmse"
MODEL_EXP_NAME = f"{EXPERIMENT_NAME_DEFAULT}-{MODEL_FOLDER}"
# PREPARING DIRS
exp_eval_dir = os.path.join(MODEL_PERFORMANCE_PATH, EXPERIMENT_NAME_DEFAULT)
exp_model_dir = os.path.join(MODEL_SAVE_PATH, EXPERIMENT_NAME_DEFAULT, MODEL_FOLDER)
os.makedirs(exp_eval_dir, exist_ok=True)
os.makedirs(exp_model_dir, exist_ok=True)
# PREPARING FILES AND PATHS TO SAVE TO
training_file = os.path.join(DATA_PATH, TRAINING_FILE)
testing_file = os.path.join(DATA_PATH, TESTING_FILE)
weights_file = os.path.join(DATA_PATH, WEIGHTS_FILE)
model_save_file = os.path.join(exp_model_dir, MODEL_NAME)
model_hyperparameter_performance_file = os.path.join(
    MODEL_PERFORMANCE_PATH,
    EXPERIMENT_NAME_DEFAULT,
    MODEL_HYPERPARAMETER_PERFORMANCE_NAME,
)
model_performance_file = os.path.join(
    MODEL_PERFORMANCE_PATH, EXPERIMENT_NAME_DEFAULT, MODEL_PERFORMANCE_NAME
)
hyperparameters_json_file = os.path.join(
    MODEL_PERFORMANCE_PATH, EXPERIMENT_NAME_DEFAULT, HYPERPARAMETERS_JSON_NAME
)


################################################################################
################################################################################
################################################################################
# MODEL TRAINING


def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)

        groups = list(df["genus_id"].fillna(-1))

        columns_to_drop = [TARGET] + [
            col for col in META_DATA_COLUMNS if col in df.columns
        ]
        y = df[TARGET]
        X = df.drop(columns_to_drop, axis=1)
        return X, y, groups
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def objective(trial, X, y, groups):
    param = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
    }
    model = KNeighborsRegressor(**param)

    # sample weights is a dictionary defined in if __name == "__main__"
    def custom_scorer(estimator, X, y, sample_weights = sample_weights):
        y_pred = estimator.predict(X)
        return -extreme_rmse_sampled(y, y_pred, X.shape[1], sample_weights)


    score = cross_val_score(
        model, X, y, cv=GroupKFold(n_splits=N_CV), scoring=custom_scorer, n_jobs=N_JOBS, groups = groups,
    )
    rmse = -score.mean()
    return rmse


def cross_validate_with_optimal_params(X, y, best_params, groups, cv=N_CV):

    X, y, groups = shuffle(X, y, groups, random_state=RANDOM_STATE)

    kf = GroupKFold(n_splits=cv)
    cv_predictions = []
    cv_true_values = []
    for train_index, val_index in kf.split(X, groups = groups):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = KNeighborsRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        cv_predictions.append(y_pred)
        cv_true_values.append(y_val)
    return cv_true_values, cv_predictions


################################################################################
################################################################################
################################################################################
# MAIN - RUNNING ML


if __name__ == "__main__":

    with open(weights_file, "r") as infile:
        sample_weights = json.load(infile)

    sample_weights = {float(i):float(j) for i,j in sample_weights.items()}

    try:
        # Load data
        X_train_unscaled, y_train, groups_train = load_and_preprocess_data(training_file)
        X_test_unscaled, y_test, groups_test = load_and_preprocess_data(testing_file)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_unscaled)
        X_test = scaler.transform(X_test_unscaled)

        # Set up Optuna logger
        optuna_logger = get_logger("optuna")
        optuna_logger.addHandler(logging.FileHandler(log_file_path))
        optuna_logger.setLevel(logging.INFO)

        # Optuna study
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed = RANDOM_STATE))
        logging.info("Starting Optuna hyperparameter optimization")
        try:
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, groups_train),
                n_trials=N_TRIALS
            )
        except Exception as e:
            logging.error(f"An error occurred during optimization: {str(e)}")
        # Log best trial information
        logging.info(f"Best trial:")
        logging.info(f"  Value: {study.best_trial.value}")
        logging.info(f"  Params: {study.best_trial.params}")

        # Get the best hyperparameters
        best_params = study.best_trial.params
        logging.info(f"BEST TRIAL PARAMETERS: {best_params}")
        new_hyperparameters = {
            f"{EXPERIMENT_NAME_DEFAULT}": {
                f"{MODEL_FOLDER}": {
                    "n_neighbors": best_params["n_neighbors"],
                    "weights": best_params["weights"],
                    "p": best_params["p"],
                }
            }
        }

        # Load existing hyperparameters from JSON file
        if os.path.exists(hyperparameters_json_file):
            with open(hyperparameters_json_file, "r") as f:
                hyperparameters = json.load(f)
        else:
            hyperparameters = {}
        # Update hyperparameters
        if EXPERIMENT_NAME_DEFAULT in hyperparameters:
            hyperparameters[EXPERIMENT_NAME_DEFAULT].update(
                new_hyperparameters[EXPERIMENT_NAME_DEFAULT]
            )
        else:
            hyperparameters.update(new_hyperparameters)
        # Save to json
        with open(hyperparameters_json_file, "w") as f:
            json.dump(hyperparameters, f, indent=4)
        logging.info(f"Hyperparameters saved to {hyperparameters_json_file}")

        # Train final model
        final_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", KNeighborsRegressor(**best_params)),
            ]
        )
        final_model.fit(X_train_unscaled, y_train)
        joblib.dump(final_model, model_save_file)
        logging.info(f"Final model saved to {model_save_file}")

        # Log number of observations
        total_obs = len(X_train) + len(X_test)
        train_obs = len(X_train)
        logging.info(f"Total number of observations: {total_obs}")
        logging.info(f"Number of observations in 5% group: {int(0.05 * total_obs)}")
        logging.info(f"Number of observations in 10% group: {int(0.10 * total_obs)}")
        logging.info(f"Number of observations in 20% group: {int(0.20 * total_obs)}")
        logging.info(f"Number of observations in training set: {train_obs}")
        logging.info(f"Number of observations in testing set: {len(X_test)}")
        # Log number of observations during cross-validation
        cv_train = int(train_obs * (N_CV - 1) / N_CV)
        cv_val = train_obs - cv_train
        logging.info(f"During {N_CV}-fold cross-validation:")
        logging.info(f"  Number of observations in each training fold: {cv_train}")
        logging.info(f"  Number of observations in each validation fold: {cv_val}")
        logging.info(f"  CV training set 5% group: {int(0.05 * cv_train)}")
        logging.info(f"  CV training set 10% group: {int(0.10 * cv_train)}")
        logging.info(f"  CV training set 20% group: {int(0.20 * cv_train)}")
        logging.info(f"  CV validation set 5% group: {int(0.05 * cv_val)}")
        logging.info(f"  CV validation set 10% group: {int(0.10 * cv_val)}")
        logging.info(f"  CV validation set 20% group: {int(0.20 * cv_val)}")

        # Evaluation
        for dataset_name, X, y in [
            ("Testing", X_test_unscaled, y_test),
            ("Cross-Validation", X_train_unscaled, y_train),
        ]:
            save_path = os.path.join(exp_model_dir, dataset_name.lower())
            os.makedirs(save_path, exist_ok=True)

            if dataset_name == "Cross-Validation":
                yCV, y_predCV = cross_validate_with_optimal_params(X, y, best_params, groups_train)
                y = np.concatenate(yCV).ravel()
                y_pred = np.concatenate(y_predCV).ravel()
            else:
                y_pred = final_model.predict(X)

            # Calculate and log performance metrics
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            adj_r2 = adjusted_r2_score(y, y_pred, X.shape[1])
            overall_metrics = {
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Adjusted R2": adj_r2,
            }
            # Calculate and log binned performance metrics

            if dataset_name == "Cross-Validation":
                rmse_mean = np.mean([np.sqrt(mean_squared_error(a, b)) for a,b in zip(yCV,y_predCV)])
                rmse_std = np.std([np.sqrt(mean_squared_error(a, b)) for a,b in zip(yCV,y_predCV)])
                mae_mean = np.mean([np.sqrt(mean_absolute_error(a, b)) for a,b in zip(yCV,y_predCV)])
                mae_std = np.std([np.sqrt(mean_absolute_error(a, b)) for a,b in zip(yCV,y_predCV)])
                r2_mean = np.mean([np.sqrt(r2_score(a, b)) for a,b in zip(yCV,y_predCV)])
                r2_std = np.std([np.sqrt(r2_score(a, b)) for a,b in zip(yCV,y_predCV)])
                adj_r2_mean = np.mean([np.sqrt(adjusted_r2_score(a, b,X.shape[1])) for a,b in zip(yCV,y_predCV)])
                adj_r2_std = np.std([np.sqrt(adjusted_r2_score(a, b,X.shape[1])) for a,b in zip(yCV,y_predCV)])
                CV_metrics = {
                "RMSE": rmse_mean,
                "MAE": mae_mean,
                "R2": r2_mean,
                "Adjusted R2": adj_r2_mean,
                "RMSE std": rmse_std,
                "MAE std": mae_std,
                "R2 std": r2_std,
                "Adjusted R2 std": adj_r2_std,
                }

            binned_metrics = calculate_binned_metrics(y, y_pred, X)
            restructured_binned_metrics = {
                bin_name: {
                    "RMSE": metrics[0],
                    "MAE": metrics[1],
                    "R2": metrics[2],
                    "Adjusted R2": metrics[3],
                }
                for bin_name, metrics in binned_metrics.items()
            }
            save_performance_to_csv(
                model_performance_file,
                MODEL_NAME.replace(".pkl", ""),
                MODEL_EXP_NAME,
                dataset_name,
                overall_metrics,
                restructured_binned_metrics,
            )

            if dataset_name == "Cross-Validation":
                  save_performance_to_csv(
                  os.path.join(os.path.dirname(model_performance_file), "CV_metrics.csv"),
                  MODEL_NAME.replace(".pkl",""),
                  MODEL_EXP_NAME,
                  dataset_name,
                  CV_metrics,
                )

            # Log performance metrics
            logging.info(f"{dataset_name} Performance:")
            for metric, value in overall_metrics.items():
                logging.info(f"{metric}: {value:.4f}")
            logging.info(f"{dataset_name} Binned Performance:")
            for bin_name, metrics in restructured_binned_metrics.items():
                logging.info(f"{bin_name}:")
                for metric, value in metrics.items():
                    logging.info(f"  {metric}: {value:.4f}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("An error occurred in the main script:")
        logging.error(traceback.format_exc())
