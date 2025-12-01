"""
MACHINE LEARNING UTILS
"""
import contextlib
import io
import logging
import os
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# for extreme_rmse function
import scipy.stats as stats


def extreme_rmse_sampled(y_true, y_pred, n_features,sample_weights):
    weights = [sample_weights[el] for el in y_true]
    mse = np.mean(weights * ((y_true - y_pred) ** 2))
    return np.sqrt(mse)


def extreme_rmse(y_true, y_pred, n_features, alpha=0.05, delta = 0.01):
    # get density distribution of evaluation data (during model training)
    density = stats.gaussian_kde(y_true)
    evaluated = density.evaluate(y_true)
    # inverse, in case density = 0 introduced a delta value
    inverse_PDF = 1 / (evaluated + delta)
    # nomralize to 0-1 range
    weights = (inverse_PDF-inverse_PDF.min())/(inverse_PDF.max()-inverse_PDF.min())

    # calculate the RMSE
    mse = np.mean(weights * ((y_true - y_pred) ** 2))
    return np.sqrt(mse)


def adjusted_rmse(y_true, y_pred, n_features, alpha=0.05):
    n = len(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Calculate penalty factor | larger alpha = larger penalty for more features
    penalty = 1 + alpha * (n_features / n)
    adj_rmse = rmse * penalty
    return adj_rmse


def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    if n <= n_features + 1 or r2 == 1:
        return r2  # Return regular R2 in these cases
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return max(min(adj_r2, 1), -1)


def calculate_binned_metrics(y_true, y_pred, X):

    def metrics_for_bin(y_true_bin, y_pred_bin, n_features):
        rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
        mae = mean_absolute_error(y_true_bin, y_pred_bin)
        r2 = r2_score(y_true_bin, y_pred_bin)
        adj_r2 = adjusted_r2_score(y_true_bin, y_pred_bin, n_features)
        return rmse, mae, r2, adj_r2

    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    n = len(y_true)
    n_features = X.shape[1]
    bins = {
        "top_5%": slice(int(0.95 * n), None),
        "top_10%": slice(int(0.90 * n), None),
        "top_20%": slice(int(0.80 * n), None),
        "bottom_5%": slice(None, int(0.05 * n)),
        "bottom_10%": slice(None, int(0.10 * n)),
        "bottom_20%": slice(None, int(0.20 * n)),
    }
    results = {}
    for bin_name, bin_slice in bins.items():
        y_true_bin = y_true_sorted[bin_slice]
        y_pred_bin = y_pred_sorted[bin_slice]
        results[bin_name] = metrics_for_bin(y_true_bin, y_pred_bin, n_features)
    return results


def save_performance_to_csv(
    file_path,
    model_name,
    experiment_name,
    dataset_type,
    overall_metrics,
    binned_metrics=None,
):
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
        if binned_metrics:
            fieldnames = [
                "Model",
                "Experiment",
                "Dataset",
                "Metric",
                "Overall",
                "Top 5%",
                "Top 10%",
                "Top 20%",
                "Bottom 5%",
                "Bottom 10%",
                "Bottom 20%",
            ]
        else:
            fieldnames = [
                "Model",
                "Experiment",
                "Dataset",
                "Metric",
                "Overall"
            ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write headers if file doesn't exist
        if not file_exists:
            writer.writeheader()
        # Write overall metrics
        for metric, value in overall_metrics.items():
            if binned_metrics:
                row = {
                    "Model": model_name,
                    "Experiment": experiment_name,
                    "Dataset": dataset_type,
                    "Metric": metric,
                    "Overall": value,
                    "Top 5%": binned_metrics["top_5%"][metric],
                    "Top 10%": binned_metrics["top_10%"][metric],
                    "Top 20%": binned_metrics["top_20%"][metric],
                    "Bottom 5%": binned_metrics["bottom_5%"][metric],
                    "Bottom 10%": binned_metrics["bottom_10%"][metric],
                    "Bottom 20%": binned_metrics["bottom_20%"][metric],
                }
            else:
                row = {
                    "Model": model_name,
                    "Experiment": experiment_name,
                    "Dataset": dataset_type,
                    "Metric": metric,
                    "Overall": value,
                }

            writer.writerow(row)
