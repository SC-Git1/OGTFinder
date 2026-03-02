"""Diagnostic plot utilities for ensemble models.

Ported from deep_learning/utils.py — matplotlib/seaborn only, no Keras dependency.
All plots saved as PNG.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def plot_predictions_enhanced(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str = "model",
    suffix: str = "",
) -> str:
    """Actual vs Predicted scatter with y=x line, fitted regression line, and stats."""
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    slope, intercept, r_value, _, _ = stats.linregress(y_actual, y_pred)
    r2 = r_value ** 2

    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_pred, alpha=0.5, s=10, label="Predictions")

    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect fit (y=x)")

    x_line = np.linspace(min_val, max_val, 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, "r-", lw=2, label=f"Fit: y={slope:.3f}x+{intercept:.2f}")

    plt.title(f"{prefix}{suffix}: Actual vs Predicted\nR²={r2:.4f}, Slope={slope:.3f}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{prefix}{suffix}_predictions_enhanced.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_residuals_vs_predicted(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str = "model",
    suffix: str = "",
) -> str:
    """Residuals vs predicted scatter to show heteroscedasticity and bias."""
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    residuals = y_actual - y_pred

    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color="red", linestyle="--", lw=2, label="Zero residual")
    plt.title(f"{prefix}{suffix}: Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{prefix}{suffix}_residuals_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_residual_distribution(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str = "model",
    suffix: str = "",
) -> str:
    """Residual distribution histogram with KDE and mean/median markers."""
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    residuals = y_actual - y_pred
    mean_res = np.mean(residuals)
    median_res = np.median(residuals)

    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True, alpha=0.7)
    plt.axvline(x=mean_res, color="red", linestyle="-", lw=2, label=f"Mean={mean_res:.2f}")
    plt.axvline(x=median_res, color="green", linestyle="--", lw=2, label=f"Median={median_res:.2f}")
    plt.title(f"{prefix}{suffix}: Residual Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{prefix}{suffix}_residual_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_residuals_qq(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str = "model",
    suffix: str = "",
) -> str:
    """QQ-plot of residuals to check normality and heavy tails."""
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    residuals = y_actual - y_pred

    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"{prefix}{suffix}: QQ-Plot of Residuals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{prefix}{suffix}_qq_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_error_by_decile(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str = "model",
    suffix: str = "",
) -> str:
    """Violin plot of errors per target decile to show model performance across range."""
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    errors = y_actual - y_pred

    try:
        deciles = pd.qcut(y_actual, 10, labels=False, duplicates="drop")
    except ValueError:
        deciles = pd.cut(y_actual, 10, labels=False)

    df = pd.DataFrame({"Decile": deciles, "Error": errors, "Actual": y_actual})

    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Decile", y="Error", data=df, inner="box")
    plt.axhline(y=0, color="red", linestyle="--", lw=1, alpha=0.7)
    plt.title(f"{prefix}{suffix}: Error Distribution by Target Decile")
    plt.xlabel("Target Decile (0=lowest, 9=highest)")
    plt.ylabel("Error (Actual - Predicted)")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(output_dir, f"{prefix}{suffix}_error_by_decile.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_calibration_curve(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str = "model",
    suffix: str = "",
    n_bins: int = 10,
) -> str:
    """Calibration curve: binned average predicted vs average actual with error bars."""
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        bins = pd.qcut(y_pred, n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(y_pred, n_bins, labels=False)

    df = pd.DataFrame({"Bin": bins, "Actual": y_actual, "Predicted": y_pred})

    bin_stats = df.groupby("Bin").agg(
        {"Actual": ["mean", "std", "count"], "Predicted": "mean"}
    ).reset_index()
    bin_stats.columns = ["Bin", "Actual_Mean", "Actual_Std", "Count", "Predicted_Mean"]
    bin_stats["SE"] = bin_stats["Actual_Std"] / np.sqrt(bin_stats["Count"])

    plt.figure(figsize=(8, 8))

    min_val = min(bin_stats["Predicted_Mean"].min(), bin_stats["Actual_Mean"].min())
    max_val = max(bin_stats["Predicted_Mean"].max(), bin_stats["Actual_Mean"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect calibration")

    plt.errorbar(
        bin_stats["Predicted_Mean"],
        bin_stats["Actual_Mean"],
        yerr=bin_stats["SE"],
        fmt="o-",
        capsize=4,
        capthick=2,
        markersize=8,
        label="Model calibration",
    )

    plt.title(f"{prefix}{suffix}: Calibration Curve ({n_bins} bins)")
    plt.xlabel("Mean Predicted")
    plt.ylabel("Mean Actual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{prefix}{suffix}_calibration_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_train_diagnostics(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str = "model",
) -> dict[str, str]:
    """Generate subset of diagnostic plots for training (OOF) data.

    Includes: predictions enhanced, error by decile, calibration curve.

    Returns:
        Dictionary mapping plot name to file path.
    """
    paths: dict[str, str] = {}
    paths["predictions_enhanced_train"] = plot_predictions_enhanced(
        y_actual, y_pred, output_dir, prefix, suffix="_train"
    )
    paths["error_by_decile_train"] = plot_error_by_decile(
        y_actual, y_pred, output_dir, prefix, suffix="_train"
    )
    paths["calibration_curve_train"] = plot_calibration_curve(
        y_actual, y_pred, output_dir, prefix, suffix="_train"
    )
    return paths


def plot_all_diagnostics(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str = "model",
) -> dict[str, str]:
    """Generate all diagnostic plots for test data.

    Returns:
        Dictionary mapping plot name to file path.
    """
    paths: dict[str, str] = {}
    paths["predictions_enhanced"] = plot_predictions_enhanced(
        y_actual, y_pred, output_dir, prefix
    )
    paths["residuals_vs_predicted"] = plot_residuals_vs_predicted(
        y_actual, y_pred, output_dir, prefix
    )
    paths["residual_distribution"] = plot_residual_distribution(
        y_actual, y_pred, output_dir, prefix
    )
    paths["qq_plot"] = plot_residuals_qq(y_actual, y_pred, output_dir, prefix)
    paths["error_by_decile"] = plot_error_by_decile(
        y_actual, y_pred, output_dir, prefix
    )
    paths["calibration_curve"] = plot_calibration_curve(
        y_actual, y_pred, output_dir, prefix
    )
    return paths
