#!/usr/bin/env python3
"""
Model Performance Analysis Script

Analyzes model_performance.csv and generates comparison bar plots for:
- Overall R² and RMSE on test set
- RMSE for top/bottom percentile sections (5%, 10%, 20%)

Output: Saves plots to ./analysis/ directory
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
INPUT_FILE = "model_performance.csv"
OUTPUT_DIR = "analysis"
FIGSIZE_SINGLE = (12, 6)
FIGSIZE_MULTI = (14, 10)
DPI = 150

# Color palette for consistent styling
COLORS = {
    'r2': '#2ecc71',        # Green for R²
    'rmse': '#e74c3c',      # Red for RMSE
    'bottom': '#3498db',    # Blue for bottom percentiles
    'top': '#f39c12',       # Orange for top percentiles
    'overall': '#9b59b6',   # Purple for overall
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate the performance CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Models found: {df['model_name'].nunique()}")
    print(f"Dataset types: {df['dataset_type'].unique()}")
    return df


def filter_by_dataset_type(df: pd.DataFrame, dataset_type: str = 'test') -> pd.DataFrame:
    """Filter for specific dataset_type results."""
    filtered_df = df[df['dataset_type'] == dataset_type].copy()
    print(f"Filtered to {len(filtered_df)} '{dataset_type}' entries")
    return filtered_df


def sort_by_metric(df: pd.DataFrame, metric: str, ascending: bool = True) -> pd.DataFrame:
    """Sort dataframe by a metric."""
    return df.sort_values(metric, ascending=ascending)


def plot_overall_metrics(df: pd.DataFrame, output_dir: str, dataset_label: str = 'Test Set'):
    """
    Create bar plots for overall R² and RMSE on test set.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Sort by R² (descending - higher is better)
    df_r2 = sort_by_metric(df, 'r2', ascending=False)

    # Plot 1: R² comparison
    ax1 = axes[0]
    bars1 = ax1.barh(df_r2['model_name'], df_r2['r2'], color=COLORS['r2'], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('R² Score', fontsize=12)
    ax1.set_title(f'{dataset_label} R² by Model (Higher is Better)', fontsize=14, fontweight='bold')
    ax1.axvline(x=df_r2['r2'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df_r2['r2'].mean():.3f}")
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, max(df_r2['r2'].max() * 1.1, 1.0))

    # Add value labels
    for bar, val in zip(bars1, df_r2['r2']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=9)

    # Sort by RMSE (ascending - lower is better)
    df_rmse = sort_by_metric(df, 'rmse', ascending=True)

    # Plot 2: RMSE comparison
    ax2 = axes[1]
    bars2 = ax2.barh(df_rmse['model_name'], df_rmse['rmse'], color=COLORS['rmse'], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('RMSE', fontsize=12)
    ax2.set_title(f'{dataset_label} RMSE by Model (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.axvline(x=df_rmse['rmse'].mean(), color='blue', linestyle='--', linewidth=2, label=f"Mean: {df_rmse['rmse'].mean():.3f}")
    ax2.legend(loc='lower right')

    # Add value labels
    for bar, val in zip(bars2, df_rmse['rmse']):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                va='center', fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'overall_r2_rmse_comparison.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_percentile_rmse(df: pd.DataFrame, output_dir: str, dataset_label: str = 'Test Set'):
    """
    Create bar plots comparing RMSE for top/bottom percentiles.
    """
    # Sort by overall RMSE for consistent ordering
    df = sort_by_metric(df, 'rmse', ascending=True)
    models = df['model_name'].tolist()

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    percentiles = [5, 10, 20]

    for idx, pct in enumerate(percentiles):
        # Bottom percentile
        ax_bottom = axes[idx, 0]
        bottom_col = f'rmse_bottom_{pct}'
        bottom_vals = df[bottom_col].values

        colors_bottom = [COLORS['bottom'] if v > 0 else '#c0392b' for v in bottom_vals]
        bars_b = ax_bottom.barh(models, bottom_vals, color=colors_bottom, edgecolor='black', linewidth=0.5)
        ax_bottom.set_xlabel('RMSE', fontsize=11)
        ax_bottom.set_title(f'RMSE - Bottom {pct}% (Coldest Samples)', fontsize=12, fontweight='bold')
        ax_bottom.axvline(x=df['rmse'].mean(), color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Overall Mean')

        # Add value labels
        for bar, val in zip(bars_b, bottom_vals):
            x_pos = val + 0.1 if val >= 0 else val - 0.5
            ax_bottom.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                          va='center', fontsize=8)

        # Top percentile
        ax_top = axes[idx, 1]
        top_col = f'rmse_top_{pct}'
        top_vals = df[top_col].values

        colors_top = [COLORS['top'] if v > 0 else '#c0392b' for v in top_vals]
        bars_t = ax_top.barh(models, top_vals, color=colors_top, edgecolor='black', linewidth=0.5)
        ax_top.set_xlabel('RMSE', fontsize=11)
        ax_top.set_title(f'RMSE - Top {pct}% (Hottest Samples)', fontsize=12, fontweight='bold')
        ax_top.axvline(x=df['rmse'].mean(), color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Overall Mean')

        # Add value labels
        for bar, val in zip(bars_t, top_vals):
            x_pos = val + 0.1 if val >= 0 else val - 0.5
            ax_top.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                          va='center', fontsize=8)

    plt.suptitle(f'RMSE Performance on Extreme Temperature Percentiles ({dataset_label})',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'percentile_rmse_comparison.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_combined_percentile_summary(df: pd.DataFrame, output_dir: str, dataset_label: str = 'Test Set'):
    """
    Create a grouped bar chart showing overall RMSE vs top/bottom 10% RMSE.
    """
    df = sort_by_metric(df, 'rmse', ascending=True)
    models = df['model_name'].tolist()

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(models))
    width = 0.25

    # Get values
    overall = df['rmse'].values
    bottom_10 = df['rmse_bottom_10'].values
    top_10 = df['rmse_top_10'].values

    # Create grouped bars
    bars1 = ax.bar(x - width, overall, width, label='Overall RMSE', color=COLORS['overall'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, bottom_10, width, label='Bottom 10% RMSE', color=COLORS['bottom'], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, top_10, width, label='Top 10% RMSE', color=COLORS['top'], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'RMSE Comparison: Overall vs Extreme Percentiles ({dataset_label})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(max(overall), max(bottom_10), max(top_10)) * 1.15)

    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'grouped_rmse_percentiles.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_r2_comparison_detailed(df: pd.DataFrame, output_dir: str, dataset_label: str = 'Test Set'):
    """
    Create a detailed R² comparison with adjusted R².
    """
    df = sort_by_metric(df, 'r2', ascending=False)
    models = df['model_name'].tolist()

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, df['r2'], width, label='R²', color=COLORS['r2'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, df['adj_r2'], width, label='Adjusted R²', color='#27ae60', edgecolor='black', linewidth=0.5, alpha=0.7)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'R² vs Adjusted R² by Model ({dataset_label})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='lower right', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'r2_adj_r2_comparison.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_model_ranking_table(df: pd.DataFrame, output_dir: str, dataset_label: str = 'Test Set'):
    """
    Create a summary ranking visualization.
    """
    df = df.copy()

    # Calculate ranks (1 = best)
    df['rank_r2'] = df['r2'].rank(ascending=False).astype(int)
    df['rank_rmse'] = df['rmse'].rank(ascending=True).astype(int)
    df['rank_rmse_bottom_10'] = df['rmse_bottom_10'].rank(ascending=True).astype(int)
    df['rank_rmse_top_10'] = df['rmse_top_10'].rank(ascending=True).astype(int)

    # Average rank
    df['avg_rank'] = (df['rank_r2'] + df['rank_rmse'] + df['rank_rmse_bottom_10'] + df['rank_rmse_top_10']) / 4
    df = df.sort_values('avg_rank')

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    # Create table data
    table_data = []
    headers = ['Model', 'R² Rank', 'RMSE Rank', 'Bottom 10%\nRank', 'Top 10%\nRank', 'Avg Rank']

    for _, row in df.iterrows():
        table_data.append([
            row['model_name'],
            int(row['rank_r2']),
            int(row['rank_rmse']),
            int(row['rank_rmse_bottom_10']),
            int(row['rank_rmse_top_10']),
            f"{row['avg_rank']:.1f}"
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#3498db'] * len(headers)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Color code by average rank (green = best, red = worst)
    n_models = len(df)
    for row_idx in range(1, len(table_data) + 1):
        # Color the avg rank column based on rank
        rank_val = float(table_data[row_idx - 1][5])
        intensity = (rank_val - 1) / (n_models - 1) if n_models > 1 else 0
        color = plt.cm.RdYlGn(1 - intensity)  # Green for low rank, red for high
        table[(row_idx, 5)].set_facecolor(color)

    plt.title(f'Model Performance Ranking Summary ({dataset_label})\n(1 = Best)', fontsize=14, fontweight='bold', pad=20)

    filepath = os.path.join(output_dir, 'model_ranking_table.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

    # Also save ranking as CSV
    ranking_df = df[['model_name', 'r2', 'rmse', 'rmse_bottom_10', 'rmse_top_10',
                     'rank_r2', 'rank_rmse', 'rank_rmse_bottom_10', 'rank_rmse_top_10', 'avg_rank']]
    ranking_df = ranking_df.sort_values('avg_rank')
    csv_path = os.path.join(output_dir, 'model_rankings.csv')
    ranking_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def plot_rmse_percentile_lines(df: pd.DataFrame, output_dir: str, dataset_label: str = 'Test Set'):
    """
    Create a dotted line plot showing RMSE across percentile categories.
    X-axis: Bottom 5% -> Bottom 10% -> Bottom 20% -> Overall -> Top 20% -> Top 10% -> Top 5%
    Y-axis: RMSE
    Each model is a separate line with dots.
    """
    # Define the x-axis categories in order
    categories = ['Bottom 5%', 'Bottom 10%', 'Bottom 20%', 'Overall', 'Top 20%', 'Top 10%', 'Top 5%']
    columns = ['rmse_bottom_5', 'rmse_bottom_10', 'rmse_bottom_20', 'rmse', 'rmse_top_20', 'rmse_top_10', 'rmse_top_5']

    # Sort by overall RMSE for legend ordering
    df = sort_by_metric(df, 'rmse', ascending=True)

    # Create a distinct color palette for all models
    n_models = len(df)
    cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i / n_models) for i in range(n_models)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 9))

    x = np.arange(len(categories))

    for idx, (_, row) in enumerate(df.iterrows()):
        model_name = row['model_name']
        values = [row[col] for col in columns]

        # Plot dotted line with markers
        ax.plot(x, values,
                marker='o',
                markersize=8,
                linestyle='--',
                linewidth=2,
                color=colors[idx],
                label=model_name,
                alpha=0.8)

    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_xlabel('Percentile Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title(f'RMSE Across Temperature Percentiles ({dataset_label})\nBottom = Coldest, Top = Hottest',
                 fontsize=14, fontweight='bold')

    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)

    # Add vertical line at "Overall"
    ax.axvline(x=3, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Add legend outside the plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
              title='Models (sorted by overall RMSE)', title_fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'rmse_percentile_lines.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

    # Also create a version with fewer models (top 10 best) for clarity
    if n_models > 10:
        fig, ax = plt.subplots(figsize=(14, 9))

        df_top10 = df.head(10)

        for idx, (_, row) in enumerate(df_top10.iterrows()):
            model_name = row['model_name']
            values = [row[col] for col in columns]

            ax.plot(x, values,
                    marker='o',
                    markersize=10,
                    linestyle='--',
                    linewidth=2.5,
                    color=colors[idx],
                    label=model_name,
                    alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_xlabel('Percentile Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_title(f'RMSE Across Temperature Percentiles - Top 10 Models ({dataset_label})\nBottom = Coldest, Top = Hottest',
                     fontsize=14, fontweight='bold')

        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.grid(True, linestyle=':', alpha=0.3)
        ax.set_axisbelow(True)
        ax.axvline(x=3, color='gray', linestyle='-', linewidth=1, alpha=0.5)

        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
                  title='Top 10 Models', title_fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(output_dir, 'rmse_percentile_lines_top10.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")


def generate_summary_stats(df: pd.DataFrame, output_dir: str, dataset_label: str = 'Test Set'):
    """
    Generate and save summary statistics.
    """
    summary = {
        'Metric': ['R²', 'Adjusted R²', 'RMSE', 'MAE', 'Weighted RMSE',
                   'RMSE Bottom 10%', 'RMSE Top 10%'],
        'Best Model': [
            df.loc[df['r2'].idxmax(), 'model_name'],
            df.loc[df['adj_r2'].idxmax(), 'model_name'],
            df.loc[df['rmse'].idxmin(), 'model_name'],
            df.loc[df['mae'].idxmin(), 'model_name'],
            df.loc[df['weighted_rmse'].idxmin(), 'model_name'],
            df.loc[df['rmse_bottom_10'].idxmin(), 'model_name'],
            df.loc[df['rmse_top_10'].idxmin(), 'model_name'],
        ],
        'Best Value': [
            f"{df['r2'].max():.4f}",
            f"{df['adj_r2'].max():.4f}",
            f"{df['rmse'].min():.4f}",
            f"{df['mae'].min():.4f}",
            f"{df['weighted_rmse'].min():.4f}",
            f"{df['rmse_bottom_10'].min():.4f}",
            f"{df['rmse_top_10'].min():.4f}",
        ],
        'Mean': [
            f"{df['r2'].mean():.4f}",
            f"{df['adj_r2'].mean():.4f}",
            f"{df['rmse'].mean():.4f}",
            f"{df['mae'].mean():.4f}",
            f"{df['weighted_rmse'].mean():.4f}",
            f"{df['rmse_bottom_10'].mean():.4f}",
            f"{df['rmse_top_10'].mean():.4f}",
        ],
        'Std': [
            f"{df['r2'].std():.4f}",
            f"{df['adj_r2'].std():.4f}",
            f"{df['rmse'].std():.4f}",
            f"{df['mae'].std():.4f}",
            f"{df['weighted_rmse'].std():.4f}",
            f"{df['rmse_bottom_10'].std():.4f}",
            f"{df['rmse_top_10'].std():.4f}",
        ]
    }

    summary_df = pd.DataFrame(summary)
    csv_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print to console
    print("\n" + "="*60)
    print(f"SUMMARY STATISTICS ({dataset_label})")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60 + "\n")


def run_analysis(df: pd.DataFrame, dataset_type: str, dataset_label: str, output_path: Path):
    """Run the full analysis pipeline for a given dataset type."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {dataset_label} (dataset_type='{dataset_type}')")
    print(f"{'='*60}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}\n")

    # Filter data
    filtered_df = filter_by_dataset_type(df, dataset_type)

    if filtered_df.empty:
        print(f"WARNING: No '{dataset_type}' data found — skipping.")
        return

    print(f"Models in '{dataset_type}' set: {filtered_df['model_name'].tolist()}\n")

    # Generate all plots
    print("Generating plots...")
    print("-" * 40)

    plot_overall_metrics(filtered_df, output_path, dataset_label)
    plot_percentile_rmse(filtered_df, output_path, dataset_label)
    plot_combined_percentile_summary(filtered_df, output_path, dataset_label)
    plot_r2_comparison_detailed(filtered_df, output_path, dataset_label)
    plot_rmse_percentile_lines(filtered_df, output_path, dataset_label)
    plot_model_ranking_table(filtered_df, output_path, dataset_label)
    generate_summary_stats(filtered_df, output_path, dataset_label)

    print("-" * 40)
    n_plots = len(list(output_path.glob('*.png')))
    print(f"Analysis complete! {n_plots} plots generated.")
    print(f"Results saved to: {output_path}")


def main():
    """Main execution function."""
    print("="*60)
    print("Model Performance Analysis")
    print("="*60)

    # Setup paths
    script_dir = Path(__file__).parent
    input_path = script_dir / INPUT_FILE
    base_output = script_dir / OUTPUT_DIR

    # Load data once
    df = load_data(input_path)

    # --- Test set analysis (main output directory) ---
    run_analysis(df, dataset_type='test', dataset_label='Test Set',
                 output_path=base_output)

    # --- CV analysis (cv_results subdirectory) ---
    run_analysis(df, dataset_type='cv', dataset_label='CV',
                 output_path=base_output / 'cv_results')

    print(f"\n{'='*60}")
    print("All analyses complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
