"""
!!!RELATIVE HARD-CODED PATHS!!!

Prepares datasets for OGT ML
 - taxonomy metadata
 - test split (with genus split)
 - alternative test split (with genus AND uniform temp)
 - checks for missing values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


# "out" DIR SHOULD BE IN THIS WORKING DIRECTORY (HARD-CODED IN FUNCTIONS BELOW)
# RELATIVE PATH TO DATASET
INPUT_DATASET_DIR = "../data/raw/"
INPUT_DATASET_NAME = "Input_opt_uniqtaxids.tsv"
INPUT_DATASET = os.path.join(INPUT_DATASET_DIR, INPUT_DATASET_NAME)
# RELATIVE PATHS TO CCT DATASETS
INPUT_DATASET_CCT_DIR = ""

# TARGET VARIABLE
from config_preprocess import (TARGET_VAR, LEVEL, RANDOM_STATE)

# RELATIVE PATHS TO OUTPUT DIRECTORY FOR DATAFRAMES
OUTPUT_DIR = "../data/training/"
TRAINING_PHYLO_SPLIT = os.path.join(OUTPUT_DIR, "train_" + LEVEL + ".csv")
TEST_PHYLO_SPLIT = os.path.join(OUTPUT_DIR, "test_" + LEVEL + ".csv")
TRAINING_UNIFORM_SPLIT = os.path.join(OUTPUT_DIR, "train_uniform.csv")
TEST_UNIFORM_SPLIT = os.path.join(OUTPUT_DIR, "test_uniform.csv")

def bin_and_split(df, target_var, n_bins=10, test_size=0.10, random_state = RANDOM_STATE):
    """
    VALUE ERROR IF NOT ENOUGH OBSERVATIONS FOR BINS AND TEST SIZE
    """
    df["bin"] = pd.qcut(df[target_var], q=n_bins, duplicates="drop", labels=False)
    n_obs = len(df)
    test_obs_per_bin = int(np.floor(test_size * n_obs / n_bins))
    if test_obs_per_bin == 0:
        raise ValueError(
            "Not enough observations for the number of bins and test size."
        )
    test_indices = np.array([], dtype=int)
    for b in range(n_bins):
        bin_indices = df[df["bin"] == b].index
        np.random.seed(random_state)
        bin_test_indices = np.random.choice(
            bin_indices, size=test_obs_per_bin, replace=False
        )
        test_indices = np.concatenate((test_indices, bin_test_indices))
    if len(test_indices) > test_size * n_obs:
        test_indices = test_indices[: int(test_size * n_obs)]
    train_df = df.drop(test_indices)
    test_df = df.loc[test_indices]
    train_df = train_df.drop(columns=["bin"])
    test_df = test_df.drop(columns=["bin"])
    return train_df, test_df


def phylo_target_split(df, target_var, level):
    unique_genus_ids = df[level + "_id"].unique()
    train_genus_ids, test_genus_ids = train_test_split(
        unique_genus_ids, test_size=0.102, random_state=RANDOM_STATE
    )
    train_df1 = df[df[level + "_id"].isin(train_genus_ids)]
    test_df1 = df[df[level + "_id"].isin(test_genus_ids)]
    # split based on bins
    train_df2, test_df2 = bin_and_split(df, target_var, n_bins=8, test_size=0.102)
    return train_df1, test_df1, train_df2, test_df2


def check_no_essential_missing_data(df, essential_cols):
    missing_rows = df.isnull().any(axis=0).loc[essential_cols]
    if sum(missing_rows) == 0:
       return df
    else:
        for col in essential_cols:
            if sum(df[col].isna()) > 0:
                print("Column:" + str(col))
        raise Exception('Some essential columns have NaN values')


def generate_overview_figures(df, train_df1, test_df1, train_df2, test_df2, target_var, level):
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target_var], kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution of {TARGET_VAR}")
    plt.xlabel(target_var)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"{TARGET_VAR}_{LEVEL}_distribution.png"))
    plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    datasets = [train_df1, test_df1, train_df2, test_df2]
    titles = ["Train " + level, "Test " + level, "Train uniform", "Test uniform"]
    for ax, dataset, title in zip(axs.flat, datasets, titles):
        sns.histplot(dataset[target_var], kde=True, bins=30, ax=ax, color="skyblue")
        ax.set(title=title, xlabel=target_var, ylabel="Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{TARGET_VAR}_{LEVEL}_splits_distribution.png"))
    plt.close()


def generate_combined_barplot(df, train_df1, test_df1, train_df2, test_df2, level):
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    datasets = [train_df1, test_df1, train_df2, test_df2]
    titles = ["Train " + level, "Test " + level, "Train uniform", "Test uniform"]
    counts = [len(d) for d in datasets]
    percentages = [len(d) / len(df) * 100 for d in datasets]
    plt.figure(figsize=(10, 6))
    colors = ["skyblue", "lightgreen", "salmon", "lightyellow"]
    bars = sns.barplot(x=titles, y=counts, color="lightblue")
    for bar, color in zip(bars.patches, colors):
        bar.set_color(color)
    for bar, count, percentage in zip(bars.patches, counts, percentages):
        yval = bar.get_height()
        plt.annotate(
            f"{count} ({percentage:.2f}%)",
            (bar.get_x() + bar.get_width() / 2, yval),
            ha="center",
            va="bottom",
        )
    plt.title("# Observations and Percentages")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, f"splits_counts_{LEVEL}.png"))
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(INPUT_DATASET, header = 0, sep = "\t")
    print(len(df))
    df = df[~df["species"].isin(["Levilactobacillus zymae", "Levilactobacillus acidifarinae"])]
    print(len(df))
    essential_cols = [col for col in df.columns if "_mean" in col]
    df_cleaned = check_no_essential_missing_data(df, essential_cols)


    train_phylosplit, test_phylosplit, train_uniform, test_uniform = phylo_target_split(
        df_cleaned, TARGET_VAR, LEVEL
    )
    generate_overview_figures(
        df_cleaned, train_phylosplit, test_phylosplit, train_uniform, test_uniform, TARGET_VAR, LEVEL
    )
    generate_combined_barplot(
        df_cleaned, train_phylosplit, test_phylosplit, train_uniform, test_uniform, LEVEL
    )

    train_phylosplit.to_csv(TRAINING_PHYLO_SPLIT, index=False)
    test_phylosplit.to_csv(TEST_PHYLO_SPLIT, index=False)
    train_uniform.to_csv(TRAINING_UNIFORM_SPLIT, index=False)
    test_uniform.to_csv(TEST_UNIFORM_SPLIT, index=False)

    print("Data preparation complete.")
