import json
import pandas as pd
import os

# for extreme_rmse function
import scipy.stats as stats

from config_preprocess import LEVEL, TARGET_VAR


OUTPUT_DIR = "../data/training/"
WEIGHTS_FILE = os.path.join(OUTPUT_DIR, "weights_train_" + LEVEL + ".json")
TRAINING_FILE = os.path.join(OUTPUT_DIR, "train_" + LEVEL + ".csv")


if __name__ == "__main__":

    df = pd.read_csv(TRAINING_FILE)

    y_true = list(df[TARGET_VAR])
    density = stats.gaussian_kde(y_true)
    evaluated = density.evaluate(y_true)
    # inverse, in case density = 0 introduced a delta value
    delta = 0.01
    inverse_PDF = 1 / (evaluated + delta)
    # nomralize to 0-1 range
    inverse_PDF = (inverse_PDF-inverse_PDF.min())/(inverse_PDF.max()-inverse_PDF.min())

    weights = dict()

    for value, weight in zip(y_true, inverse_PDF):
        if value not in list(weights.keys()):
            weights[value] = weight

    # write to file
    with open(WEIGHTS_FILE, "w") as outfile:
        json.dump(weights, outfile)
