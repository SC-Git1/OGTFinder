import pandas as pd
from Bio import SeqIO
import numpy as np
import re
import json
import math
import os
import pathlib
import sys
import joblib
from copy import deepcopy
import logging


dirname = os.path.dirname(__file__)

def ogt_predict(dfvalues, domain: str) -> float:

    # Load the ogt prediction model
    with open(os.path.join(dirname, 'data/mlp_rmse.pkl'), "rb") as file:
        loaded_model = joblib.load(file)

    predictions = loaded_model.predict(dfvalues)
    ogt_pred = predictions[0]

    if (ogt_pred > 122) or (ogt_pred < -20):
        logging.warning("The OGT prediction is {0:.2f}°C, which lays outside the range of reported growth temperatures in literature: -20°C to 122°C.".format(ogt_pred))

    return ogt_pred


def calc_mean_descriptors(sequences, domain, outdir, remove_upper_1 = True):

    # Load amino acid descriptors
    with open(os.path.join(dirname, 'data/AAdata.json'), "r") as f:
        df = json.load(f)

    df_dict = deepcopy(df)

    # header
    correct_cols = ["B1_mean","B2_mean","B3_mean","B4_mean","B5_mean","B6_mean","B7_mean","B8_mean","B9_mean","B10_mean","PP1_mean","PP2_mean","PP3_mean",
            "F1_mean","F2_mean","F3_mean","F4_mean","F5_mean","F6_mean","K1_mean","K2_mean","K3_mean","K4_mean","K5_mean","K6_mean","K7_mean",
            "K8_mean","K9_mean","K10_mean","MSWHIM1_mean","MSWHIM2_mean","MSWHIM3_mean","ST1_mean","ST2_mean","ST3_mean","ST4_mean","ST5_mean",
            "ST6_mean","ST7_mean","ST8_mean","T1_mean","T2_mean","T3_mean","T4_mean","T5_mean","VHSE1_mean","VHSE2_mean","VHSE3_mean","VHSE4_mean",
            "VHSE5_mean","VHSE6_mean","VHSE7_mean","VHSE8_mean","Z1_mean","Z2_mean","Z3_mean","Z4_mean","Z5_mean"]


    if remove_upper_1 == True:
        # Remove all proteins from percentile 99
        orig_lengths = [len(seq) for seq in sequences]
        q99 = np.quantile(orig_lengths, 0.99)
        sequences = [sequence for sequence in sequences if len(sequence) <= q99]

        logging.debug(f"Proteins with lengths greater than {math.floor(q99)} amino acids (above the 99th percentile) were removed.")


    # remove incorrect AAs and any consequential empty sequences
    sequences = [re.sub(r"[UXOB]", "", x) for x in sequences if re.sub(r"[UXOB]", "", x)]

    if len(sequences) == 0:
        logging.error("No sequences left after removing non-canonical amino acids. Exiting ...")
        sys.exit()

    list_scales = []
    for sequence in sequences:
        # Without np.float64 fixation, very small differences exist when comparing python < 3.12 vs higher. Fixed here for consistency (in paper: no fix)
        out = [np.array(list(df_dict[x].values()), dtype=np.float64) for x in sequence]
        list_scales.append([sum(x) / len(x) for x in zip(*out)])

    # take the average
    scales = list(map(np.mean,np.array(list_scales).T))

    dfvalues = pd.DataFrame([scales], columns = correct_cols)

    if domain == "Bacteria":
        superkingdom_dummy = 0
    elif domain == "Archaea":
        superkingdom_dummy = 1

    dfvalues.insert(0, "superkingdom_dummy", superkingdom_dummy)

    return dfvalues

