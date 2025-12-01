import json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import joblib


# weighted RMSE function
def extreme_rmse_sampled(y_true, y_pred, n_features,sample_weights):
    weights = [sample_weights[el] for el in y_true]
    mse = np.mean(weights * ((y_true - y_pred) ** 2))
    return np.sqrt(mse)

## Helper Function: load everything for feature normalization (from Tome package)
def load_means_stds(predictor):
    means=dict()
    stds=dict()
    features=list()
    for line in open(predictor.replace('pkl','f'),'r'):
        if line.startswith('#'):continue
        cont=line.split()
        means[cont[0]]=float(cont[1])
        stds[cont[0]]=float(cont[2])
        features.append(cont[0])
    return means,stds,features


## Helper Function: feature values to dictionary (from Tome package)
def get_dimer_frequency(X, keys):
    dimers_fq = dict()
    for i in range(len(keys)):
        dimers_fq[keys[i]] = X[i]
    return dimers_fq


## load model (from Tome package)
def load_model():
    path = os.path.dirname(os.path.realpath(__file__)).replace('core','')
    predictor = os.path.join(path,'tome_model/OGT_svr.pkl')
    model=joblib.load(predictor)
    means,stds,features = load_means_stds(predictor)

    return model,means,stds,features


if __name__ == "__main__":

    # load Tome model
    model, means, stds, features = load_model()

    # read in the weigths of test data
    weights_file = "/data/leuven/345/vsc34502/PhD/ogt_final_runs/data/training/weights_test_genus.json"

    with open(weights_file, "r") as infile:
        sample_weights = json.load(infile)
        sample_weights = {float(i):float(j) for i,j in sample_weights.items()}

    # read in test data
    dftest = pd.read_csv("tome_data/tome_test.csv", header = 0, index_col = 0)
    y = dftest["ogt"]

    # initialize list of predicted ogt values
    y_preds = []

    ## For each test datapoint
    for i in range(len(dftest)):
        # get features of observation "i"
        X = dftest.values[i,:-1]
        # list of feature names
        Xcols = dftest.columns[:-1]
        # features as dictionary
        dimers_fq = get_dimer_frequency(X, Xcols)
        
        # normalize
        Xs = list()
        for fea in features:
            Xs.append((dimers_fq[fea]-means[fea])/stds[fea])
        Xs = np.array(Xs).reshape([1,len(Xs)])

        # append predicted ogt
        y_preds.append(model.predict(Xs)[0])

    # calculate overall test metrics
    r2_test = r2_score(y, y_preds)
    rmse_test = np.sqrt(mean_squared_error(y, y_preds))
    weighted_rmse_test = -extreme_rmse_sampled(y, y_preds, Xs.shape[1], sample_weights)

    print(f"r2_test: {r2_test}")
    print(f"rmse_test: {rmse_test}")
    print(f"weighted_rmse_test: {weighted_rmse_test}")

    ## calculate percentile-specific test metrics

    # bottom 5, 10 and 20%
    percentiles_bottom = [5, 10, 20]

    # top 5, 10 and 20%
    percentiles_top = [80, 90, 95]

    # Convert list to NumPy array
    y_preds = np.array(y_preds)  

    for p in percentiles_bottom:
        # temperature threshold
        thresh = np.percentile(y, p)
        # true/false
        keep = y <= thresh
        # get real ogt values for selected
        real = y[keep]
        # get predicted ogt values for selected
        pred = y_preds[keep]
        # calculate and print metrics
        r2 = r2_score(real, pred)
        rmse = np.sqrt(mean_squared_error(real, pred))
        print(f"<{p}th pct — R²: {r2:.3f}, RMSE: {rmse:.2f}")
        print(f"Temperature threshold [°C]: {thresh}")

    for p in percentiles_top:
         # temperature threshold
        thresh = np.percentile(y, p)
        # true/false
        keep = y >= thresh
        # get real ogt values for selected
        real = y[keep]
        # get predicted ogt values for selected
        pred = y_preds[keep]
        # calculate and print metrics
        r2 = r2_score(real, pred)
        rmse = np.sqrt(mean_squared_error(real, pred))
        print(f">{p}th pct — R²: {r2:.3f}, RMSE: {rmse:.2f}")
        print(f"threshold: {thresh}")

    # Write to output file
    dfout = pd.DataFrame({"ID":dftest.index,"OGT":y, "Tome_Predicted":y_preds})
    dfout.to_csv("/path/to/bench_methods/tome_output.tsv", sep = "\t", index = False)
