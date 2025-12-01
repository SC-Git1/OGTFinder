"""
This script was modified from the GitHub repo: https://github.com/EngqvistLab/Tome
"""

from sklearn import svm
from sklearn.metrics import r2_score
from scipy.stats import spearmanr,pearsonr
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import os
import pandas as pd
import joblib


def train_model():

    # define path for output model
    path = os.path.dirname(os.path.realpath(__file__)).replace('core','')
    predictor = os.path.join(path,'tome_model/OGT_svr.pkl')

    # setup normalization
    def get_standardizer(X):
        mean,std=list(),list()
        for i in range(X.shape[1]):
            mean.append(np.mean(X[:,i]))
            std.append(float(np.var(X[:,i]))**0.5)
        return mean,std

    # perform normalization
    def standardize(X):
        Xs=np.zeros_like(X)
        n_sample,n_features=X.shape[0],X.shape[1]
        for i in range(n_features):
            Xs[:,i]=(X[:,i]-np.mean(X[:,i]))/float(np.var(X[:,i]))**0.5
        return Xs

    # load training dataset
    trainfile = os.path.join(path,'tome_data/tome_train.csv')
    df = pd.read_csv(trainfile,index_col=0)
    X = df.values[:,:-1]
    Y = df.values[:,-1].ravel()
    features = df.columns[:-1]

    Xs = standardize(X)
    model = svm.SVR(kernel='rbf',C = 64.0, epsilon = 1.0) # use original optimal hyperparameters
    model.fit(Xs,Y)

    # save model
    print('A new model has beed successfully trained.')
    print('Saving the new model to replace the original one...')
    joblib.dump(model, predictor)

    # Save everything for normalization
    fea = open(predictor.replace('pkl','f'),'w')
    means, stds = get_standardizer(X)
    fea.write('#Feature_name\tmean\tstd\n')
    for i in range(len(means)):
        fea.write('{0}\t{1}\t{2}\n'.format(features[i], means[i], stds[i]))
    fea.close()
    print('Done!')


if __name__ == "__main__":
    # Train Tome on OGTFinder dataset
    train_model()