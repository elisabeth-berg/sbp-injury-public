import pandas as pd
import numpy as np
import multiprocessing
import pickle
import boto3

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score

from src import data_merge
from src import models


def put_pickle_model(fit_model, filename):
    '''
    Output
    -------
    Writes pickled model to s3 bucket location
    '''
    with open('model.pkl', 'wb') as f:
        pickle.dump(fit_model, f)
        s3 = boto3.client('s3')
        s3.put_object(Bucket='sbp-data-etc',
                      Body=open('model.pkl', 'rb'), Key=filename)


def run_bunch(X, y, n_splits, model_type, pool_size):
    """
    Split X and y into train/test splits n_splits times, fit and make predictions
    in parallel.

    Input
    ------
    X : Feature matrix (numpy array)
    y : Labels corresponding to X rows (numpy array)
    n_splits : The number of train/test splits to perform
    model_type : one of 'RFC' or 'GBC'

    Output
    ------
    model_output : Pandas dataframe. Each row contains the indices of the test
                   set, the corresponding predictions, and the AUC for this split.
    """
    pool = multiprocessing.Pool(pool_size)
    splits = make_splits(X, y, model_type, n_splits)

    model_output = pool.map(predict_parallel, splits)
    model_output = pd.DataFrame(model_output)
    model_output.columns = ['indices', 'y_hat', 'auc']
    return model_output


def make_splits(X, y, model_type, n_splits):
    """
    Input
    ------
    X : Feature matrix (numpy array)
    y : Labels corresponding to X rows (numpy array)
    n_splits : The number of train test splits to perform
    model_type : one of 'RFC' or 'GBC'

    Output
    ------
    splits : List of tuples (X_train, X_test, y_train, y_test)
    """
    splits = []
    for n in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        splits.append((X_train, X_test, y_train, y_test, model_type))
    return splits


def predict_parallel(splits):
    """
    Input
    ------
    splits : A tuple of train/test splits & model_type, the output of make_splits

    Output
    ------
    indices : The test set indices of the original dataframe
    y_hat : Predicted values
    auc : AUC for this particular split

    ------
    This function can be implemented in parallel on an EC2 instance with at least n_splits cores.
        >> splits = make_splits(df, n_splits)
        >> pool = multiprocessing.Pool(n_splits)
        >> results = pool.map(predict_parallel, splits)

    results will be a list of tuples (indices, y_hat, auc)
    """
    (X_train, X_test, y_train, y_test, model_type) = splits
    indices = X_test.index
    model = models.InjuryModel(model_type)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)
    return indices, y_hat, auc



def bootstrap_train(model, X, y, bootstraps=1000, **kwargs):
    """
    Train a model on multiple bootstrap samples of some data and
    return the fit model.

    Input
    ----------
    model: A sklearn class whose instances have a `fit` method
    X: A two dimensional numpy array of shape (n_observations, n_features).
    y: A one dimensional numpy array of shape (n_observations).
    bootstraps: An integer, the number of boostrapped models to train.

    Output
    -------
    bootstrap_models: A list of fit models.
    """
    bootstrap_models = []
    for i in range(bootstraps):
        boot_idxs = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_boot = X[boot_idxs, :]
        y_boot = y[boot_idxs]
        M = model(**kwargs)
        M.fit(X_boot, y_boot)
        bootstrap_models.append(M)
    return bootstrap_models


def make_clusters(clustering_df, n_features, best_k=None):
    """
    clustering_df : The output from injury_full_data, with columns chosen & dummified
    n_features : Number of top features to extract from each cluster centroid
    """
    features = clustering_df.columns
    scaler = StandardScaler()
    clustering_df = scaler.fit_transform(clustering_df)
    maxk = len(features)//2
    silhouette = np.zeros(maxk)
    if best_k == None:
        for k in range(1, maxk):
            km = KMeans(k)
            y = km.fit_predict(clustering_df)
            if k > 1:
                silhouette[k] = silhouette_score(clustering_df, y)
        best_k = np.argmax(silhouette) + 2

    kmeans = KMeans(n_clusters=best_k).fit(clustering_df)
    centroids = kmeans.cluster_centers_
    centroids2 = scaler.inverse_transform(centroids)

    for i, (c1, c2) in enumerate(zip(centroids, centroids2)):
        ind = c1.argsort()[::-1][:n_features]
        print('Cluster {}'.format(i))
        for i in ind:
            print('{} || {}'.format(features[i], c2[i]))
        print('----------------------')
    return centroids, silhouette
