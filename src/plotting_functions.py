import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt

def gridsearch_means(gridsearch_df, n_stages, tree_depths, error_column):
    """
    Input
    ------
    gridsearch_df : Pandas df containing the results of gridsearching the tree_depth using 10 folds at each depth.
                    Columns : 'tree_depth', 'k', 'log_losses', 'aucs', 'y_pred', 'y_hat'
                    Most notably, 'log_losses' and 'aucs' are lists of the error at each boosting stage
                    (for a given tree_depth and current fold k).
                    >> Use the data in 'data/cross_val_results.csv'

    n_stages :      Total number of boosting stages to perform.
    tree_depths :   List of tree depths that were gridsearched (the unique values in gridsearch_df['tree_depth']).
    error_column :  One of 'log_losses' or 'aucs', the column on which to compute means.

    Output
    ------
    mean_vals :     Numpy array of shape (n_stages, n_trees).
                    Each column corresponds to one tree depth, each row corresponds to a single boosting stage.
                    The values are the mean error across the 10 folds at the given tree depth & boosting stage.
    """
    n_trees = len(tree_depths)
    mean_vals = np.zeros((n_stages, n_trees))
    for i, td in enumerate(tree_depths):
        acc = np.zeros(n_stages)
        errors = gridsearch_df[gridsearch_df['tree_depth'] == td][error_column]
        for error_list in errors:
            e_list = error_list.split(',')
            e_list[0] = e_list[0][1:]
            e_list[-1] = e_list[-1][:-1]
            e_list = [float(val) for val in e_list]
            e_list = np.array(e_list)
            acc += e_list
        mean_vals[:, i] = acc/10
    return mean_vals


def bootstrapped_roc2(y_trues_gb, y_hats_gb, y_trues_rf, y_hats_rf, all_plots=True):
    """
    Plot 2 ROC curves from bootstrapped Gradient Boost & Random Forest Models

    """
    fig, ax = plt.subplots(figsize=(11,8))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for y_true, y_hat in zip(y_trues_gb, y_hats_gb):
        fpr, tpr, thresholds = roc_curve(y_true, y_hat)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        if all_plots:
            plt.plot(fpr, tpr, 'purple', lw=1, alpha=0.1)
    mean_tpr /= len(y_trues_gb)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, 'purple', alpha=0.6, label='GB Model (AUC = %0.2f)' % mean_auc, lw=3)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for y_true, y_hat in zip(y_trues_rf, y_hats_rf):
        fpr, tpr, thresholds = roc_curve(y_true, y_hat)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        if all_plots:
            plt.plot(fpr, tpr, 'g', lw=1, alpha=0.01)
    mean_tpr /= len(y_trues_rf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, 'g', alpha=0.6, label='RF Model (AUC = %0.2f)' % mean_auc, lw=3)

    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Mean ROC: Boosting vs. Forest', size='xx-large')
    plt.legend(loc="lower right")
    ax.legend(fontsize='large')
    ax.xaxis.set_tick_params(labelsize = 12)
    ax.yaxis.set_tick_params(labelsize = 12)
    plt.show()
    return plt, fig, ax


def plot_feat_scores(fit_model, feat_names, filepath=None):
    """
    Create a horizontal bar plot of the feature importances in decreasing order
    of importance. Optionally save the figure to the filepath.

    Input
    ------
    fit_model : A model with attribute "feature_importances_", already fit.

    Output
    ------
    fig, ax : Matplotlib plotting objects
    """

    feat_scores = pd.DataFrame({
    'Fraction of Samples Affected' : fit_model.feature_importances_ },
                           index=feat_names)
    feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')

    fig, ax = plt.subplots(figsize=(8,11))
    feat_scores.plot(ax=ax, kind='barh', color='g')
    ax.xaxis.set_tick_params(labelsize = 12)
    ax.yaxis.set_tick_params(labelsize = 14)
    ax.set_title('Feature Importances', size='xx-large')
    ax.legend(fontsize='x-large', loc='lower right')
    if filepath:
        fig.savefig(filepath)
    return fig, ax


def plot_pd_bootstraps(model, boot_models):
    """
    Create a partial dependence plot of a feature in a model, along with its
    bootstrapped partial dependences.

    Input
    ------
    model : The full fit model
    boot_models : A list of models fit to bootstrapped samples of the data set 
    """
    fig, ax = plt.subplots(figsize=(12,3))
    for M in boot_models:
        plotting_functions.plot_partial_dependence(ax, M, df, 'age', color='g', alpha=0.05)
    plotting_functions.plot_partial_dependence(ax, model, df, 'age', color='g', linewidth=3)
    return fig, ax


def plot_partial_dependence(ax, model, X, var_name,
                            n_points=250, **kwargs):
    """
    Create a partial dependence plot of a feature in a model.

    Input
    ----------
    ax: A matplotlib axis object to draw the partial dependence plot on.
    model: A trained sklearn model.  Must implement a `predict` method.
    X: The raw data to use in making predictions when drawing the partial
    dependence plot. Must be a pandas DataFrame.
    var_name: A string, the name of the variable to make the partial dependence
    plot of.
    n_points: The number of points to use in the grid when drawing the plot.

    Very slightly adapted from madrury
    """
    Xpd = make_partial_dependence_data(X, var_name, n_points)
    x_plot = Xpd[var_name]
    y_hat = model.predict_proba(Xpd)[:,1]
    ax.plot(x_plot, y_hat, **kwargs)


def make_partial_dependence_data(X, var_name, n_points=250):
    Xpd = np.empty((n_points, X.shape[1]))
    Xpd = pd.DataFrame(Xpd, columns=X.columns)
    all_other_var_names = set(X.columns) - {var_name}
    for name in all_other_var_names:
        if is_numeric_array(X[name]):
            Xpd[name] = X[name].mean()
        else:
            # Array is of object type, fill in the mode.
            array_mode = mode(X[name])[0][0]
            Xpd[name] = mode
    min, max = np.min(X[var_name]), np.max(X[var_name])
    Xpd[var_name] = np.linspace(min, max, num=n_points)
    return Xpd

def is_numeric_array(arr):
    """Check if a numpy array contains numeric data.
    Source:
        https://codereview.stackexchange.com/questions/128032
    """
    numerical_dtype_kinds = {'b', # boolean
                             'u', # unsigned integer
                             'i', # signed integer
                             'f', # floats
                             'c'} # complex
    return arr.dtype.kind in numerical_dtype_kinds
