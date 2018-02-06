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
