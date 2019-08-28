# roc_plot.py
# plot roc curve

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, roc_curve, auc


def roc_plot(y_true, y_scores, title):    
    '''
    plots an roc curve
    Input:
        y_true:     array. true binary labels for y. Positive label should = 1.
        y_scores:   array. Target scores, can either be probability estimates of the positive class, confidence values, 
                    or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).
        title:      str. name the plot
    '''
    
    n_classes = 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    print(fpr, tpr)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    
    # plotting functions
    plt.figure() 
    lw = 2 
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') 
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title(title) 
    plt.legend(loc="lower right") 
    plt.show()


def plot_confuse(y_true, y_pred):
    '''
    creates a small confusion matrix table (2x2)
    ---
    Input:
        y_true:     list. actual labels (ground truth) from the test set
        y_pred:     list. predictions of the label for each corresponding element in test set
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() 
    header = [np.array(['Actual','Actual']), np.array(['True','False'])] 
    indexer = [np.array(['Predicted','Predicted']), np.array(['True','False'])] 
    print(pd.DataFrame([[tp,fp], [fn, tn]], columns = header, index = indexer))
    conf = np.array([[tp, fp], [fn, tn]])
    return conf

