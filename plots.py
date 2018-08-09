# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 13:36:36 2018

@author: Jan
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

import utils

def log_BG_odds(x):
    """binary, bad=1"""
    dist_bad = np.mean(x)
    return np.log(dist_bad/(1-dist_bad))  

def dependence_plot(label_col, feature, bins=30):
    
    df = pd.DataFrame()
    
    df["label"] = label_col
    df[feature.name] = feature
    df.dropna(inplace=True)
    df["Group"] = pd.qcut(df[feature.name], q=bins)

    agg = df.groupby("Group").agg({"label" : log_BG_odds, feature.name : "median"})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.distplot(df[feature.name], ax=ax1)    
    sns.regplot(agg[feature.name], agg["label"], ax=ax2)

    ax1.set_title('Histogram')
    ax2.set_title('RegPlot')
    
    ax1.set_ylabel("frequency")
    ax2.set_ylabel("log B/G odds")    
    
    plt.suptitle(feature.name)

    plt.show()
        

    
def dependence_cat(label_col, feature, estimator=log_BG_odds, figsize=None, y_label="log_BG_odds"):
    
    df = pd.DataFrame()
    
    df["label"] = label_col
    df[feature.name] = feature
    
    fig, ax1 = plt.subplots(figsize=figsize)
    sns.countplot(feature, ax=ax1, color="#448ee4")
    
    ax2 = ax1.twinx()
    sns.pointplot(y="label", x=feature.name, data=df, estimator=estimator, ax=ax2, color="k")
    ax2.set_ylabel(y_label)
    
    plt.show()
    
def plot_dependencies(df, label):
    
    X = df.copy()
    label_col = X.pop(label)
    
    problem_feats = []
    
    numeric_feats = X.dtypes[X.dtypes != "object"].index
    cat_feats = X.dtypes[X.dtypes == "object"].index
    
    for feature in numeric_feats:
        try:
            dependence_plot(label_col, X[feature], bins=30)
        except:
            try:
               dependence_plot(label_col, X[feature], bins=20)
            except:
                try:
                    dependence_plot(label_col, X[feature], bins=10)
                except:
                    try:
                        dependence_plot(label_col, X[feature], bins=5)
                    except:
                        problem_feats.append(feature)
                    
    for feature in cat_feats:
        dependence_cat(label_col, X[feature])
        
    if len(problem_feats) != 0:
        print("Problematic features:")
        for f in problem_feats:
            print(f)
            
    for feature in problem_feats:
        dependence_cat(label_col, X[feature].apply(utils.make_bins, tresholds=[np.median(X[feature])]))
        
    return problem_feats    

def plot_correlation_matrix(d, figsize=(11, 9)):
    
    # Compute the correlation matrix
    corr = d.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        
    plt.show()
    
def FeaturesImportanceLM(model, columns, head=None, figsize=(11, 9)):
    if head is None:
        head = len(columns)
    coefs = pd.Series(model.coef_[0], index = columns)
    print("Linear model picked " + str(sum(coefs > 0.0001)) + " features and eliminated the other " +  str(sum(coefs < 0.0001)) + " features")
    if head != len(columns):        
        imp_coefs = pd.concat([coefs.sort_values().head(head),
                             coefs.sort_values().tail(head)])
    else:
        imp_coefs = coefs.sort_values()
    f, ax = plt.subplots(figsize=figsize)
    imp_coefs.plot(kind = "barh")
    plt.title("Coefficients in the Linear Model")
    plt.show()
    
    
def FeaturesImportanceTree(model, columns, head=None, figsize=(11, 9), plot=True, ret_idx=False):
    if head is None:
        head = len(columns)
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(head) + .5
    
    if plot:
        f, ax = plt.subplots(figsize=figsize)
        plt.barh(pos, feature_importance[sorted_idx][-head:], align='center')
        plt.yticks(pos, columns[sorted_idx][-head:])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()
    
    if ret_idx:
        return sorted_idx
    
def plot_ROC_curve(model, X_test, y_test, preds=None, figsize=None):
    
    if model is None:
        y_score = preds
    else:    
        y_score = model.predict_proba(X_test.values)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    
    plt.figure(figsize=figsize)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
        
def plot_ROC_multiple(label, X_test, names_list, models_list, figsize=(10,10)):
    
    f, ax = plt.subplots(figsize=figsize)
    
    for i in range(len(models_list)):
        score = models_list[i].predict_proba(X_test.values)[:,1]
        fpr, tpr, _ = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)
        
        lw = 2
        plt.plot(fpr, tpr,
             lw=lw, label='%s (area = %0.3f)' % (names_list[i], roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()