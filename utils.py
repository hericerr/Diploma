# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:26:29 2018

@author: Jan
"""
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer

import plots
import transformers as tran


def print_full(x):
    """
    print full pd DataFrame
    """
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
    
def stratified_train_test_split(df, label, test_size=0.3, random_state=None, verbose=True):
    """
    Train-test split stratified by binary label column
    df : data (pd DataFrame)
    label : label column name (string)
    test_size : proportion of test data (float 0.0-1.0)
    random_state : (None or int)
    verbose : wether to print stats (boolean)
    returns : train_df, test_df (tuple of pd DataFrames)
    """
    
    #separate positives and negatives
    pos_df = df.loc[df[label] == 1]
    neg_df = df.loc[df[label] == 0]
    
    pos_df = pos_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    neg_df = neg_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    num_test_pos = int(len(pos_df)*test_size)
    num_test_neg = int(len(neg_df)*test_size)
    
    test_pos_df = pos_df.iloc[0:num_test_pos]
    train_pos_df = pos_df.iloc[num_test_pos:]
    
    test_neg_df = neg_df.iloc[0:num_test_neg]
    train_neg_df = neg_df.iloc[num_test_neg:]

    test_df = pd.concat([test_pos_df, test_neg_df])
    train_df = pd.concat([train_pos_df, train_neg_df])

    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)        
 
    if verbose:
        print("Train sample:")
        print("Rows total: {}".format(len(train_df)))
        print("Positive observations: {}".format(sum(train_df[label])))
        print("Negative observations: {}".format(len(train_df)-sum(train_df[label])))
        print("Positive rate: {:0.3f}%".format(np.mean(train_df[label])*100))
        print("")
        print("Test sample:")
        print("Rows total: {}".format(len(test_df)))
        print("Positive observations: {}".format(sum(test_df[label])))
        print("Negative observations: {}".format(len(test_df)-sum(test_df[label])))
        print("Positive rate: {:0.3f}%".format(np.mean(test_df[label])*100))
        
    return train_df, test_df
    
def convert_revol_util(row):
    try:
        return float(row[:-1])
    except:
        return row
        
def gini(y_true, y_score):
    return 2*roc_auc_score(y_true, y_score)-1
    
def ProbaScoreProxy(y_true, y_probs, class_idx, proxied_func, **kwargs):
    return proxied_func(y_true, y_probs[:, class_idx], **kwargs)
    
GINI_SCORER = make_scorer(ProbaScoreProxy, greater_is_better=True, needs_proba=True, class_idx=1, proxied_func=gini)

      
def get_univariate_ginis(df, label, model="logit", random_state=None):
    """
    df : pd DataFrame of data
    label : string
    model : sci-kit model, default LogisticRegression
    returns : DataFrame gini, cat, #cat (sorted by gini)
    """
    
    df = df.copy()
    
    if model =="logit":
        model = LogisticRegression(random_state=random_state)
        
        
    y = df.pop(label).values[:]
    num = df._get_numeric_data().columns
    
    features = []
    ginis = []
    types = []
    cats = [] 
        
    for feature in df.columns:
        
        if feature in num:
            X = df[feature].values.reshape(len(y),1)
            typ = "Num"
            num_cats = "N/A"
        else:
            X = pd.get_dummies(df[feature])
            typ = "Cat"
            num_cats = len(df[feature].unique())
        
        model.fit(X, y)
        preds = model.predict_proba(X)[:,1]
        score = gini(y, preds)        
        
        features.append(feature)
        ginis.append(score)
        types.append(typ)
        cats.append(num_cats)
        
    scores = pd.DataFrame()
    scores["Feature"] = features
    scores.set_index("Feature", inplace=True, drop=True)
    scores["Gini"] = ginis
    scores["Type"] = types
    scores["#categories"] = cats
  
    return scores.sort_values(by="Gini", ascending=False)
    
def make_flag(x):
    if np.isnan(x):
        return "N/A"
    else:
        return "1"
        
def make_bins(x, tresholds):    
    if np.isnan(x):
        return "N/A"    
    prev = ""    
    for t in tresholds:
        if x <= t:
            return prev+"<"+str(t)
        prev = str(t)
    return str(t)+"<"
    
def check_missing(df):
    if df.isnull().sum().sum() == 0:
        print("No missings: OK")
    else:
        null_df = df.isnull().sum()
        print("Warning set contains missing values!")
        print(null_df[null_df != 0])
        
def log_BG_odds(x):
    """binary, bad=1"""
    dist_bad = np.mean(x)
    return np.log(dist_bad/(1-dist_bad))  
    
def print_results(train_preds_proba, y_train, test_preds_proba, y_test):
    df = pd.DataFrame()
    df["metric"] = ["Train gini", "Test gini"]
    df["value"] = [gini(y_train, train_preds_proba), gini(y_test, test_preds_proba)]
    
    print(df)
    
def get_col_dtype(col):
        """
        Infer datatype of a pandas column, process only if the column dtype is object. 
        input:   col: a pandas Series representing a df column. 
        """


        if col.dtype =="object":

            # try numeric
            try:
                col_new = pd.to_datetime(col.dropna().unique())
                return col_new.dtype
            except:
                try:
                    col_new = pd.to_numeric(col.dropna().unique())
                    return col_new.dtype
                except:
                    try:
                        col_new = pd.to_timedelta(col.dropna().unique())
                        return col_new.dtype
                    except:
                        return "object"

        else:
            return col.dtype
            
def experiment(model, tansformer, train_df, test_df, label, plot=True, figsize=(11, 9)):
    
    #prepare data
    X_train = tansformer.fit_transform(train_df.drop(label, axis=1))
    y_train = train_df[label]
    
    X_test = tansformer.transform(test_df.drop(label, axis=1))
    y_test = test_df[label]
    
    #fit
    tt = datetime.now()
    model.fit(X_train.values, y_train.values)
    fit_time = datetime.now() - tt
    #predict
    train_preds_proba = model.predict_proba(X_train.values)[:,1]
    test_preds_proba = model.predict_proba(X_test.values)[:,1]
    #results
    gini_train = gini(y_train, train_preds_proba)
    gini_test =   gini(y_test, test_preds_proba)
    roc_train = roc_auc_score(y_train, train_preds_proba)
    roc_test = roc_auc_score(y_test, test_preds_proba)
    
    
    if plots:
        print(model)
        print("")
        print_results(train_preds_proba, y_train, test_preds_proba, y_test)
        plots.plot_ROC_curve(model, X_test, y_test)
        try:
            plots.FeaturesImportanceTree(model, X_train.columns, figsize=figsize)
        except:
            plots.FeaturesImportanceLM(model, X_train.columns, figsize=figsize)
        
    results_dict = {}
    
    results_dict["gini_train"] = gini_train
    results_dict["gini_test"] = gini_test
    results_dict["roc_train"] = roc_train
    results_dict["roc_test"] = roc_test
    results_dict["fit_time"] = fit_time
    results_dict["model"] = model
    
    return results_dict
    
def evaluate_param(model, X_train, y_train, parameter, num_range, index, def_params, scoring, verbose, shape):
    if verbose:
        print("evaluating {}".format(parameter))
        tt = datetime.now()
        
    param_grid = {parameter: num_range}
    for param in def_params.keys():
        if param not in param_grid.keys():
            param_grid[param] = def_params[param]
    
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]
       
    
    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')
    
    plt.subplot(shape[0],shape[1],index)
    try:
        plot = plt.plot(df['index'], df[0])
        plt.title(parameter)
    except:
        df["index"] = df["index"].astype(str)
        plot = sns.barplot(x="index", y=0, data=df, color="#448ee4", ci=None)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(parameter)
    
    if verbose:
        print("{} evaluated".format(parameter))
        print("Duration: {}".format(datetime.now()-tt))
        print("")
        
    return plot, df
    
def evaluate_all(model, X_train, y_train, param_grid, def_params=None,scoring=GINI_SCORER, verbose=True):
    if verbose:
        t = datetime.now()
        
    shape = (len(param_grid), 2)
    index = 1
    plt.figure(figsize=(16, 4*shape[0]))
    for parameter, param_range in dict.items(param_grid):   
        evaluate_param(model, X_train, y_train, parameter, param_range, index, def_params, scoring, verbose, shape)
        index += 1
        
    if verbose:
        print("Total duration: {}".format(datetime.now()-t))
        
def select_best_features(cols, num_cols=40, strip=False):
    new_cols = []
    if strip:
        for col in list(cols)[::-1]:
            if "_" in col:
                new_col = col[:col.rfind("_")]
            else:
                new_col = col
            if new_col not in new_cols:
                new_cols.append(new_col)
        return new_cols[:num_cols]
    else:
        return cols[-num_cols:]
        
def recursive_elimination_results(model, train_df, test_df, LABEL, batch=False):
    scores = []
    if not batch:
        ohe = tran.OHE_transformer()
        X_train = ohe.fit_transform(train_df.drop(LABEL, axis=1))
        X_test = ohe.transform(test_df.drop(LABEL, axis=1))
        
        model.fit(X_train.values, train_df[LABEL].values)
        imps = plots.FeaturesImportanceTree(model, X_train.columns, plot=False, ret_idx=True)
        for i in range(len(X_train.columns)):
            selected = select_best_features(X_train.columns[imps], num_cols=i+1, strip=False)
            model.fit(X_train[selected].values, train_df[LABEL].values)
            p = model.predict_proba(X_test[selected].values)[:,1]
            score = gini(test_df[LABEL], p)
            print(selected)
            print(i+1, score)
            scores.append(score)
            
    else:
        ohe = tran.OHE_transformer()
        X_train = ohe.fit_transform(train_df.drop(LABEL, axis=1))
        X_test = ohe.transform(test_df.drop(LABEL, axis=1))
         
        model.fit(X_train.values, train_df[LABEL].values)
        imps = plots.FeaturesImportanceTree(model, X_train.columns, plot=False, ret_idx=True)
        for i in range(len(train_df.columns)):
            selected = select_best_features(X_train.columns[imps], num_cols=i+1, strip=True)
            ohe_sel = tran.OHE_transformer()
            X_train_sel = ohe_sel.fit_transform(train_df[selected])
            X_test_sel = ohe_sel.transform(test_df[selected])
            model.fit(X_train_sel.values, train_df[LABEL].values)
            p = model.predict_proba(X_test_sel.values)[:,1]
            score = gini(test_df[LABEL], p)
            scores.append(score)            
                
            
    return scores
	
def foreward_selection(indata, yVar, xVar, categorical, stopn = np.inf):
    numerical = list(set(xVar)-set(categorical))
    flist = []
    flist_cat = []
    best_score = np.inf
    nx = min(len(xVar), stopn)
    
    while len(flist) < nx:
        best_score_iter = np.inf
        for i in xVar:
            if i in categorical:
                i_cat = "C("+i+")"
            else:
                i_cat = i
            newflist = flist + [i]
            newflist_cat = flist_cat + [i_cat]
            y = indata[yVar]
            X = indata[newflist]
            f = yVar + ' ~ ' + "+".join(newflist_cat)
            print(f)
            reg = smf.logit(formula = str(f), data = indata).fit()
            score = reg.bic
            if score < best_score_iter:
                best_score_iter = score
                record_i = i
                record_i_cat = i_cat
                
        flist.append(record_i)
        flist_cat.append(record_i_cat)
        
        print(flist)
        xVar.remove(record_i)
        if best_score < best_score_iter:
            break
        else:
            best_score = best_score_iter
    y = indata[yVar]
    X = indata[flist]
    finmodel =  smf.logit(formula = f, data = indata).fit()
    print(finmodel.summary())
    return finmodel, flist