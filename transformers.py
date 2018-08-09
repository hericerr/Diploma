# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 09:08:48 2018

@author: Jan
"""
import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder, Imputer, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

import utils

        
class NaN_filler():
    
    def fit(self, X_df):
        X = X_df.copy()
        numeric_feats = X.dtypes[X.dtypes != "category"].index
        cat_feats = X.dtypes[X.dtypes == "category"].index 
        
        means = X[numeric_feats].mean()
        modes = X[cat_feats].mode().iloc[0]
        
        self.numeric_feats = numeric_feats
        self.cat_feats = cat_feats
        self.means = means
        self.modes = modes
        
    def transform(self, X_df):
        X = X_df.copy()
        X[self.numeric_feats] = X[self.numeric_feats].fillna(self.means)
        X[self.cat_feats] = X[self.cat_feats].fillna(self.modes)
        return X
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X) 

class OHE_transformer(object):
    
    def fit(self, X_df):
        X = X_df.copy()
        cat_feats = X.dtypes[X.dtypes == "category"].index
        cat_dict = {}
        for feature in cat_feats:
            X[feature] = X[feature].astype(str)
            categories = X[feature].dropna().unique()
            cat_dict[feature] = categories
        self.cat_dict = cat_dict
        
    def transform(self, X_df):
        X = X_df.copy()
        for feature in self.cat_dict.keys():
            X[feature] = X[feature].astype(str)
            X[feature] = X[feature].astype('category', categories=self.cat_dict[feature])
        return pd.get_dummies(X, columns=self.cat_dict.keys())
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
class cat_to_num(object):
    
    def fit(self, X_df):
        X = X_df.copy()
        cat_feats = X.dtypes[X.dtypes == "object"].index
        label_dict = {}
        for feat in cat_feats:            
            le = LabelEncoder()
            X[feat] = le.fit_transform(X[feat])
            label_dict[feat] = le
        self.label_dict = label_dict
        self.cat_feats = cat_feats
        
    def transform(self, X_df):
        X = X_df.copy()
        for feat in self.cat_feats:            
            X[feat] = self.label_dict[feat].transform(X[feat])
        return X
        
    def inverse_transform(self, X_df):
        X = X_df.copy()
        for feat in self.cat_feats:            
            X[feat] = self.label_dict[feat].inverse_transform(X[feat])
        return X
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
class OHE_sfs(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.test = ""
    
    def fit(self, X, *_):
        X_df = pd.DataFrame(X)
        label_dict = {}
        cat_bool = []
        for feat in X_df.columns:
            if utils.get_col_dtype(X_df[feat]) == "object":            
                le = LabelEncoder()
                X_df[feat] = le.fit_transform(X_df[feat])
                label_dict[feat] = le
                cat_bool.append(True)
            else:
                cat_bool.append(False)   
        ohe = OneHotEncoder(categorical_features=np.array(cat_bool))
        ohe.fit(X_df.values)
        
        self.ohe = ohe
        self.label_dict = label_dict
        
        return self
        
    def transform(self, X, *_):
        X_df = pd.DataFrame(X)
        for feat in self.label_dict.keys():            
            X_df[feat] = self.label_dict[feat].transform(X_df[feat])
        return self.ohe.transform(X_df.values)
        
class WoE_transformer(object):
    def __init__(self, WOE_MIN=-20, WOE_MAX=20):
        self._WOE_MIN = WOE_MIN
        self._WOE_MAX = WOE_MAX

    def fit(self, X_df, y, event=1):

        X = X_df.copy()
        cat_feats = X.dtypes[X.dtypes == "category"].index
        
        res_woe_dict = {}
        res_iv_dict = {}
        for feat in cat_feats:
            X[feat] = X[feat].astype(str)
            x = X[feat].values
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe_dict[feat] = woe_dict
            res_iv_dict[feat] = iv1
        
        self.cat_feats = cat_feats
        self.woe = res_woe_dict
        self.iv = res_iv_dict
        
    def transform(self, X_df, y):
        X = X_df.copy()
        for feat in self.cat_feats:
            X[feat] = X[feat].astype(str)
            X[feat].replace(self.woe[feat], inplace=True)
        return X
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)        
    

    def woe_single_x(self, x, y, event=1):

        event_total, non_event_total = self.count_binary(y, event=event)
        x_labels = np.unique(x)
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv
        
    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count