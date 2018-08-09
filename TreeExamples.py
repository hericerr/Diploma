# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 23:16:36 2018

@author: Jan
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_classification

def test_models_classification(Xtrain, Ytrain, Xtest, Ytest, models):
    tt = datetime.now()
    num = 1
    for model in models:
        print("")
        print("Model%i:"%num,model)
        t0 = datetime.now()
        model.fit(Xtrain, Ytrain)

        print("Training time:", (datetime.now() - t0))
        
        t0 = datetime.now()
        print("Train accuracy:", model.score(Xtrain, Ytrain))
        print("Time to compute train accuracy:", (datetime.now() - t0))
        
        t0 = datetime.now()
        print("Test accuracy:", model.score(Xtest, Ytest))
        print("Time to compute test accuracy:", (datetime.now() - t0))
        num += 1
    print("")
    print("Total duration:"), (datetime.now() - tt)
    
def test_models_regression(Xtrain, Ytrain, x_axis, y_axis, models, plot=True):
    tt = datetime.now()
    num = 1
    for model in models:
        print("")
        print("Model%i"%num, model)

        t0 = datetime.now()
        model.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))
        
        t0 = datetime.now()
        prediction = model.predict(x_axis.reshape(len(y_axis), 1))
        print("Time to compute predictions:", (datetime.now() - t0))
        print("R_square:",  model.score(Xtrain, Ytrain))
        
        if plot:
            plt.plot(x_axis, prediction, label="Prediction")
            plt.plot(x_axis, y_axis, label="f(x)")
            plt.scatter(Xtrain, Ytrain, label="Data")
            plt.legend()
            plt.show()
        num += 1
    print("")
    print("Total duration:", (datetime.now() - tt))
    
#REGRESSION TEST
# create the data
T = 100
x_axis = np.linspace(0, 2*np.pi, T)
y_axis = np.sin(x_axis)
# get the training data
N = 30
idx = np.random.choice(T, size=N, replace=False)
Xtrain = x_axis[idx].reshape(N, 1) 
Ytrain = y_axis[idx] + np.random.randn(N)*0.5

reg_models = [DecisionTreeRegressor(max_depth=1),
              DecisionTreeRegressor(max_depth=3),
              DecisionTreeRegressor(max_depth=5),
              DecisionTreeRegressor()
              ]

test_models_regression(Xtrain, Ytrain, x_axis, y_axis, reg_models, plot=True)



#CLASSIFICATION TEST
h = .02  # step size in the mesh

names = ["1", "3", "5", "None"]

classifiers = [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=3),
        DecisionTreeClassifier(max_depth=5),
        DecisionTreeClassifier()
    ]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0)
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()