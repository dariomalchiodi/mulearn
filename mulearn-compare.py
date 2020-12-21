#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:06:53 2020.

@author: malchiodi
"""

import logging
import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

# import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, \
                                  MinMaxScaler, QuantileTransformer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, \
                                    KFold, StratifiedKFold
import sklearn.metrics as metrics
import sklearn.datasets as ds

from tqdm import tqdm

# from fcm import FCM
from mulearn import FuzzyInductor
from mulearn.fuzzifier import *
from mulearn.kernel import GaussianKernel
from mulearn.optimization import TensorFlowSolver
from mulearn.distributions import GaussianKernelDistribution, \
                                  ExponentialFuzzifierDistribution

from scipy.stats import uniform


logging.basicConfig(
    level=logging.INFO,
    format='[{%(asctime)s %(filename)s:'
           '%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(filename='mulearn-compare.log'),
        # logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('LOGGER_MU-LEARN_COMPARE')


def _make_experiment(category, name, learning_algorithm, learning_params,
                     X, y, outer_folds=7, inner_folds=5, logger=None):
    if category not in ('classification', 'regression'):
        raise ValueError("'category' should be either equal to "
                         "'classification' or 'regression' "
                         f"(found {category})")
    if logger:
        logger.info(f'starting experiment: {name}')

    pipeline_desc = [('scaler', None),
                     ('learning_algorithm', learning_algorithm)]
    pipe = Pipeline(pipeline_desc)

    scalers = [StandardScaler(), RobustScaler(),
               MinMaxScaler(), QuantileTransformer(n_quantiles=50)]

    params = {'scaler': scalers}
    for k in learning_params:
        params['learning_algorithm__' + k] = learning_params[k]

    fold_gen = StratifiedKFold if category == 'classification' else KFold
    outer_fold = fold_gen(n_splits=outer_folds)
    scores = []
    best_models = []
    best_params = []

    progress = tqdm(outer_fold.split(X, y),
                    total=outer_fold.get_n_splits(),
                    desc=name,
                    leave=False)

    for train_idx, test_idx in progress:
        if logger:
            logger.info(f'Outer fold {progress.n}')
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        gs = RandomizedSearchCV(pipe, params, verbose=0, cv=inner_folds,
                                error_score=np.nan, n_jobs=-1,
                                pre_dispatch=10)
        gs = gs.fit(X_train, y_train)

        predictions = gs.predict(X_test)
        perf = _f1_score if category == 'classification' else _rmse
        score = perf(y_test, predictions)
        scores.append(score)

        best_models.append(gs.best_estimator_)
        best_params.append(gs.best_params_)
        if logger:
            logger.info(f'score {score}')
            logger.info(f'best params {gs.best_params_}')

    if logger:
        logger.info('ended experiment.')

        logger.info(f'mean test error {np.mean(scores)}')

    progress.close()
    print(f'{name}: {np.mean(scores):.3f}')
    return scores, best_models, best_params


def _f1_score(labels, predictions):
    return metrics.f1_score(labels,
                            [1 if p > 0.5 else 0 for p in predictions])


def _rmse(labels, predictions):
    return metrics.mean_squared_error(labels, predictions, squared=False)


def gr_membership_contour(estimated_membership, fig):
    print(estimated_membership([(1, 1)]))
    x = np.linspace(2, 8, 100)
    y = np.linspace(-.5, 3, 100)
    X, Y = np.meshgrid(x, y)
    # zs = np.array([estimated_membership((x, y))
    #                for x, y in zip(np.ravel(X), np.ravel(Y))])
    zs = estimated_membership(np.array((np.ravel(X), np.ravel(Y))).T)
    Z = zs.reshape(X.shape)

    fig.add_trace(go.Contour(x=x, y=y, z=Z,
                             colorscale='Blues', line_smoothing=0.85,
                             contours={"start": 0, "end": 1, "size": .2,
                                       "showlabels": True,
                                       "labelfont": {"size": 12,
                                                     "color": "white"}
                                       }))


if __name__ == '__main__':
    os.environ['GRB_LICENSE_FILE'] = "/home/malchiodi/.gurobi/gurobi.lic"

    datasets = {}

    iris_X, iris_y = ds.load_iris(return_X_y=True)
    # Only focus on virginica and versicolor classes.
    iris_X = iris_X[iris_y != 0]
    # Subtract 1 to class labels in order to get 0 and 1 as new labels.
    iris_y = iris_y[iris_y != 0] - 1
    datasets['iris'] = (iris_X, iris_y, 'classification')

    boston_X, boston_y = ds.load_boston(return_X_y=True)
    boston_y = np.digitize(boston_y, np.quantile(boston_y, [.3, .5, .7]))/3
    datasets['boston'] = (boston_X, boston_y, 'regression')

    diab_X, diab_y = ds.load_diabetes(return_X_y=True)
    diab_y = np.digitize(diab_y, np.quantile(diab_y, [.3, .5, .7]))/3
    datasets['diabetes'] = (diab_X, diab_y, 'regression')

    breast_X, breast_y = ds.load_breast_cancer(return_X_y=True)
    datasets['breast'] = (breast_X, breast_y, 'classification')

    digits_X, digits_y = ds.load_digits(return_X_y=True)
    for d in range(10):
        datasets[f'digit-{d}'] = (digits_X,
                                  np.array(digits_y == d, dtype=np.float32),
                                  'classification')

    algs = {}

    # algs['FCM'] = (FCM(), {'m': np.arange(2, 7)})

    fi = FuzzyInductor(fuzzifier=ExponentialFuzzifier(profile="infer"),
                       k=GaussianKernel(.7))
                      # solver=TensorFlowSolver())

    iris_X = iris_X[:, 2:]
    fi.fit(iris_X, iris_y)
    print(fi.score(iris_X, iris_y))


    fig = go.Figure()
    gr_membership_contour(fi.decision_function, fig)
    fig.add_trace(go.Scatter(x=iris_X[iris_y == 0][:, 0],
                             y=iris_X[iris_y == 0][:, 1],
                             mode="markers",
                             marker_color="green"))
    fig.add_trace(go.Scatter(x=iris_X[iris_y == 1][:, 0],
                             y=iris_X[iris_y == 1][:, 1],
                             mode="markers",
                             marker_color="blue"))
    fig.show()

    params = {'c': uniform(loc=0, scale=10),
              'k': GaussianKernelDistribution(low=0.001, high=10),
              'fuzzifier': ExponentialFuzzifierDistribution()}

    # params = {'c': uniform(loc=0, scale=10),
    #           'k': GaussianKernelDistribution(low=0.001, high=10),
    #           'fuzzifier': [(LinearFuzzifier, {'profile': 'fixed'}),
    #                         (LinearFuzzifier, {'profile': 'infer'}),
    #                         (ExponentialFuzzifier, {'profile': 'fixed'}),
    #                         (ExponentialFuzzifier, {'profile': 'infer'})] +
    #                        [(ExponentialFuzzifier, {'profile': 'alpha',
    #                                                 'alpha': a})
    #                         for a in np.arange(.1, 1, .1)]}

    # algs['FI'] = (fi, params)
    #
    # for alg_name in algs:
    #     alg, grid = algs[alg_name]
    #     for i, name in enumerate(datasets):
    #         X, y, category = datasets[name]
    #         score, _, _ = _make_experiment(category,
    #                                        f'{alg_name}+{name} ({i+1} of '
    #                                        f'{len(datasets)})',
    #                                        alg, grid,
    #                                        X, y, logger=logger)
