import copy
import itertools as it
import logging
import multiprocessing as mp
import pickle
import time
import unittest
import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.datasets import load_iris
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score

from mulearn import FuzzyInductor
from mulearn.fuzzifier import ExponentialFuzzifier
from mulearn.kernel import LinearKernel, PolynomialKernel, GaussianKernel
from mulearn.kernel import HyperbolicKernel

RANDOM_STATE = 42
NUM_CORES = mp.cpu_count()

def make_hp_configurations(grid):
    return [{n: v for n, v in zip(grid.keys(), t)}
            for t in it.product(*grid.values())]

def fit_and_score(estimator,
                  X_trainval, y_trainval,
                  hp_configuration, model_selection,
                  scorer=metrics.root_mean_squared_error):

    estimator.set_params(**hp_configuration)
    current_scores = []
    for train_index, val_index in model_selection.split(X_trainval, y_trainval):
        X_train, X_val = X_trainval[train_index], X_trainval[val_index]
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]

        estimator.fit(X_train, y_train)
        y_hat = estimator.predict(X_val)
        score = scorer(y_val, y_hat)
        current_scores.append(score)

    return np.mean(current_scores), hp_configuration

def learn_parallel(X, y, estimator, param_grid,
                   model_selection=StratifiedKFold(n_splits=5,
                                                   shuffle=True,
                                                   random_state=RANDOM_STATE),
                   model_assessment=StratifiedKFold(n_splits=5,
                                                    shuffle=True,
                                                    random_state=RANDOM_STATE),
                   gs_scorer=metrics.root_mean_squared_error,
                   test_scorers=[metrics.root_mean_squared_error,
                                 metrics.hinge_loss],
                   test_scorer_names=['RMSE', 'Hinge'],
                   n_jobs=-1, pre_dispatch=None):

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    ping = time.time()

    outer_scores = []

    for trainval_index, test_index in model_assessment.split(X, y):
        X_trainval, X_test = X[trainval_index], X[test_index]
        y_trainval, y_test = y[trainval_index], y[test_index]

        gs_result = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)( \
                    delayed(fit_and_score)(copy.deepcopy(estimator),
                                           X_trainval, y_trainval,
                                           hp_conf,
                                           model_selection=model_selection,
                                           scorer=gs_scorer)
                            for hp_conf in make_hp_configurations(param_grid))

        best_conf = sorted(gs_result, key=lambda t: t[0])[0][1]
        estimator.set_params(**best_conf)
        estimator.fit(X_trainval, y_trainval)

        y_hat = estimator.predict(X_test)
        outer_scores.append([score(y_test, y_hat) for score in test_scorers])

    pong = time.time()
    # Refit estimator with best configuration
    # of last external cv fold on all data
    estimator.fit(X, y)

    avg = np.mean(outer_scores, axis=0)
    std = np.std(outer_scores, axis=0, ddof=1)
    result = {'model': estimator.__class__.__name__, 'type': 'FINAL'} | \
             {n + ' mean': m for n, m in zip(test_scorer_names, avg)} | \
             {n + ' std': s for n, s in zip(test_scorer_names, std)} | \
             {'time': pong-ping}

    return estimator, best_conf, result


class TestFuzzyInductor(unittest.TestCase):
    def setUp(self):
        d = load_iris()
        self.X = d['data']
        y = d['target']
        y[y==2] = 0
        self.y = y

    def test_serialization(self):
        fi = FuzzyInductor()
        fi.fit(self.X, self.y)
        s = pickle.dumps(fi)
        fi_clone = pickle.loads(s)

        self.assertEqual(fi, fi_clone)

    def test_fit(self):
        kernel = [LinearKernel(), PolynomialKernel(2),
                  GaussianKernel(.1), HyperbolicKernel()]
        scores = [0.3679416879198661, 0.3954864950288751,
                  3.858380026406454e-08, 0.5]
        logging.disable(logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k, s in zip(kernel, scores):
                fi = FuzzyInductor(k=k)
                fi.fit(self.X, self.y)
                y_hat = fi.predict(self.X)
                rmse = metrics.root_mean_squared_error(self.y, y_hat)
                self.assertAlmostEqual(s, rmse)
        logging.disable(logging.NOTSET)
    
    def test_standard_train(self):
        model = FuzzyInductor(fuzzifier=ExponentialFuzzifier(profile='fixed'))

        grid = {'c': np.linspace(0.1, 0.2, 2),
                'k': [GaussianKernel(.01), GaussianKernel(.1)]}
        cv_out = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=RANDOM_STATE)
        cv_in = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=RANDOM_STATE)
        gs = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error',
                          cv=cv_in, n_jobs=NUM_CORES, pre_dispatch=2*NUM_CORES)
        score = cross_val_score(gs, self.X, self.y,
                                scoring='neg_root_mean_squared_error',
                                cv=cv_out)
        target = np.array([-0.49120162, -0.54772224, -0.54772203,
                           -0.51639634, -0.47835032])

        for t, s in zip(score, target):
            self.assertAlmostEqual(t, s)
    
    def test_custom_train(self):
        model = FuzzyInductor()

        grid = {'c': np.linspace(0.1, 0.3, 2),
                'k': [GaussianKernel(.1), GaussianKernel(.01)]}

        n_cores = mp.cpu_count()
        model, best_conf, result = learn_parallel(self.X, self.y, model, grid,
                        n_jobs=NUM_CORES, pre_dispatch=2*NUM_CORES)

        result = {'configuration': best_conf} | result
        self.assertAlmostEqual(result['RMSE mean'], 0.5162785111296674,
                               delta=1E-5)
        self.assertAlmostEqual(result['RMSE std'], 0.03179943202573793,
                               delta=1E-4)

if __name__ == '__main__':
    unittest.main()