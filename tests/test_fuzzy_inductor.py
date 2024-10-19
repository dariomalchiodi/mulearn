import logging
import os
import pickle
import unittest
import warnings

from sklearn.datasets import load_iris
import sklearn.metrics as metrics

from mulearn import FuzzyInductor
from mulearn.kernel import LinearKernel, PolynomialKernel, GaussianKernel
from mulearn.kernel import HomogeneousPolynomialKernel, HyperbolicKernel


class TestCrispFuzzifier(unittest.TestCase):
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

    def test_persistence(self):
        fi = FuzzyInductor()
        fi.fit(self.X, self.y)

        with open('object.pickle', 'wb') as f:
            pickle.dump(fi, f)

        with open('object.pickle', 'rb') as f:
            fi_clone = pickle.load(f)

        os.remove('object.pickle')

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

if __name__ == '__main__':
    unittest.main()