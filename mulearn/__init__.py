
import numpy as np
import copy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError

from mulearn.kernel import GaussianKernel
from mulearn.optimization import GurobiSolver
from mulearn.fuzzifier import ExponentialFuzzifier

import logging

import warnings
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import FitFailedWarning

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


class FuzzyInductor(BaseEstimator, RegressorMixin):
    """FuzzyInductor class."""

    def __init__(self,
                 c=1,
                 k=GaussianKernel(),
                 fuzzifier=ExponentialFuzzifier(), # noqa
                 solver=GurobiSolver(),
                 random_state=None):
        r"""Create an instance of :class:`FuzzyInductor`.

        :param c: Trade-off constant, defaults to 1.
        :type c: `float`
        :param k: Kernel function, defaults to :class:`GaussianKernel()`.
        :type k: :class:`mulearn.kernel.Kernel`
        :param fuzzifier: fuzzifier mapping distance values to membership
           degrees, defaults to `ExponentialFuzzifier()`.
        :type fuzzifier: :class:`mulearn.fuzzifier.Fuzzifier`
        :param solver: Solver to be used to obtain the optimization problem
          solution, defaults to `GurobiSolver()`.
        :type solver: :class:`mulearn.optimization.Solver`
        :param random_state: Seed of the pseudorandom generator.
        :type random_state: `int`
        """
        self.c = c
        self.k = k
        self.fuzzifier = fuzzifier
        self.solver = solver
        self.random_state = random_state
        self.estimated_membership_ = None
        self.x_to_sq_dist_ = None
        self.chis_ = None
        self.gram_ = None
        self.fixed_term_ = None
        self.train_error_ = None

    def __repr__(self, **kwargs):
        return f"FuzzyInductor(c={self.c}, k={self.k}, f={self.fuzzifier}, " \
               f"solver={self.solver})"

    def _fix_object_state(self, X, y):
        """Ensure object consistency."""
        self.X = X
        self.y = y

        def x_to_sq_dist(x_new):
            ret = self.k.compute(x_new, x_new) \
                  - 2 * np.array([self.k.compute(x_i, x_new)
                                  for x_i in X]).dot(self.chis_) \
                  + self.fixed_term_
            return ret

        self.fuzzifier.x_to_sq_dist = x_to_sq_dist

        chi_SV_index = [i for i, (chi, mu) in enumerate(zip(self.chis_, y))
                        if -self.c * (1 - mu) < chi < self.c * mu]

        chi_sq_radius = map(x_to_sq_dist, X[chi_SV_index])
        chi_sq_radius = list(chi_sq_radius)

        if len(chi_sq_radius) == 0:
            self.estimated_membership_ = None
            self.train_error_ = np.inf
            self.chis_ = None
            logger.warning('No support vectors found')
            return self

        self.fuzzifier.sq_radius_05 = np.mean(chi_sq_radius)
        self.fuzzifier.fit(X, y)

        return self.fuzzifier.get_membership()

    def fit(self, X, y, warm_start=False):
        r"""Induce the membership function starting from a labeled sample.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the
          same length
        :param y: Membership for the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :param warm_start: flag triggering the non reinitialization of
          independent variables of the optimization problem, defaults to
          None.
        :type warm_start: `bool`
        :raises: ValueError if the values in `y` are not between 0 and 1, if
          `X` and have different lengths, or if `X` contains elements of
          different lengths.
        :returns: self -- the trained model.
        """
        if type(X) is not np.array:
            X = np.array(X)

        for e in y:
            if e < 0 or e > 1:
                raise ValueError("`y` values should belong to [0, 1]")

        check_X_y(X, y)
        self.random_state = check_random_state(self.random_state)

        if warm_start:
            check_is_fitted(self, ["chis_"])
            if self.chis_ is None:
                raise NotFittedError("chis variable are set to None")
            self.solver.initial_values = self.chis_

        self.chis_ = self.solver.solve(X, y, self.c, self.k)

        if type(self.k) is kernel.PrecomputedKernel:
            self.gram_ = self.k.kernel_computations
        else:
            self.gram_ = np.array([[self.k.compute(x1, x2) for x1 in X]
                                   for x2 in X])
        self.fixed_term_ = np.array(self.chis_).dot(self.gram_.dot(self.chis_))

        self.estimated_membership_ = self._fix_object_state(X, y)

        self.train_error_ = np.mean([(self.estimated_membership_(x) - mu) ** 2
                                     for x, mu in zip(X, y)])

        return self

    def decision_function(self, X):
        r"""Compute predictions for the membership function.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the
          same length
        :returns: array of float -- the predictions for each value in `X`.
        """
        check_is_fitted(self, ['chis_', 'estimated_membership_'])
        X = check_array(X)
        return np.array([self.estimated_membership_(x) for x in X]) # noqa

    def predict(self, X, alpha=None):
        r"""Compute predictions for membership to the set.

        Predictions are either computed through the membership function (when
        `alpha` is set to a float in [0, 1]) or obtained via an $\alpha$-cut on
        the same function (when `alpha` is set to `None`).

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param alpha: $\alpha$-cut value, defaults to `None`.
        :type alpha: float
        :raises: ValueError if `alpha` is set to a value different from
          `None` and not included in $[0, 1]$.
        :returns: array of int -- the predictions for each value in `X`.
        """
        check_is_fitted(self, ['chis_', 'estimated_membership_'])
        X = check_array(X)
        mus = np.array([mu for mu in self.decision_function(X)])
        if alpha is None:
            return mus
        else:
            if alpha < 0 or alpha > 1:
                raise ValueError("alpha cut value should belong to [0, 1]"
                                 f" (provided {alpha})")
            return np.array([1 if mu >= alpha else 0 for mu in mus])

    def score(self, X, y, **kwargs):
        r"""Compute the fuzzifier score.

        Score is obtained as the opposite of MSE between predicted
        membership values and labels.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Labels containing the *gold standard* membership values
          for the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :returns: `float` -- opposite of MSE between the predictions for the
          elements in `X` w.r.t. the labels in `y`.
        """
        check_X_y(X, y)

        if self.estimated_membership_ is None:
            return -np.inf
        else:
            return -np.mean([(self.estimated_membership_(x) - mu) ** 2
                             for x, mu in zip(X, y)])

    def __getstate__(self):
        """Return a serializable description of the fuzzifier."""
        d = copy.deepcopy(self.__dict__)
        del d['estimated_membership_']
        del d['x_to_sq_dist_']
        print(d)
        return d

    def __setstate__(self, d):
        """Ensure fuzzifier consistency after deserialization."""
        self.__dict__ = d
        try:
            check_is_fitted(self, ['chis_', 'estimated_membership_'])
            self._fix_object_state(self.X, self.y)
            self.__dict__['estimated_membership_'] = self.estimated_membership_
            self.__dict__['x_to_sq_dist_'] = self.x_to_sq_dist_
            self.fuzzifier.x_to_sq_dist = self.x_to_sq_dist_
        except NotFittedError:
            pass
