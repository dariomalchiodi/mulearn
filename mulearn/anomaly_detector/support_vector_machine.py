"""This module implements a Support Vector Machine for anomaly detection
"""

import logging

from typing import Any, Literal, Self
from numpy.typing import NDArray, ArrayLike

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

import mulearn.kernel as kernel
from mulearn.optimization import GurobiSolver, TensorFlowSolver


logger = logging.getLogger(__name__)


class SupportVectorMachine(BaseEstimator):
    """Implements a Support Vector Machine model.

    :param c: Trade-off constant, defaults to 1.
    :type c: float
    :param k: Kernel function, defaults to "gaussian".
    :type k: Literal["auto", "ball_tree", "kd_tree", "brute"]
    :param k_params: Parameters for the kernel function, defaults to None.
    :type k_params: dict[str, Any] | None
    :param solver: The solver to use for the optimization problem, defaults to "gurobi".
    :type solver: Literal["gurobi", "tensorflow"]
    :param solver_params: Parameters for the solver, defaults to None.
    :type solver_params: dict[str, Any] | None

    :param n_features_in_: Number of features of the data passed to fit().
    :type n_featues_in_: int
    :param chis_: Coefficients learnt from the fitting.
    :type chis_: list[float]
    :param fixed_term_: Fixed term of the equation.
    :type fixed_term_: NDArray[float]
    :param squared_radius_:
    :type squared_radius_: float
    """

    def __init__(self,
                 c: float = 1,
                 k: Literal["linear", "polynomial", "homogeneous", "gaussian", "hyperbolic", "precomputed"] = "gaussian",
                 k_params: dict[str, Any] | None = None,
                 solver: Literal["gurobi", "tensorflow"] = "gurobi",
                 solver_params: dict[str, Any] | None = None
                 ) -> None:
        self.c = c
        self.k = k
        self.k_params = k_params
        self.solver = solver
        self.solver_params = solver_params

    def __eq__(self, other):
        """Check equality w.r.t. other objects."""
        equal = (
            type(self) == type(other) and
            self.get_params() == other.get_params()
        )
        isfitted_self = hasattr(self, "chis_")
        isfitted_other = hasattr(other, "chis_")
        if isfitted_self and isfitted_other:
            return equal and (self.chis_ == other.chis_)
        elif isfitted_self != isfitted_other:
            return False
        return equal

    def __sklearn_is_fitted__(self) -> bool:
        """Checks if this estimator has been fit

        This method is called by `check_is_fitted(self)`

        :return: If the `chis_` are set or not
        :rtype: bool
        """

        return (
            getattr(self, "chis_", None) is not None
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, warm_start: bool = False) -> Self:
        """Fits the support vector machine to the given data points `X`

        :param X: Vectors in data space
        :type X: ArrayLike
        :param y: Labels for the vectors in `X`, defaults to None for the unsupervised version
        :type y: ArrayLike | None, optional
        :param warm_start: Flag triggering the non reinitialization of
          independent variables of the optimization problem, defaults to False
        :type warm_start: bool, optional
        :raises ValueError: If the values in `y` are not between 0 and 1, if
          `X` and have different lengths, or if `X` contains elements of
          different lengths.
        :raises NotFittedError: If we set warm_start = True but the model hasn't been fit
        :return: The trained model
        :rtype: Self
        """

        X = check_array(X)

        if y is None:
            # Unsupervised version
            y = np.ones(len(X))
        else:
            X, y = check_X_y(X, y)
            for e in y:
                if e < 0 or e > 1:
                    raise ValueError("`y` values must belong to [0, 1]")

        self.n_features_in_ = X.shape[1]
        self._original_X = X.copy()

        kernel_map = {
            "linear": kernel.LinearKernel,
            "polynomial": kernel.PolynomialKernel,
            "homogeneous": kernel.HomogeneousPolynomialKernel,
            "gaussian": kernel.GaussianKernel,
            "hyperbolic": kernel.HyperbolicKernel,
            "precomputed": kernel.PrecomputedKernel
        }
        if self.k not in kernel_map:
            raise ValueError(f"parameter `k` must be one of {', '.join(kernel_map.keys())}. Given: {self.k}")
        self._kernel = kernel_map[self.k](**(self.k_params or {}))

        solver_map = {
            "gurobi": GurobiSolver,
            "tensorflow": TensorFlowSolver
        }
        if self.solver not in solver_map:
            raise ValueError(f"parameter `solver` must be one of {', '.join(solver_map.keys())}. Given: {self.solver}")
        self._solver = solver_map[self.solver](**(self.solver_params or {}))

        if warm_start:
            check_is_fitted(self)
            if self.chis_ is None:
                raise NotFittedError("chis variable is set to None")
            self._solver.initial_values = self.chis_

        if type(self._kernel) is kernel.PrecomputedKernel:
            idx = X.flatten()
            self._gram = self._kernel.kernel_computations[idx][:, idx]
        else:
            self._gram = np.array([self._kernel.compute(x1, X) for x1 in X])

        self.chis_ = self._solver.solve(X, y, self.c, self._gram)

        self.fixed_term_ = np.array(self.chis_).dot(self._gram.dot(self.chis_))

        chi_SV_index = [i for i, (chi, mu) in enumerate(zip(self.chis_, y))
                        if -self.c * (1 - mu) < chi < self.c * mu]

        if len(chi_SV_index) == 0:
            logger.warning("No support vectors found")
            if len(X) == 1:
                self.squared_radius_ = 1e-6
            else:
                try:
                    self.squared_radius_ = np.mean(self.anomaly_score(X))
                except:
                    self.squared_radius_ = 1.0
            return self

        # if len(chi_SV_index) == 0:
        #     self.chis_ = None
        #     logger.warning("No support vectors found")
        #     return self

        chi_sq_radius = list(self.anomaly_score(X[chi_SV_index]))

        self.squared_radius_ = np.mean(chi_sq_radius)

        return self

    def anomaly_score(self, X: ArrayLike) -> NDArray[np.float64]:
        """Compute the distance associated to the samples. Higher is more anomalous

        :param X: data points
        :type X: ArrayLike
        :return: array with the squared distances of the data points
        :rtype: NDArray[np.float64]
        """

        check_is_fitted(self)
        X = check_array(X)
        t1 = self._kernel.compute(X, X)
        t2 = np.array([self._kernel.compute(x_i, X)
                        for x_i in self._original_X]).transpose().dot(self.chis_)
        ret = t1 -2 * t2 + self.fixed_term_
        return ret

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self)
        X = check_array(X)
        return -self.anomaly_score(X)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Decision function derived from the score_samples. Lower is more anomalous

        :param X: data points
        :type X: ArrayLike
        :return: array of values: more negative for more anomalous, more positive for more normal points
        :rtype: NDArray[np.float64]
        """

        check_is_fitted(self)
        X = check_array(X)
        return self.squared_radius_ - self.anomaly_score(X)

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """Prediction function. -1 for anomalies, +1 for normal points

        :param X: data points
        :type X: ArrayLike
        :return: array of predictions
        :rtype: NDArray[np.int_]
        """
        check_is_fitted(self)
        X = check_array(X)

        predictions = [1 if value>=0 else 0 for value in self.decision_function(X)]
        return np.asarray(predictions)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute the score obtained as the inverted MSE between predicted
        values and labels.

        1.0 for best score, and 0.0 for worst score.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Labels containing the *gold standard* values
          for the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :returns: `float` -- inverted MSE between the predictions for the
          elements in `X` w.r.t. the labels in `y`.
        """

        check_is_fitted(self)
        X, y = check_X_y(X, y)
        return 1.0 - np.mean((self.predict(X) - y) ** 2)