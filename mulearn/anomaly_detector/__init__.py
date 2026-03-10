""" This module defines the base anomaly detector used in mulearn.
"""

import logging
from typing import Any, Callable, Literal, Self

import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from mulearn.anomaly_detector.isolation_forest import SupervisedIsolationForest
from mulearn.anomaly_detector.support_vector_machine import SupportVectorMachine


logger = logging.getLogger(__name__)


class AnomalyDetector(BaseEstimator):
    """ Base class for anomaly detectors.

    The base class for anomaly detectors is AnomalyDetector
    """

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs) -> Self:
        """ Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space
        :type X: ArrayLike
        :param y: Labels for the vectors in `X`, defaults to None
        :type y: ArrayLike | None, optional
        :return: The trained model
        :rtype: Self
        """

        raise NotImplementedError(
            "The base class does not implement the `fit` method")

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Returns the array of predictions for each data point in X.
        """
        raise NotImplementedError(
            "The base class does not implement the `predict` method")

    def _check_X_y(self, X: ArrayLike, y: ArrayLike | None) -> tuple[ArrayLike, ArrayLike|None]:
        """Function to do the checks on the arrays of X and y, mainly if they
        are same length and if y is not none we check that its values are between 0 and 1

        :param X: Vectors in data space
        :type X: ArrayLike
        :param y: Labels for the vectors in `X`, defaults to `None`
        :type y: ArrayLike | None
        :raises ValueError: If the values of `y` are not in [0, 1]
        :return: X, y unchanged
        :rtype: tuple[ArrayLike, ArrayLike|None]
        """

        X = check_array(X)

        if y is not None:
            X, y = check_X_y(X, y)
            for e in y:
                if e < 0 or e > 1:
                    raise ValueError("`y` values should belong to [0, 1]")

        return X, y


class SVMAnomalyDetector(AnomalyDetector):
    """ Implementation of Support Vector Machine for AnomalyDetector.

    :param c: Trade-off constant, defaults to 1
    :type c: float, optional
    :param k: Kernel function, defaults to "gaussian"
    :type k: Literal[&quot;linear&quot;, &quot;polynomial&quot;, &quot;homogeneous&quot;, &quot;gaussian&quot;, &quot;hyperbolic&quot;, &quot;precomputed&quot;], optional
    :param k_params: Additional parameters for the kernel, defaults to None
    :type k_params: dict[str, Any] | None, optional
    :param solver: Solver to be used to obtain the optimization problem solution, defaults to "gurobi"
    :type solver: Literal[&quot;gurobi&quot;, &quot;tensorflow&quot;], optional
    :param solver_params: Additional parameters for the solver, defaults to None
    :type solver_params: dict[str, Any] | None, optional

    :param n_features_in_: number of features in X
    :type n_features_in_: int
    :param score_05_: estimated hypersphere radius
    :type score_05_: float
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

    def __repr__(self, **kwargs) -> str:
        return f"SVMAnomalyDetector(c={self.c}, k={self.k}, " \
               f"solver={self.solver})"

    def __eq__(self, other):
        """Check equality w.r.t. other objects."""
        return (
            type(self) == type(other) and
            self.get_params() == other.get_params()
        )

    def __sklearn_is_fitted__(self) -> bool:
        """This method is called by `check_is_fitted(self)`

        :return: Whether the instance has been fit or not
        :rtype: bool
        """
        return (
            getattr(self, "_estimator", None) is not None
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, warm_start: bool = False) -> Self:
        """Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space
        :type X: ArrayLike
        :param y: Labels for th vectors in `X`, defaults to None
        :type y: ArrayLike | None, optional
        :param warm_start: flag triggerint the non reinitialization of
        independent variables of the optimization problem, defaults to False
        :type warm_start: bool, optional
        :return: The trained model
        :rtype: Self
        """

        X, y = super()._check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self._estimator = SupportVectorMachine(
            c=self.c,
            k=self.k,
            k_params=self.k_params,
            solver=self.solver,
            solver_params=self.solver_params
        )
        self._estimator = self._estimator.fit(X, y, warm_start=warm_start)
        self.score_05_ = self._estimator.squared_radius_

        train_scores = self._estimator.score_samples(X)
        self._min_score = np.min(train_scores)
        self._max_score = np.max(train_scores)

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Returns the predictions for the given points, in range [0, 1]

        :param X: Vectors in data space
        :type X: ArrayLike
        :return: The predictions for each value in `X`, in range [0, 1]
        :rtype: NDArray[np.float64]
        """

        X = check_array(X)
        check_is_fitted(self)

        membership_scores = self._estimator.score_samples(X)

        if self._min_score == self._max_score:
            predictions = np.full_like(membership_scores, self._min_score)
        else:
            predictions = (membership_scores - self._min_score) / (self._max_score - self._min_score)

        return predictions

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute the score of the predictions.

        Score is obtained as the opposite of MSE between predicted
        values and labels.

        :param X: Vectors in data space
        :type X: ArrayLike
        :param y: Labels containing the *gold standard* values for the vectors in `X`
        :type y: ArrayLike
        :returns: Opposite of MSE between the predictions for the elements in `X` w.r.t. the labels in `y`.
        :rtype: float
        """

        check_is_fitted(self)
        X, y = check_X_y(X, y)
        return 1.0 - np.mean((self.predict(X) - y) ** 2)


class IFAnomalyDetector(AnomalyDetector):
    """ Implementation of Isolation Forest for AnomalyDetector

    :param n_trees: number of trees in the forest, defaults to 100.
    :type n_trees: int
    :param max_samples: number of samples for each tree, defaults to `auto`.
    :type max_samples: : `"auto"` or `int` or `float`
    :param contamination: expeted ratio of outliers, defaults to `auto`.
    :type contamination: `"auto"` or `float`
    :param max_features: number of features to use when splitting for each tree. If less than 1.0 then all features will be used, defaults to `1.0`.
    :type max_features: `int` or `float`
    :param bootstrap: if to reuse samples for trees. If False samples are unique per tree, defaults to `False`.
    :type bootstrap: `bool`
    :param max_depth: max depth for trees of the supervised algorithm
    :type max_depth: int
    :param n_jobs: number of parallel jobs/threads. If -1 uses all cores, defaults to `None`.
    :type n_jobs: int
    :param random_state: set a random state for results reproducibility, defaults to `None`.
    :type random_state: `int` or :class:`RandomState` or `None`

    :param n_features_in_: Number of features in X
    :type n_features_in_: int
    :param score_05_: The `offset` decision boundary of the forest
    :type score_05_: float
    """

    def __init__(self,
                 n_trees: int | None = None,
                 max_samples: Literal["auto"] | int | float = "auto",
                 contamination: Literal["auto"] | float = "auto",
                 max_features: int | float = 1.0,
                 bootstrap: bool = False,
                 max_depth: int | None = None,
                 n_jobs: int | None = -1,
                 random_state: int | None = None
                 ) -> None:
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state

    def __repr__(self, **kwargs):
        return f"IFAnomalyDetector(trees={self.n_trees}, max_samples={self.max_samples}, max_features={self.max_features})"

    def __eq__(self, other):
        return (
            type(self) == type(other) and
            isinstance(self.random_state, int) and
            isinstance(other.random_state, int) and
            self.get_params() == other.get_params()
        )

    def __sklearn_is_fitted__(self) -> bool:
        """This method is called by `check_is_fitted(self)`

        :return: Whether the instance has been fit or not
        :rtype: bool
        """
        return (
            getattr(self, "_estimator", None) is not None
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, warm_start: bool = False, verbose: int = 0) -> Self:
        """Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space.
        :type X: ArrayLike
        :param y: Labels for the vectors in `X`, defaults to `None`.
        :type y: ArrayLike | None
        :param warm_start: flag triggering the non reinitialization of
          independent variables of the optimization problem, defaults to
          `None`.
        :type warm_start: bool
        :returns: The trained model.
        :rtype: Self
        """

        X, y = super()._check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if y is None:
            trees = self.n_trees or 100
            self._estimator = IsolationForest(
                n_estimators=trees,
                max_samples=self.max_samples,
                contamination=self.contamination,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                warm_start=warm_start
            )
            self._estimator.fit(X)
            self.score_05_ = self._estimator.offset_
        else:
            trees = self.n_trees or 20
            self._estimator = SupervisedIsolationForest(
                n_estimators=trees,
                max_samples=self.max_samples,
                contamination=self.contamination,
                max_features=self.max_features,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self._estimator.fit(X, y, verbose=verbose)
            self.score_05_ = -self._estimator.offset_

        train_scores = self._estimator.score_samples(X)
        self._min_score = np.min(train_scores)
        self._max_score = np.max(train_scores)

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Returns the predictions for the given points, in range [0, 1]

        :param X: Vectors in data space
        :type X: ArrayLike
        :return: The predictions for each value in `X`, in range [0, 1]
        :rtype: NDArray[np.float64]
        """

        X = check_array(X)
        check_is_fitted(self)

        membership_scores = self._estimator.score_samples(X)

        if self._min_score == self._max_score:
            predictions = np.full_like(membership_scores, self._min_score)
        else:
            predictions = (membership_scores - self._min_score) / (self._max_score - self._min_score)

        return predictions

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute the score of the predictions.

        Score is obtained as the opposite of MSE between predicted
        values and labels.

        :param X: Vectors in data space
        :type X: ArrayLike
        :param y: Labels containing the *gold standard* values for the vectors in `X`
        :type y: ArrayLike
        :returns: Opposite of MSE between the predictions for the elements in `X` w.r.t. the labels in `y`.
        :rtype: float
        """

        check_is_fitted(self)
        X, y = check_X_y(X, y)
        return 1.0 - np.mean((self.predict(X) - y) ** 2)


class LOFAnomalyDetector(AnomalyDetector):
    """ Implementation of Local Outlier Factor AnomalyDetector

    :param n_neighbors: the number of neighbors to consider when calculating local density, defaults to `20`.
    :type n_neighbors: int
    :param algorithm: algorithm used to compute the nearest neighbors, defaults to `auto`.
    :type algorithm: `Literal["auto", "ball_tree", "kd_tree", "brute"]`
    :param leaf_size: controls leaf size in tree-based algorithms BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem, defaults to 30.
    :type leaf_size: `int`
    :param metric: the distance metric to use, defaults to `"minkowski"` which results in the standard Euclidean distance when p = 2
    :type metric: `str` or `callable`
    :param p: parameter for the Minkowski metric from sklearn.metrics.pairwise_distances. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used, defautls to 2.
    :type p: `float`
    :param metric_params: additional keyword arguments for the metric function, defaults to `None`
    :type metric_params: `dict`
    :param contamination: expeted ratio of outliers. When fitting this is used to define the threshold on the scores of the samples, defaults to `auto`.
    :type contamination: `"auto"` or `float`
    :param novelty: by default, LocalOutlierFactor is only meant to be used for outlier detection (novelty=False). Set novelty to True if you want to use LocalOutlierFactor for novelty detection. In this case be aware that you should only use predict, decision_function and score_samples on new unseen data and not on the training set; and note that the results obtained this way may differ from the standard LOF results, defaults to `True`
    :type novelty: `bool`
    :param n_jobs: number of parallel jobs/threads. If -1 uses all cores, defaults to `None`.
    :type n_jobs: `int`

    :param n_features_in_: Number of features in X
    :type n_features_in_: int
    :param score_05_: The `offset` decision boundary of the forest
    :type score_05_: float
    """

    def __init__(self,
                 n_neighbors: int = 20,
                 algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
                 leaf_size: int = 30,
                 metric: str | Callable = "minkowski",
                 p: int = 2,
                 metric_params: dict | None = None,
                 contamination: Literal["auto"] | float = "auto",
                 novelty: bool = True,
                 n_jobs: int | None = -1
                 ) -> None:
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs

    def __repr__(self, **kwargs):
        return f"LOFFuzzyInductor(n_neighbors={self.n_neighbors}, algorithm={self.algorithm}, leaf_size={self.leaf_size}, metric={self.metric}, p={self.p}, metric_params={self.metric_params}, contamination={self.contamination}, novelty={self.novelty}, n_jobs={self.n_jobs})"

    def __eq__(self, other):
        return (
            type(self) == type(other) and
            self.get_params() == other.get_params()
        )

    def __sklearn_is_fitted__(self) -> bool:
        """This method is called by `check_is_fitted(self)`

        :return: Whether the instance has been fit or not
        :rtype: bool
        """
        return (
            getattr(self, "_estimator", None) is not None
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Train the anomaly detector to be able to get the anomaly score for each data point

        :param X: Vectors in data space.
        :type X: ArrayLike
        :param y: Membership for the vectors in `X`, defaults to `None`.
        :type y: ArrayLike | None
        :raises NotImplementedError: The supervised version not implmeneted yet
        :returns: The trained model.
        :rtype: Self
        """

        X, y = super()._check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if y is None:
            self._estimator = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                metric=self.metric,
                p=self.p,
                metric_params=self.metric_params,
                contamination=self.contamination,
                novelty=self.novelty,
                n_jobs=self.n_jobs
            )
            self._estimator.fit(X)
            self.score_05_ = self._estimator.offset_
        else:
            raise NotImplementedError(
                "Supervised Local Outlier Factor not yet implemented")

        train_scores = self._estimator.score_samples(X)
        self._min_score = np.min(train_scores)
        self._max_score = np.max(train_scores)

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Returns the predictions for the given points, in range [0, 1]

        :param X: Vectors in data space
        :type X: ArrayLike
        :return: The predictions for each value in `X`, in range [0, 1]
        :rtype: NDArray[np.float64]
        """

        X = check_array(X)
        check_is_fitted(self)

        membership_scores = self._estimator.score_samples(X)

        if self._min_score == self._max_score:
            predictions = np.full_like(membership_scores, self._min_score)
        else:
            predictions = (membership_scores - self._min_score) / (self._max_score - self._min_score)

        return predictions

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute the score of the predictions.

        Score is obtained as the opposite of MSE between predicted
        values and labels.

        :param X: Vectors in data space
        :type X: ArrayLike
        :param y: Labels containing the *gold standard* values for the vectors in `X`
        :type y: ArrayLike
        :returns: Opposite of MSE between the predictions for the elements in `X` w.r.t. the labels in `y`.
        :rtype: float
        """

        check_is_fitted(self)
        X, y = check_X_y(X, y)
        return 1.0 - np.mean((self.predict(X) - y) ** 2)