import warnings
from typing import Any, Callable, Literal, Self
from numpy.typing import NDArray, ArrayLike

import math
import numpy as np
from functools import partial
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

DEFAULT_MAX_SAMPLES = 256           # default number of samples for each tree, taken from sklearn's IsolationForest max_samples
DEFAULT_OFFSET = - 0.5              # used in decision_function, related to contamination value, used when contamination = auto
MEMBERSHIP_ANOMALY_THRESHOLD = 0.5   # used for decision_function value computation. A memberhsip value less or equal than this, is considered anomaly

class TreeNode:
    def __init__(self,
                 left: Self | None = None,
                 right: Self | None = None,
                 value: Any = None,
                 feature_index: int | None = None,
                 depth: int | None = None
                 ) -> None:
        self.left = left
        self.right = right
        self.value = value
        self.feature_index = feature_index
        self.depth = depth

    def __repr__(self) -> str:
        return f"TreeNode(feature_index: {self.feature_index}, value: {self.value}, depth: {self.depth})"

    def get_value(self) -> Any:
        return self.value

    def get_feat_idx(self) -> int | None:
        return self.feature_index

    def get_depth(self) -> int:
        return self.depth or 0

    def get_child(self, side: Literal["left", "right"]) -> Self | None:
        if side == "left":
            return self.left
        else:
            return self.right

    def print_subTree(self, side: str = "", depth: int = 0) -> None:
        print(f"Depth {depth}:", "   " * depth, side, str(self.get_feat_idx()), str(self.get_value()))
        if self.left is not None:
            self.left.print_subTree("l", depth + 1)
        if self.right is not None:
            self.right.print_subTree("r", depth + 1)

class IsolationTree:
    def __init__(self,
                 max_depth: int = 20,
                 random_state: int | None = None
                 ) -> None:
        self.max_depth = max_depth
        self.random_state = random_state

    def _check_fitted(self) -> bool:
        """Checks if the tree has been fit yet

        :return: True if fitted, False otherwise.
        :rtype: bool
        """
        if getattr(self, "num_nodes_", 0) == 1:
            warnings.warn("Tree has only the root node, try feeding more data", UserWarning)
        return (
            getattr(self, "root_", None) is not None and
            getattr(self, "num_nodes_", 0) > 0
        )

    def _get_og_feature_index(self, sampled_feature_index: int) -> int:
        """Return the feature index of the original dataset, given the feature index from the sampled features

        :param sampled_feature_index: The feature index after the sampling on the features for the current tree
        :type sampled_feature_index: int
        :raises RuntimeError: If the original feature index array has not been set yet
        :return: The original feature index for the given sampled index
        :rtype: int
        """
        if not hasattr(self, "og_features_indices_") or self.og_features_indices_ is None:
            raise RuntimeError("Original feature index not set yet")
        return self.og_features_indices_[sampled_feature_index]

    def _build_splitting_function(self, X: NDArray, y: NDArray, verbose: int = 0) -> Callable[[NDArray[np.float64 | np.int_] | float | int], NDArray[np.float64]]:
        """Is the inverse function of a cumulative distribution function.

        :param X: The array of values in feature space
        :type X: NDArray
        :param y: The array of membership values, in interval [0,1]
        :type y: NDArray
        :param verbose: To print more informations, either 0, 1 or 2, defaults         if verbose>=2:
            print("normalized areas values:", normalized_areas)to 0
        :type verbose: int, optional
        :return: The inverse callable function, that accepts the y as paramter, and returns the corresponding feature value. And also a boolean, if True then the cumulative function is constant, and the inverse function will always return the maximum feature value.
        :rtype: tuple[Callable[[NDArray[np.float64 | np.int_] | float | int], NDArray[np.float64]], bool]
        """
        # Avoiding duplicates in couples (X,y):
        zipped_X_y = list(zip(X, y))
        zipped_X_y = np.unique(zipped_X_y, axis=0)
        X = zipped_X_y[:,0]
        y = zipped_X_y[:,1]

        # Calculating the intervals in feature space:
        midpoints = (X[:-1] + X[1:]) / 2
        starts = np.concatenate(([X[0]], midpoints))
        ends = np.concatenate((midpoints, [X[-1]]))
        x_neighborhoods = np.stack((starts, ends), axis=1) # [[start,end], ...]
        complemented_y = 1 - y               # ones complement for y
        # Calculating areas:
        bases = x_neighborhoods[:, 1] - x_neighborhoods[:, 0]
        areas = bases * complemented_y       # base * height

        if areas.sum() == 0:
            areas = np.ones(len(areas))
            if verbose >= 2:
                print("The cumulative function obtained is constant. Setting all areas to same value")

        normalized_areas = areas / areas.sum()    # normalize areas
        # Building the cumulative:
        # y0 = 0
        # y2 = 0 + normalized_area[0] + normalized_area[1]
        # yn = 1
        cumulative_areas = 0
        cumulative_xy = []      # [[x1,y1], .. [xn,yn]]
        for i,j in zip(x_neighborhoods, normalized_areas):
            cumulative_xy.append([i[0], cumulative_areas])
            cumulative_areas += j
            cumulative_xy.append([i[1], cumulative_areas])
        cumulative_xy[-1][1] = 1        # hardcode to be fully '1' (because it can be ~.999..)
        cumulative_xy = np.asarray(cumulative_xy)
        cumulative_xy = np.unique(cumulative_xy, axis=0)    # delete same pairs
        # Calculating the inverse of the cumulative function:
        # So that x belongs to [0.0, 1.0] and y belongs to original feature space (X)
        inv_big_f = partial(np.interp, xp=cumulative_xy[:,1], fp=cumulative_xy[:,0])

        if verbose >= 2:
            import plotly.graph_objects as go
            x_axis = np.linspace(0,1,1000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=inv_big_f(x_axis),
                mode="lines"
                )
            )
            fig.add_trace(go.Scatter(x=cumulative_xy[:,1],y=cumulative_xy[:,0],mode="markers",name="probability change"))
            fig.show()

        return inv_big_f

    def _build_subTree(self, start: int, end: int, depth: int = 0, verbose: int = 0) -> TreeNode | None:
        """Recursively tree fitting function

        :param start: The starting index of the X array
        :type start: int
        :param end: The ending index (non inclusive) of the X array
        :type end: int
        :param depth: Current tree depth. 0 for the root node, defaults to 0
        :type depth: int, optional
        :param verbose: If to print informations, values can either be 0, 1 or 2, defaults to 0
        :type verbose: int, optional
        :return: A TreeNode
        :rtype: TreeNode | None
        """
        # Increase tree depth and nodes number
        if verbose == 2:
            print("Indices interval start-end:",start,end)
        self.depth_ = max(self.depth_, depth)
        self.num_nodes_ += 1

        # Sampling the feature index
        split_feature_index = self.random_number_generator_.integers(self._X.shape[1])

        # Ordering based on the selected feature:
        # 'ord_feat_X' will be the current array where to operate, with start and end set, just one feature, and ordered
        # 'ord_y' will be the labels, with start and end set, with same ordering of 'ord_feat_X'
        sorting_indices = self._X.argsort(axis=0)[:,split_feature_index]
        ord_feat_X = self._X[sorting_indices, split_feature_index]
        ord_y = self._y[sorting_indices]

        if verbose == 2:
            print("Split feature index:", self._get_og_feature_index(split_feature_index))

        # Returning if max_depth reached or just one node left
        if depth >= self.max_depth or end - start <= 1:
            if verbose == 2:
                if depth >= self.max_depth:
                    print("Max depth reached")
                if end>start:
                    print("Single node left, with data=", ord_feat_X[start])
                else:
                    print("Returning node because end<=start")
            return TreeNode(feature_index=self._get_og_feature_index(split_feature_index), value=ord_feat_X[start] if end > start else None, depth=depth)

        min_value = ord_feat_X[start]
        max_value = ord_feat_X[end-1]
        if verbose == 2:
            print("Values interval:",min_value,max_value)

        if min_value == max_value:
            if verbose == 2:
                print("Returning node because minval == maxval, with data=",min_value)
            return TreeNode(feature_index=self._get_og_feature_index(split_feature_index), value=min_value, depth=depth)

        # Build the splitting function on the interval
        splitting_func = self._build_splitting_function(ord_feat_X[start:end], ord_y[start:end], verbose=verbose)
        # Get split feature value and index
        split_value = splitting_func(self.random_number_generator_.random())
        split_index = np.searchsorted(ord_feat_X[start:end], split_value, side='right') + start
        if verbose == 2:
            print("Split value:", split_value, "split index:", split_index)

        # Recursion on children:
        if verbose == 2:
            print("continuing left" if split_index>start else "no nodes left to the left")
        left_child = self._build_subTree(start, split_index, depth + 1, verbose=verbose) if split_index > start else None
        if verbose == 2:
            print("continuing right" if split_index<end else "no nodes left to the right")
        right_child = self._build_subTree(split_index, end, depth + 1, verbose=verbose) if split_index < end else None

        return TreeNode(feature_index=self._get_og_feature_index(split_feature_index), value=split_value, left=left_child, right=right_child, depth=depth)

    def fit(self, X: ArrayLike, y: ArrayLike, og_features_indices: NDArray[np.int_] | None = None, verbose: int = 0) -> Self:
        """Fits the tree

        :param X: Array of elements to fit in the tree
        :type X: ArrayLike
        :param y: Array of membership values, of same size of X
        :type y: ArrayLike
        :param og_features_indices: Array of the sampled feature indices relative to the original/full array. If not set, then is assumed to use all of the features, defaults to None
        :type og_features_indices: NDArray[np.int_]
        :param verbose: If to print more information, can either be 0, 1 or 2, defaults to 0
        :type verbose: int, optional
        :return: The fitted IsolationTree
        :rtype: Self
        """
        self._X = X.copy()
        self._y = y.copy()
        if og_features_indices is None:
            self.og_features_indices_ = np.array(range(X.shape[1]))
        else:
            self.og_features_indices_ = og_features_indices
        self.depth_ = 0
        self.num_nodes_ = 0
        self.random_number_generator_ = np.random.default_rng(self.random_state)
        self.root_ = self._build_subTree(0, len(self._X), verbose=verbose)
        return self

    def get_node_depth(self, target_data) -> int:
        """Returns the depth of the value of the parameter.
        Starting from the root, for each node of the tree we continue left is the `target_data` value is less or equal than the node, else we continue right. Until we find the node or we reach a leaf.

        :param target_data: The value to search for
        :type target_data: Any
        :raises TypeError: If the paramter `target_data` is multidimensional
        :raises RuntimeError: If the tree is not fitted
        :return: The depth of the found `target_data` value
        :rtype: int
        """
        target_data = np.asarray(target_data)
        if target_data.ndim > 1:
            raise TypeError(f"target_data must have dimension=1, but it has dimension={target_data.ndim}")
        if not self._check_fitted():
            raise RuntimeError("Tree not fitted")
        cur_node = self.root_
        while cur_node is not None:
            cur_feat_idx = cur_node.get_feat_idx()
            cur_data = cur_node.get_value()
            cur_depth = cur_node.get_depth()
            if target_data[cur_feat_idx] > cur_data:
                cur_node = cur_node.get_child("right")
            else:
                cur_node = cur_node.get_child("left")
        return cur_depth

    def get_max_depth(self) -> int:
        """Returns the max depth of the tree

        :return: The max depth
        :rtype: int
        """
        return self.depth_

    def get_num_nodes(self) -> int:
        """Returns the number of nodes in the tree, root inclusive.

        :return: Number of nodes
        :rtype: int
        """
        return self.num_nodes_

    def print_tree(self) -> None:
        """Prints the tree to stdout
        """
        if self.root_ is None:
            print("Empty tree")
            return
        self.root_.print_subTree()

class SupervisedIsolationForest(BaseEstimator):
    def __init__(self,
                 n_estimators: int = 20,
                 max_samples: Literal["auto"] | int | float = "auto",
                 contamination: Literal["auto"] | float = "auto",
                 max_features: int | float = 1.0,
                 max_depth: int | None = None,
                 random_state: int | None = None
                 ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state

    def __sklearn_is_fitted__(self) -> bool:
        """Checks if the tree has been fit

        This method is called by `check_is_fitted(self)`

        :return: If the tree is fitted or not
        :rtype: bool
        """
        return (
            getattr(self, "forest_", None) is not None and
            getattr(self, "c_of_n_", None) is not None and
            len(self.forest_) > 0
        )

    def fit(self, X: ArrayLike, y: ArrayLike, verbose: int = 0) -> Self:
        """Fits the isolation forest to the given X array of data

        :param X: The array of data to fit
        :type X: ArrayLike
        :param y: The array of membership values, in interval [0,1] with same length of X
        :type y: ArrayLike
        :param verbose: If to print more informations, can be 0 or 1, defaults to 0
        :type verbose: int, optional
        :raises ValueError: Value checks on the given constructor's paramters for the forest
        :raises TypeError: Type checks on the given constructor's paramters for the forest
        :return: The fitted forest
        :rtype: Self
        """

        X, y = check_X_y(X, y)

        for e in y:
            if e < 0 or e > 1:
                raise ValueError("`y` values must belong to [0, 1]")

        self._X, self._y = X.copy(), y.copy()

        dataset_size = len(self._X)
        self.n_features_in_ = self._X.shape[1]

        # 'max_samples' parameter check and setting:
        if self.max_samples == "auto":
            max_samples = DEFAULT_MAX_SAMPLES
        elif isinstance(self.max_samples, (float, np.floating)):
            maxs = self.max_samples
            if maxs > 1. or maxs < .0:
                raise ValueError(f"if max_samples is a float, it should be between 0.0 and 1.0. Given value: {maxs}")
            max_samples = int(dataset_size * maxs) or 1
        elif isinstance(self.max_samples, (int, np.integer)):
            if self.max_samples <= 0:
                raise ValueError(f"if max_samples is a int, it must be greater than 0. Given value: {self.max_samples}")
            max_samples = self.max_samples
        else:
            raise TypeError(f"max_samples parameter should be either a 'int', 'float' or the string value: \"auto\", given: {type(self.max_samples)}")
        max_samples = min(max_samples, dataset_size)

        # 'contamination' parameter check:
        if isinstance(self.contamination, (float, np.floating)):
            if self.contamination >= 1. or self.contamination < .0:
                raise ValueError(f"if contamination is a float, it should be between 0.0 and 1.0 excluded. Given value: {self.contamination}")
            self.anomalies_number_ = int(len(self._X) * self.contamination)
        elif isinstance(self.contamination, (str, np.str_)):
            if self.contamination != "auto":
                raise ValueError(f"if contamination is a string, it should be: \"auto\". Given value: {self.contamination}")
            # Default behaviour of sklearn IsolationForest:
            # self.offset_ = DEFAULT_OFFSET     # DEFAULT_OFFSET = - 0.5
            # A different, custom, implementation to infer the contamination value, using membership informations:
            self.anomalies_number_ = np.sum(self._y <= MEMBERSHIP_ANOMALY_THRESHOLD)        # MEMBERSHIP... = 0.5
        else:
            raise TypeError(f"contamination parameter should be either a 'float' or the string value: \"auto\", given: {type(self.contamination)}")

        # 'max_features' parameter check and setting:
        if isinstance(self.max_features, (int, np.integer)):
            max_features = self.max_features
        elif isinstance(self.max_features, (float, np.floating)):
            maxf = self.max_features
            if maxf > 1. or maxf < .0:
                raise ValueError(f"if max_features is a float, it should be between 0.0 and 1.0. Given value: {maxf}")
            max_features = int(self.n_features_in_ * maxf) or 1
        else:
            raise TypeError(f"max_features parameter should be either a 'int' or 'float', given: {type(self.max_features)}")
        max_features = min(max_features, self.n_features_in_)

        # 'max_depth' parameter check and setting:
        if self.max_depth is None:
            # According to sklearn Isolation Forest:
            max_depth = math.ceil(math.log2(dataset_size))
        elif isinstance(self.max_depth, (int, np.integer)):
            if self.max_depth < 1:
                raise ValueError(f"max_depth parameter cannot be less than 1. Given value: {self.max_depth}")
            else:
                max_depth = self.max_depth
        else:
            raise TypeError(f"max_depth parameter should be either 'None' or 'int', given: {type(self.max_depth)}")

        self.random_number_generator_ = np.random.default_rng(self.random_state)

        if verbose == 2:
            import time
            start = time.time()
        if verbose >= 1:
            from tqdm import tqdm
            v = [_ for _ in range(self.n_estimators)]
            trees_iter = tqdm(v, desc=f'Fitting {self.n_estimators} trees')
        else:
            trees_iter = range(self.n_estimators)

        self.forest_ = []

        # Fitting trees:
        for _ in trees_iter:
            # Subsampling from (X,y):
            samples_indeces = np.sort(self.random_number_generator_.permutation(dataset_size)[:max_samples])
            sampled_X = X[samples_indeces]
            sampled_y = y[samples_indeces]

            # Feature setting for the samples:
            sampled_features = np.sort(self.random_number_generator_.choice(self.n_features_in_, size=max_features, replace=False))
            sampled_X = sampled_X[:,sampled_features]

            # Training a tree:
            tree = IsolationTree(max_depth=max_depth, random_state=self.random_state)
            tree.fit(sampled_X, sampled_y, sampled_features, verbose=verbose-1)
            self.forest_.append(tree)

        # Saving 'c(n)' term to later calculate the score for any data point:
        # 'dataset_size' is 'n'
        if dataset_size == 1:
            self.c_of_n_ = 1
        else:
            self.c_of_n_ = 2 * (math.log(dataset_size-1) + np.euler_gamma) - (2*(dataset_size-1)/dataset_size)

        # Calculating the offset value that gives us the decided percentage of anomalies among the scores:
        sorted_scores = np.sort(self.score_samples(self._X))
        if self.anomalies_number_ == dataset_size:  # if all anomalies
            self.offset_ = sorted_scores[dataset_size-1]
        else:
            self.offset_ = (sorted_scores[self.anomalies_number_] + sorted_scores[self.anomalies_number_ - 1]) / 2

        if verbose == 2:
            print(time.time()-start,"seconds")

        return self

    def anomaly_score(self, X: ArrayLike, verbose: int = 0) -> NDArray[np.float64]:
        """Indicates how much a point is anomalous. Is calculated from the IsolationForest formula of the original paper.

        :param X: The samples to compute the scores for
        :type X: ArrayLike
        :return: The array of scores with values in interval [0,1]
        :rtype: NDArray[np.float64]
        """
        check_is_fitted(self)
        X = check_array(X)

        if verbose >= 1 :
            import time
            start = time.time()
        res = []
        for sample in X:
            avg_depth = 0
            for tree in self.forest_:
                avg_depth += tree.get_node_depth(sample)
            avg_depth = avg_depth/self.n_estimators
            # Original sklearn's IsolationForest anomaly score formula:
            anomaly_score = 2 ** (- avg_depth / self.c_of_n_)
            res.append(anomaly_score)
        if verbose >= 1:
            print(time.time()-start,"seconds")
        return np.asarray(res)

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        """Gives a score for each sample in X. Higher is more inlier, lower is more outlier. Is the opposite of the anomaly score.

        :param X: The samples to compute the scores for
        :type X: ArrayLike
        :return: The array of scores with values in interval [0,1]
        :rtype: NDArray[np.float64]
        """
        check_is_fitted(self)
        X = check_array(X)
        # As stated in sklearn 'score_samples is the opposite of the anomaly score':
        return -self.anomaly_score(X)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Shifts the score_samples to be between -1 and 1. The lower the more abnormal

        :param X: The samples to calculate the values for
        :type X: ArrayLike
        :return: Array of floats, same length of X, in interval [-1,1]. Dimension 1
        :rtype: NDArray[np.float64]
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.score_samples(X) - self.offset_

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """Array of 0 if anomalous, +1 if non-anomaly

        Instead of the normal sklearn behaviour of -1,+1.
        Since we are getting label values in fit [0,1]
        """
        check_is_fitted(self)
        X = check_array(X)
        return np.where(self.decision_function(X) > 0, 1, 0)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        check_is_fitted(self)
        X, y = check_X_y(X, y)
        return 1.0 - np.mean((self.predict(X) - y) ** 2)