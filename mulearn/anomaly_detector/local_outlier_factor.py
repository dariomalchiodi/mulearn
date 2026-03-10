""" This module implements a Local Outlier Factor model for anomaly detection
"""

from typing import Any, Callable, Literal, Self
from numpy.typing import NDArray, ArrayLike

import math
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LocalOutlierFactor:
    """ The class implementing the Local Outlier Factor
    """

    def __init__(self) -> None:
        raise NotImplementedError("Not implemented")

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        raise NotImplementedError("Not implemented")

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        raise NotImplementedError("Not implemented")

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        raise NotImplementedError("Not implemented")

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        raise NotImplementedError("Not implemented")