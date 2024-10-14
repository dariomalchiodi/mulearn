
"""Mulearn distributions.

This module implements the kernel and fuzzifier distributions used in
randomized model selection.
"""

from scipy.stats import uniform
from scipy.stats import rv_continuous

from mulearn.kernel import GaussianKernel
from mulearn.fuzzifier import ExponentialFuzzifier

class GaussianKernelDistribution(rv_continuous):
    """Uniform distribution for gaussian kernels."""

    def __init__(self, low=0, high=1):
        """Build an object of type `GaussianKernelDistribution`.

        :param low: Lower bound of the interval defining the support of the
          uniform distribution associated to the parameter $\sigma$ of a
          :class:`mulearn.kernel.GaussianKernel`, defaults to `0`.
        :type low: `float`
        :param high: Upper bound of the interval defining the support of the
          uniform distribution associated to the parameter $\sigma$ of a
          :class:`mulearn.kernel.GaussianKernel`, defaults to `1`.
        :type high: `float`
        :raises: ValueError if the arguments `low` and `high` do not define
          an interval (that is, `low` is not lower than `high`).
        """
        super().__init__()
        if high <= low:
            raise ValueError(f"the provided upper extreme {high} is lower or"
                             f"equal to the lower one {low}.")
        self.base_dist = uniform(loc=low, scale=high-low)


    def rvs(self, *args, **kwargs):
        """Generate a Gaussian kernel with uniformly distributed parameter.

        :returns: :class:`mulearn.kernel.GaussianKernel` -- Gaussian kernel
          having a parameter uniformly chosen at random in the interval
          having `self.low` and `self.high` as extremes.
        """
        return GaussianKernel(self.base_dist.rvs())


class ExponentialFuzzifierDistribution(rv_continuous):
    """Uniform distribution for gaussian kernels."""

    def __init__(self, low=0, high=1):
        """Build an object of type `ExponentialFuzzifierDistribution`.

        :param low: Lower bound of the interval defining the support of the
          uniform distribution associated to the parameter $\alpha$ of a
          :class:`mulearn.fuzzifier.ExponentialFuzzifier`, defaults to `0`.
        :type low: `float`
        :param high: Upper bound of the interval defining the support of the
          uniform distribution associated to the parameter $\alpha$ of a
          :class:`mulearn.fuzzifier.ExponentialFuzzifier`, defaults to `1`.
        :type high: `float`
        :raises: ValueError if the arguments `low` and `high` do not define
          an interval (that is, `low` is not lower than `high`).
        """
        super().__init__()
        self.base_dist = uniform(loc=low, scale=high-low)

    def rvs(self, *args, **kwargs):
        """Generate an Exponential fuzzifier with uniformly distributed
        parameter.

        :returns: :class:`mulearn.fuzzifier.ExponentialFuzzifier` -- Exponential
          fuzzifier having a parameter uniformly chosen at random in the
          interval having `self.low` and `self.high` as extremes.
        """
        return ExponentialFuzzifier(profile=self.base_dist.rvs())
