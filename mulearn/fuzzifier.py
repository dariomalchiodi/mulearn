
"""This module implements fuzzifiers used in mulearn.
"""

import copy
import logging
import warnings

import json_fix
import numpy as np
from scipy.optimize import curve_fit
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

np.seterr(over='ignore')

logger = logging.getLogger(__name__)

def _safe_exp(r):
    with np.errstate(over="raise"):
        try:
            return np.exp(r)
        except FloatingPointError:
            return 1

def exp_clip(a):
    return np.where(a > 0, 1, np.exp(a))


class Fuzzifier:
    """Base class for fuzzifiers.

    The base class for fuzzifiers is Fuzzifier: it exposes a basic constructor
    which is called from the subclasses, and two methods `get_membership`
    (returning the membership function inferred from data) and `get_profile`
    computing information exploitable in order to visualize the fuzzifier
    in graphical form."""

    def __init__(self, profile):
        self.profile = profile

    def get_membership(self):
        """Return the induced membership function.

        :raises: NotFittedError if `fit` has not been called
        :returns: function -- the induced membership function
        """
        raise NotImplementedError(
            'The base class does not implement the `get_membership` method')

    def get_profile(self, squared_R):
        r"""Return information about the learnt membership function profile.

        The profile of a membership function $\mu: X \rightarrow [0, 1]$ is
        intended here as the associated function $p: \mathbb R^+ \rightarrow
        [0, 1]$ still returning membership degrees, but considering its
        arguments in the feature space. More precisely, if `X` contains the
        values $x_1, \dots, x_n$, $R^2$ is the function mapping any point in
        data space into the squared distance between its image and the center
        $a$ of the learnt fuzzy set in feature space, the function
        `get_profile` computes the following information about $p$:

        * a list $r_\mathrm{data} = [ R^2(x_i), i = 1, \dots, n]$ containing
          the distances between the images of the points in `X` and $a$;
        * a list $\tilde{r}_\mathrm{data}$ containing 200 possible
          distances between $a$ and the image of a point in data space, evenly
          distributed between $0$ and $\max r_{\mathrm{data}}$;
        * a list $e = [\hat\mu(r_i), r_i \in \tilde{r}_{\mathrm{data}}]$
          gathering the profile values for each element in
          $\tilde{r}_{\mathrm{data}}$.

        This information can be used in order to graphically show the
        membership profile, which is always plottable, whereas the membership
        function isn't mostly of the time (unless the elements in `X` are
        either one- or bidimensional vectors).

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: list -- $[r_{\mathrm{data}}, \tilde{r}_\mathrm{data}, e]$.

        """
        rdata_synth = np.linspace(0, max(squared_R) * 1.1, 200)
        estimate = self.get_membership(rdata_synth)
        return [squared_R, rdata_synth, estimate]

    def __str__(self):
        """Return the string representation of a fuzzifier."""
        return self.__repr__()

    def __eq__(self, other):
        """Check fuzzifier equality w.r.t. other objects."""
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Check fuzzifier inequality w.r.t. other objects."""
        return not self == other

    def __hash__(self):
        """Generate hashcode for a fuzzifier."""
        return hash(self.__repr__())

    @staticmethod
    def __nonzero__():
        """Check if a fuzzifier is non-null."""
        return True
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __json__(self):
        return {'class': self.__class__.__name__, 'profile': self.profile}
    
    def __repr__(self):
        """Return the python representation of the fuzzifier."""
        arg = self.profile if self.profile != self.default_profile else ''
        return f'{self.__class__.__name__}({arg})'


class CrispFuzzifier(Fuzzifier):
    """Crisp fuzzifier.

    Fuzzifier corresponding to a crisp (classical) set: membership is always
    equal to either $0$ or $1$."""

    default_profile = "fixed"
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`CrispFuzzifier`.

        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core, while `'infer'` fits a generic threshold
          function on the provided examples.
        :type profile: str
        """
        super().__init__(profile)
    
    def fit(self, squared_R, mu, squared_radius):
        r"""Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float

        :raises: ValueError if self.profile is not set either to `'fixed'` or
          to `'infer'`.

        The fitting process is done considering a threshold-based membership
        function, in turn corresponding to a threshold-based profile of the
        form

        .. math::
          p(r) = \begin{cases} 1 & \text{if $r \leq r_\text{crisp}$,} \\
                               0 & \text{otherwise.} \end{cases}

        The threshold $r_\text{crisp}$ is set to the learnt square radius of
        the sphere when the `profile` attribute of the class have been set to
        `'fixed'`, and induced via interpolation of `X` and `y` attributes when
        it is has been set to `'infer'`.
        """

        self.name = 'Crisp'
        self.latex_name = '$\\hat\\mu_{\\text{crisp}}$'

        assert len(squared_R) == len(mu)

            
        if self.profile == "fixed":
            self.threshold_ = squared_radius

        elif self.profile == "infer":

            def r2_to_mu(r, threshold):
                result = np.ones(len(r))
                result[r > threshold] = 0
                return result

            try:
                [t_opt], _ = curve_fit(r2_to_mu, squared_R, mu, 
                                    bounds=((0,), (np.inf,)))
                self.threshold_ = t_opt
                
                if self.threshold_ < 0:
                    logger.warning("Profile fit returned a negative parameter "
                                   f"({self.threshold_})")
            except RuntimeError:
                # interpolation could not take place, fall back to fixed profile
                self.profile = 'fixed'
                self.fit(squared_R, mu, squared_radius)
            
        else:
            raise ValueError("'profile' parameter should either be equal to "
                             f"'fixed' or 'infer' (provided: {self.profile})")
    
    def get_membership(self, R_2):
        check_is_fitted(self, 'threshold_')
        return np.array([1 if r_2 < self.threshold_ else 0 for r_2 in R_2])


class LinearFuzzifier(Fuzzifier):
    """Linear fuzzifier.

    Fuzzifier corresponding to a fuzzy set whose membership in feature space
    linearly decreases from 1 to 0."""

    default_profile = "fixed"
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`LinearFuzzifier`.
        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core, `'triangular'` aims at inducing a triangular or
          trapezoidal membership function in data space and `'infer'` fits a
          generic threshold function on the provided examples.
        :type profile: str"""
        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        r"""Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float

        :raises: ValueError if self.profile is not set to `'fixed'`,
          `'triangular`' or `'infer'`.

        The fitting process is done considering a membership function
        linearly decreasing from $1$ to $0$, in turn corresponding to a profile
        having the general form

        .. math::
          p(r) = \begin{cases} 1 & \text{if $r \leq r_1$,} \\
                               l(r) & \text{if $r_1 < r \leq r_0$,} \\
                               0 & \text{otherwise.} \end{cases}

        The free parameters are chosen in order to guarantee continuity;
        moreover, when the `profile` attribute of the class have been set to
        `'fixed'` the membership profile will be equal to 0.5 when $r$ is equal
        to the learnt square radius of the sphere, and induced via
        interpolation of `X` and `y` when it is has been set to `'infer'`.
        """

        self.name = 'Linear'
        self.latex_name = '$\\hat\\mu_{\\text{lin}}$'
        
        assert len(squared_R) == len(mu)
        
        r_2_1_guess = np.median([r2 for r2, m in zip (squared_R, mu)
                                              if m >= max(mu)*0.99])
        r_2_0_guess = np.median([r2 for r2, m in zip (squared_R, mu)
                                              if m <= min(mu)*1.01])
        r_2_05_guess = (r_2_1_guess + r_2_0_guess) / 2

                                      
        if self.profile == 'fixed':
            def r2_to_mu(R_2, r_2_1):

                result = [np.clip(1 - 0.5 * (r_2-r_2_1)/(squared_radius-r_2_1),
                                  0, 1) for r_2 in R_2]
                
                return result

            [r_2_1_opt], _ = curve_fit(r2_to_mu, squared_R, mu,
                                       p0=(r_2_1_guess,),
                                       bounds=((0,), (np.inf,)))
            self.slope_ = -1 / (2 * (squared_radius - r_2_1_opt))
            self.intercept_ = 1 + r_2_1_opt / (2 * (squared_radius - r_2_1_opt))
        
            
        elif self.profile == 'triangular':
            def r2_to_mu(R_2, r_2_05):
                # TODO: check
                # return [np.clip(1 - r_2 / (2 * squared_radius),
                #                 0, 1) - r_2_1
                #         for r_2 in R_2]
                return [np.clip(1 - r_2 / (2 * r_2_05),
                                0, 1)
                        for r_2 in R_2]

            try:
                [r_2_05_opt], _ = curve_fit(r2_to_mu, squared_R, mu,
                                            p0=(r_2_05_guess,),
                                            bounds=((0,), (np.inf,)))
                self.slope_ = -1 / (2 * r_2_05_opt)
                self.intercept_ = 1
            except RuntimeError:
                # interpolation could not take place, fall back to fixed profile
                self.profile = 'fixed'
                self.fit(squared_R, mu, squared_radius)

        elif self.profile == 'infer':

            def r2_to_mu(R_2, r_2_1, r_2_0):
                return [np.clip(1 - (r_2 - r_2_1) / (r_2_0 - r_2_1), 0, 1)
                        for r_2 in R_2]

            try:
                p_opt, _ = curve_fit(r2_to_mu, squared_R, mu,
                                    p0=(r_2_1_guess, r_2_0_guess), 
                                    bounds=((-np.inf, -np.inf),
                                            (np.inf, np.inf,)))
                r_2_1_opt, r_2_0_opt = p_opt
                self.slope_ = -1 / (r_2_0_opt - r_2_1_opt)
                self.intercept_ = 1 + r_2_1_opt / (r_2_0_opt - r_2_1_opt)
            except RuntimeError:
                # interpolation could not take place, fall back to fixed profile
                self.profile = 'fixed'
                self.fit(squared_R, mu, squared_radius)

        else:
            raise ValueError("'profile' parameter should be equal to "
                        "'fixed' or 'infer' (provided value: {self.profile})")
        if self.slope_ > 0:
            logger.warning('Profile fitting returned a positive slope '
                            f'({self.slope_})')
        if self.intercept_ < 0:
            logger.warning('Profile fitting returned a negative intercept '
                            f'({self.intercept_})')
        
        return self

    def get_membership(self, R_2):
        check_is_fitted(self, ['slope_', 'intercept_'])
        return np.array([np.clip(self.slope_ * r_2 + self.intercept_, 0, 1)
                         for r_2 in R_2])


class ExponentialFuzzifier(Fuzzifier):
    """Exponential fuzzifier.

    Fuzzifier corresponding to a fuzzy set whose membership in feature space
    exponentially decreases from 1 to 0."""

    default_profile = "fixed"
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`ExponentialFuzzifier`.

        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core, `'infer'` fits the profile function on the
          provided examples, and a numeric value allows for manually setting
          the exponential decay of the fuzzifier.
        :type profile: str or numeric value"""

        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        r"""Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float
        
        :raises: ValueError if self.profile is not set either to `'fixed'`,
          `'infer'` or to a numeric value in $[0, 1]$.

        In this fuzzifier, the function that transforms the square distance
        between the center of the learnt sphere and the image of a point in
        the original space into a membership degree has the form

        .. math::
          \mu(r) = \begin{cases}  1    & \text{if $r \leq r_1$,} \\
                                  e(r) & \text{otherwise,}
                   \end{cases}

        where $e$ is an exponential function decreasing from 1 to 0. The
        shape of this function is chosen so that:
        - when `self.profile`=`'fixed'` the membership profile will be equal
          equal to 0.5 when $r$ is equal to the learnt square radius of the
          sphere,
        - induced via interpolation of `squared_R` and `mu` when
          `self.profile`=`'infer'` and
        - manually set when `self.profile` is a number $\alpha \in [0, 1]$, the
          latter now implying that the fuzzifier value will be 0.5 exactly when
          its argument equals the $\alpha$-quantile of the squared distances
          of the provided examples in feature space.
        """

        self.name = "Exponential"
        self.latex_name = r"$\hat\mu_{\text{exp}}$"

        assert len(squared_R) == len(mu)

        if isinstance(self.profile, (int, float)):
            if self.profile < 0 or self.profile > 1:
                raise ValueError("profile must be set to a number between 0 "
                                 "and 1, or either to 'fixed' or 'infer'")

        r_2_1_guess = np.median([r_2 for r_2, m in zip(squared_R, mu)
                                if m >= max(mu)*0.9])
        s_guess = np.log(2) / (squared_radius - r_2_1_guess)

        if self.profile == "fixed":
            def r2_to_mu(R_2, r_2_1):
                return [exp_clip(-np.log(2) * \
                                 (r_2 - r_2_1) / (squared_radius - r_2_1))
                        for r_2 in R_2]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                [r_2_1_opt], _ = curve_fit(r2_to_mu, squared_R, mu,
                                           p0=(r_2_1_guess,), maxfev=2000,
                                           bounds=((0,), (np.inf,)))
                denominator = squared_radius - r_2_1_opt
                self.slope_ = -np.log(2) / denominator
                self.intercept_ = r_2_1_opt * np.log(2) / denominator

        elif self.profile == "infer":
            def r2_to_mu(R_2, r_2_1, s):
                return [exp_clip(-(r_2 - r_2_1) / s) for r_2 in R_2]

            try:
                p_opt, _ = curve_fit(r2_to_mu, squared_R, mu,
                                    p0=(r_2_1_guess, s_guess),
                                    # bounds=((0, 0), (np.inf, np.inf)),
                                    maxfev=2000)
                r_2_1_opt, s_opt = p_opt
                self.slope_ = -1 / s_opt
                self.intercept_ = r_2_1_opt / s_opt
            except RuntimeError:
                # interpolation could not take place, fall back to fixed profile
                self.profile = 'fixed'
                self.fit(squared_R, mu, squared_radius)

        elif isinstance(self.profile, (int, float)):
            alpha = self.profile
            def r2_to_mu(R_2, r_2_1):
                inner = [r_2 - r_2_1 for r_2 in squared_R if r_2 > r_2_1]
                if len(inner) > 0:
                    q = np.percentile(inner, 100 * alpha)
                    return [exp_clip(np.log(alpha) / q * (r_2 - r_2_1))
                            for r_2 in R_2]
                else:
                    # all points have within the sphere -> unit membership
                    return [1] * len(R_2)

            try:
                [r_2_1_opt], _ = curve_fit(r2_to_mu, squared_R, mu,
                                        p0=(r_2_1_guess,),
                                        bounds=((0,), (np.inf,)))
                inner = [r_2 - r_2_1_opt for r_2 in squared_R if r_2 > r_2_1_opt]
                q = np.percentile(inner, 100 * alpha)
                self.slope_ = np.log(alpha) / q
                self.intercept_ = -r_2_1_opt * np.log(alpha) / q
            except RuntimeError:
                # interpolation could not take place, fall back to fixed profile
                self.profile = 'fixed'
                self.fit(squared_R, mu, squared_radius)
            
        else:
            raise ValueError("'self.profile' attribute should be equal to "
                             "'infer', 'fixed' or 'alpha' "
                             f"(provided value: {self.profile})")
        
        if self.slope_ > 0:
            logger.warning('Profile fitting returned a positive slope '
                            f'({self.slope_})')
        if self.intercept_ < 0:
            logger.warning('Profile fitting returned a negative intercept '
                            f'({self.intercept_})')
        
        return self
    
    def get_membership(self, R_2):
        check_is_fitted(self, ['slope_', 'intercept_'])
        return exp_clip(self.slope_ * R_2 + self.intercept_)


class QuantileConstantPiecewiseFuzzifier(Fuzzifier):
    """Quantile-based constant piecewise fuzzifier.

    Fuzzifier corresponding to a fuzzy set with a piecewise constant membership
    function, whose steps are defined according to the quartiles of the squared
    distances between images of points and center of the learnt sphere."""

    default_profile = 'infer'
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`QuantileConstantPiecewiseFuzzifier`.
        
        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core and `'infer'` fits the profile function on the
          provided examples.
        :type profile: str"""

        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        """Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float
        
        :raises: ValueError if self.profile is not set either to `'fixed'`,
          `'infer'` or to a numeric value in $[0, 1]$.

        The piecewise membership function is built so that its steps are chosen
        according to the quartiles of square distances between images of the
        points in `X` center of the learnt sphere.
        """

        if self.profile != 'infer':
            raise NotImplementedError("profile='fixed' not yet implemented.")

        self.name = 'QuantileConstPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{q\\_const}}$'

        assert len(squared_R) == len(mu)

        self.r_2_1_ = np.median([r_2 for r_2, m in zip(squared_R, mu)
                               if m >= max(mu)*0.99])
        
        external_dist = [r_2 - self.r_2_1_ for r_2 in squared_R
                                           if r_2 > self.r_2_1_]
              
        if external_dist:
            self.m_ = np.median(external_dist)
            self.q1_ = np.percentile(external_dist, 25)
            self.q3_ = np.percentile(external_dist, 75)
        else:
            self.m_ = self.q1_ = self.q3_ = 0
        
        return self
    
    def get_membership(self, R_2):
        check_is_fitted(self, ['r_2_1_', 'm_', 'q1_', 'q3_'])
        return np.array([1 if r_2 <= self.r_2_1_ \
                         else 0.75 if r_2 <= self.r_2_1_ + self.q1_ \
                         else 0.5 if r_2 <= self.r_2_1_ + self.m_ \
                         else 0.25 if r_2 <= self.r_2_1_ + self.q3_ \
                         else 0 for r_2 in R_2])


class QuantileLinearPiecewiseFuzzifier(Fuzzifier):
    """Quantile-based linear piecewise fuzzifier.

    Fuzzifier corresponding to a fuzzy set with a piecewise linear membership
    function, whose steps are defined according to the quartiles of the squared
    distances between images of points and center of the learnt sphere."""

    default_profile = 'infer'
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`QuantileLinearPiecewiseFuzzifier`.
        
        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core and `'infer'` fits the profile function on the
          provided examples.
        :type profile: str"""

        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        """Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float

        The piecewise membership function is built so that its steps are chosen
        according to the quartiles of square distances between images of the
        points in `X` center of the learnt sphere.
        """

        self.name = 'QuantileLinPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{q\\_lin}}$'

        assert len(squared_R) == len(mu)

        self.r_2_1_ = np.median([r_2 for r_2, m in zip(squared_R, mu)
                               if m >= max(mu)*0.99])
        
        
        external_dist = [r_2 - self.r_2_1_ for r_2 in squared_R
                                           if r_2 > self.r_2_1_]
              
        if external_dist:
            self.m_ = np.median(external_dist)
            self.q1_ = np.percentile(external_dist, 25)
            self.q3_ = np.percentile(external_dist, 75)
            self.max_ = np.max(external_dist)
        else:
            self.m_ = self.q1_ = self.q3_ = self.max_ = 0

        return self
        
    def get_membership(self, R_2):
        check_is_fitted(self, ['r_2_1_', 'm_', 'q1_', 'q3_'])
        return np.array([1 if r_2 <= self.r_2_1_ \
                else (-r_2+self.r_2_1_)/(4*self.m_) + 1 \
                            if r_2 <= self.r_2_1_+self.q1_ \
                else (-r_2+self.r_2_1_+self.q1_)/(4*(self.m_-self.q1_)) + 3/4 \
                            if r_2 <= self.r_2_1_+self.m_ \
                else (-r_2+self.r_2_1_+self.m_)/(4*(self.q3_-self.m_)) + 1/2 \
                            if r_2 <= self.r_2_1_+self.q3_ \
                else (-r_2+self.r_2_1_+self.q3_)/(4*(self.max_-self.q3_)) + 1/4\
                            if r_2 <= self.r_2_1_+self.max_\
                else 0 for r_2 in R_2])

    
