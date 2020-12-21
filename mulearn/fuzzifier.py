
"""This module implements fuzzifiers used in mulearn.
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import copy


def _safe_exp(r):
    with np.errstate(over="raise"):
        try:
            return np.exp(r)
        except FloatingPointError:
            return 1


class Fuzzifier:
    """Base class for fuzzifiers.

    The base class for fuzzifiers is Fuzzifier: it exposes a basic constructor
    which is called from the subclasses, and two methods `get_membership`
    (returning the membership function inferred from data) and `get_profile`
    computing information exploitable in order to visualize the fuzzifier
    in graphical form."""

    def __init__(self):
        """Create an instance of :class:`Fuzzifier`."""
        self.sq_radius_05 = None
        self.x_to_sq_dist = None
        self.r_to_mu = None

    def _get_r_to_mu(self):
        r"""Build membership function in feature space.

        Return a function that transforms the square distance between
        center of the learnt sphere and the image of a point in data
        space into a membership degree.

        **Note** This function is meant to be called internally by the
        `get_membership` method in the base `Fuzzifier` class.


        :returns: function -- function mapping square distance to membership.

        """

        check_is_fitted(self, ["r_to_mu"])
        return self.r_to_mu

    def get_membership(self):
        """Return the induced membership function.

        :raises: NotFittedError if `fit` has not been called
        :returns: function -- the induced membership function
        """
        r_to_mu = self._get_r_to_mu()
        return lambda x: r_to_mu(self.x_to_sq_dist(np.array(x)))

    def get_profile(self, X):
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
        rdata = list(map(self.x_to_sq_dist, X))
        rdata_synth = np.linspace(0, max(rdata) * 1.1, 200)
        estimate = list(map(self._get_r_to_mu(), rdata_synth))
        return [rdata, rdata_synth, estimate]

    def __str__(self):
        """Return the string representation of a fuzzifier."""
        return self.__repr__()

    def __eq__(self, other):
        """Check fuzzifier equality w.r.t. other objects."""
        return type(self) == type(other)

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
        """Return a serializable description of the fuzzifier."""
        d = copy.deepcopy(self.__dict__)
        if 'x_to_sq_dist' in d:
            del d['x_to_sq_dist']
        return d

    def __setstate__(self, d):
        """Ensure fuzzifier consistency after deserialization."""
        self.__dict__ = d


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
        super().__init__()

        self.profile = profile

        self.name = 'Crisp'
        self.latex_name = '$\\hat\\mu_{\\text{crisp}}$'

    def fit(self, X, y):
        r"""Fit the fuzzifier on training data.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: membership degrees of the values in `X`.
        :type y: vector of floats having the same length of `X`
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
        check_array(X)
        check_X_y(X, y)

        if self.profile == "fixed":
            self.r_to_mu = lambda r: 1 if r <= self.sq_radius_05 else 0

        elif self.profile == "infer":
            R = np.fromiter(map(self.x_to_sq_dist, X), dtype=float)

            def r_to_mu(r, sq_radius_05):
                result = np.ones(len(r))
                result[r > sq_radius_05] = 0
                return result

            p_opt, _ = curve_fit(r_to_mu, R, y,
                                 bounds=((0,), (np.inf,)))

            if p_opt[0] < 0:
                raise ValueError("Profile fit returned a negative parameter")

            self.r_to_mu = lambda r: r_to_mu([r], *p_opt)[0]
        else:
            raise ValueError("'profile' parameter should either be equal to "
                             f"'fixed' or 'infer' (provided: {self.profile})")

    def __repr__(self):
        """Return the python representation of the fuzzifier."""
        if self.profile != self.default_profile:
            return f"CrispFuzzifier(profile={self.profile})"
        else:
            return "CrispFuzzifier()"


class LinearFuzzifier(Fuzzifier):
    """Crisp fuzzifier.

    Fuzzifier corresponding to a fuzzy set whose membership in feature space
    linearly decreases from 1 to 0."""

    default_profile = "fixed"

    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`LinearFuzzifier`.

        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core, while `'infer'` fits the profile function on the
          provided examples.
        :type profile: str
        """
        super().__init__()

        self.profile = profile

        self.name = 'Linear'
        self.latex_name = '$\\hat\\mu_{\\text{lin}}$'

    def fit(self, X, y):
        r"""Fit the fuzzifier on training data.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: membership degrees of the values in `X`.
        :type y: vector of floats having the same length of `X`
        :raises: ValueError if self.profile is not set either to `'fixed'` or
          to `'infer'`.

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
        check_array(X)
        check_X_y(X, y)
        R = np.fromiter(map(self.x_to_sq_dist, X), dtype=float)

        sq_radius_1_guess = np.median([self.x_to_sq_dist(x)
                                       for x, mu in zip(X, y) if mu >= 0.99])

        if self.profile == 'fixed':
            def r_to_mu(R_arg, sq_radius_1):
                return [np.clip(1 - 0.5 *
                                (r - sq_radius_1) /
                                (self.sq_radius_05 - sq_radius_1),
                                0, 1)
                        for r in R_arg]

            p_opt, _ = curve_fit(r_to_mu, R, y,
                                 p0=(sq_radius_1_guess,),
                                 bounds=((0,), (np.inf,)))

        elif self.profile == 'infer':

            def r_to_mu(R_arg, sq_radius_1, sq_radius_0):
                return [np.clip(1 - (sq_radius_1 - r) /
                                (sq_radius_1 - sq_radius_0), 0, 1)
                        for r in R_arg]

            p_opt, _ = curve_fit(r_to_mu, R, y,
                                 p0=(sq_radius_1_guess, 10 * self.sq_radius_05),
                                 bounds=((0, 0), (np.inf, np.inf,)))
        else:
            raise ValueError("'profile' parameter should be equal to "
                             "'fixed' or 'infer' (provided value: {profile})")
        if min(p_opt) < 0:
            raise ValueError('Profile fitting returned a negative parameter')

        self.r_to_mu = lambda r: r_to_mu([r], *p_opt)[0]

    def __repr__(self):
        """Return the python representation of the fuzzifier."""
        if self.profile != self.default_profile:
            return f"LinearFuzzifier(profile={self.profile})"
        else:
            return "LinearFuzzifier()"


class ExponentialFuzzifier(Fuzzifier):
    """Exponential fuzzifier.

    Fuzzifier corresponding to a fuzzy set whose membership in feature space
    exponentially decreases from 1 to 0."""

    default_profile = "fixed"
    default_alpha = -1

    def __init__(self, profile=default_profile, alpha=default_alpha):
        r"""Create an instance of :class:`ExponentialFuzzifier`.

        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core, `'infer'` fits the profile function on the
          provided examples, and `'alpha'` allows for manually setting the
          exponential decay via the `alpha` parameter.
        :type profile: str
        :param alpha: fixed exponential decay of the fuzzifier.
        :type alpha: float
        """

        super().__init__()

        self.profile = profile
        self.alpha = alpha

        self.name = "Exponential"
        self.latex_name = r"$\hat\mu_{\text{exp}}$"

    def fit(self, X, y):
        r"""Fit the fuzzifier on training data.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: membership degrees of the values in `X`.
        :type y: vector of floats having the same length of `X`
        :raises: ValueError if self.profile is not set either to `'fixed'`,
          `'infer'`, or `'alpha'`.

        In this fuzzifier, the function that transforms the square distance
        between the center of the learnt sphere and the image of a point in
        the original space into a membership degree has the form

        .. math::
          \mu(r) = \begin{cases}  1    & \text{if $r \leq r_1$,} \\
                                  e(r) & \text{otherwise,}
                   \end{cases}

        where $e$ is an exponential function decreasing from 1 to 0. The
        shape of this function is chosen so that the membership profile will be
        equal to 0.5 when $r$ is equal to the learnt square radius of the
        sphere, and induced via interpolation of `X` and `y` when it is has
        been set to `'infer'`; finally, when the parameter is set to `'alpha'`
        the exponential decay of $e$ is manually set via the `alpha` parameter
        of the class constructor.
        """
        check_array(X)
        check_X_y(X, y)

        if self.alpha > 0 and self.profile != "alpha":
            raise ValueError(f"'alpha' value is specified, but 'profile' "
                             f"is set to '{self.profile}'")

        if self.profile == "alpha":
            if self.alpha < 0 or self.alpha > 1:
                raise ValueError("alpha must be set to a float between 0 and 1 "
                                 "when 'profile' is 'alpha'")

        r_1_guess = np.median([self.x_to_sq_dist(x)
                               for x, mu in zip(X, y) if mu >= 0.99])

        s_guess = (self.sq_radius_05 - r_1_guess) / np.log(2)

        R = np.fromiter(map(self.x_to_sq_dist, X), dtype=float)

        if self.profile == "fixed":
            def r_to_mu(R_data, sq_radius_1):
                return [np.clip(_safe_exp(-(r - sq_radius_1) /
                                          (self.sq_radius_05 - sq_radius_1)
                                          * np.log(2)), 0, 1) for r in R_data]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_opt, _ = curve_fit(r_to_mu, R, y, p0=(r_1_guess,),
                                     maxfev=2000, bounds=((0,), (np.inf,)))
            self.r_to_mu = lambda r: r_to_mu([r], *p_opt)[0]

        elif self.profile == "infer":
            def r_to_mu(R_data, r_1, s):
                return [np.clip(_safe_exp(-(r - r_1) / s), 0, 1)
                        for r in R_data]

            p_opt, _ = curve_fit(r_to_mu, R, y, p0=(r_1_guess, s_guess),
                                 # bounds=((0, 0), (np.inf, np.inf)),
                                 maxfev=2000)

            self.r_to_mu = lambda r: r_to_mu([r], *p_opt)[0]

        elif self.profile == "alpha":
            r_sample = map(self.x_to_sq_dist, X)

            q = np.percentile([s - self.sq_radius_05 for s in r_sample
                               if s > self.sq_radius_05], 100 * self.alpha)

            def r_to_mu(R_data, sq_radius_1):
                return [np.clip(_safe_exp(np.log(self.alpha) /
                                          q * (r - sq_radius_1)), 0, 1)
                        for r in R_data]

            p_opt, _ = curve_fit(r_to_mu, R, y, p0=(r_1_guess,),
                                 bounds=((0,), (np.inf,)))
            self.r_to_mu = lambda r: r_to_mu([r], *p_opt)[0]
        else:
            raise ValueError("'profile' parameter should be equal to "
                             "'infer', 'fixed' or 'alpha' "
                             f"(provided value: {self.profile})")

    def __repr__(self):
        obj_repr = "ExponentialFuzzifier("
        if self.profile != self.default_profile:
            obj_repr += f", profile={self.profile}"
        if self.alpha != self.default_alpha:
            obj_repr += f", alpha={self.alpha}"
        if obj_repr.endswith(", "):
            return obj_repr + ")"
        else:
            return "ExponentialFuzzifier()"


class QuantileConstantPiecewiseFuzzifier(Fuzzifier):
    """Quantile-based constant piecewise fuzzifier.

    Fuzzifier corresponding to a fuzzy set with a piecewise constant membership
    function, whose steps are defined according to the quartiles of the squared
    distances between images of points and center of the learnt sphere."""

    def __init__(self):
        r"""Create an instance of :class:`QuantileConstantPiecewiseFuzzifier`"""

        super().__init__()

        self.name = 'QuantileConstPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{q\\_const}}$'

    def fit(self, X, y):
        """Fit the fuzzifier on training data.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: membership degrees of the values in `X`.
        :type y: vector of floats having the same length of `X`

        The piecewise membership function is built so that its steps are chosen
        according to the quartiles of square distances between images of the
        points in `X` center of the learnt sphere.
        """
        check_array(X)
        check_X_y(X, y)

        R = np.fromiter(map(self.x_to_sq_dist, X), dtype=float)

        sq_radius_1 = np.median([self.x_to_sq_dist(x)
                                 for x, mu in zip(X, y) if mu >= 0.99])
        external_dist = [r - sq_radius_1
                         for r in R if r > sq_radius_1]

        if external_dist:
            m = np.median(external_dist)
            q1 = np.percentile(external_dist, 25)
            q3 = np.percentile(external_dist, 75)
        else:
            m = q1 = q3 = 0

        def r_to_mu(r):
            return 1 if r <= sq_radius_1 \
                else 0.75 if r <= sq_radius_1 + q1 \
                else 0.5 if r <= sq_radius_1 + m \
                else 0.25 if r <= sq_radius_1 + q3 \
                else 0

        self.r_to_mu = r_to_mu

    def __repr__(self):
        return "QuantileConstantPiecewiseFuzzifier()"


class QuantileLinearPiecewiseFuzzifier(Fuzzifier):
    """Quantile-based linear piecewise fuzzifier.

    Fuzzifier corresponding to a fuzzy set with a piecewise linear membership
    function, whose steps are defined according to the quartiles of the squared
    distances between images of points and center of the learnt sphere."""

    def __init__(self):
        r"""Create an instance of :class:`QuantileLinearPiecewiseFuzzifier`."""

        super().__init__()

        self.name = 'QuantileLinPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{q\\_lin}}$'

    def fit(self, X, y):
        """Fit the fuzzifier on training data.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: membership degrees of the values in `X`.
        :type y: vector of floats having the same length of `X`

        The piecewise membership function is built so that its steps are chosen
        according to the quartiles of square distances between images of the
        points in `X` center of the learnt sphere.
        """
        check_array(X)
        check_X_y(X, y)

        R = np.fromiter(map(self.x_to_sq_dist, X), dtype=float)

        sq_radius_1 = np.median([self.x_to_sq_dist(x)
                                 for x, mu in zip(X, y) if mu >= 0.99])
        external_dist = [r - sq_radius_1
                         for r in R if r > sq_radius_1]

        if external_dist:
            m = np.median(external_dist)
            q1 = np.percentile(external_dist, 25)
            q3 = np.percentile(external_dist, 75)
            mx = np.max(external_dist)
        else:
            m = q1 = q3 = mx = 0

        def r_to_mu(r):
            ssd = sq_radius_1
            return 1 if r <= ssd \
                else (-r + ssd) / (4 * q1) + 1 if r <= ssd + q1 \
                else (-r + ssd + q1) / (4 * (m - q1)) + 3 / 4 if r <= ssd + m \
                else (-r + ssd + m) / (4 * (q3 - m)) + 1 / 2 if r <= ssd + q3 \
                else (-r + ssd + q3) / (4 * (mx - q3)) + 1 / 4 if r <= ssd + mx\
                else 0

        self.r_to_mu = r_to_mu

    def __repr__(self):
        return "QuantileLinearPiecewiseFuzzifier()"
