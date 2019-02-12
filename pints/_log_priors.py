#
# Defines different prior distributions
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy
import scipy.special
import scipy.stats


class ComposedLogPrior(pints.LogPrior):
    """
    N-dimensional LogPrior composed of one or more other Ni-dimensional
    LogPriors, such that ``sum(Ni) = N``. The evaluation of the composed
    log-prior assumes the input log-priors are all independent from each other.

    For example, a composed log prior::

        p = ComposedLogPrior(log_prior1, log_prior2, log_prior3)

    where ``log_prior1``, 2, and 3 each have dimension 1 will have dimension 3
    itself.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, *priors):
        # Check if sub-priors given
        if len(priors) < 1:
            raise ValueError('Must have at least one sub-prior')

        # Check if proper priors, count dimension
        self._n_parameters = 0
        for prior in priors:
            if not isinstance(prior, pints.LogPrior):
                raise ValueError('All sub-priors must extend pints.LogPrior.')
            self._n_parameters += prior.n_parameters()

        # Store
        self._priors = priors

    def __call__(self, x):
        output = 0
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            output += prior(x[lo:hi])
        return output

    def evaluateS1(self, x):
        """
        See :meth:`LogPDF.evaluateS1()`.

        *This method only works if the underlying :class:`LogPrior` classes all
        implement the optional method :class:`LogPDF.evaluateS1().`.*
        """
        output = 0
        doutput = np.zeros(self._n_parameters)
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            p, dp = prior.evaluateS1(x[lo:hi])
            output += p
            doutput[lo:hi] = np.asarray(dp)
        return output, doutput

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._n_parameters

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        output = np.zeros((n, self._n_parameters))
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            output[:, lo:hi] = prior.sample(n)
        return output


class MultivariateNormalLogPrior(pints.LogPrior):
    """
    Defines a multivariate normal (log)prior with a given ``mean`` and
    ``covariance`` matrix.

    For example::

        p = MultivariateNormalLogPrior(
                np.array([0, 0]), np.array([[1, 0],[0, 1]]))

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, mean, covariance):
        # Check input
        mean = pints.vector(mean)
        covariance = np.array(covariance, copy=True)
        covariance.setflags(write=False)
        if covariance.ndim != 2:
            raise ValueError('Given covariance must be a matrix.')
        if not (mean.shape[0] == covariance.shape[0] == covariance.shape[1]):
            raise ValueError('Sizes of mean and covariance do not match.')

        # Store
        self._mean = mean
        self._covariance = covariance
        self._n_parameters = mean.shape[0]

    def __call__(self, x):
        return np.log(
            scipy.stats.multivariate_normal.pdf(
                x, mean=self._mean, cov=self._covariance))

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._n_parameters

    def sample(self, n=1):
        """ See :meth:`LogPrior.call()`. """
        # Note: size=n returns shape (n, d)
        return np.random.multivariate_normal(
            self._mean, self._covariance, size=n)


class NormalLogPrior(pints.LogPrior):
    """
    Defines a 1-d normal (log) prior with a given ``mean`` and
    ``standard_deviation``.

    For example: ``p = NormalLogPrior(0, 1)`` for a mean of ``0`` and standard
    deviation of ``1``.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, mean, standard_deviation):
        # Parse input arguments
        self._mean = float(mean)
        self._sigma = float(standard_deviation)

        # Cache constants
        self._offset = np.log(1 / np.sqrt(2 * np.pi * self._sigma ** 2))
        self._factor = 1 / (2 * self._sigma ** 2)
        self._factor2 = 1 / self._sigma**2

    def __call__(self, x):
        return self._offset - self._factor * (x[0] - self._mean)**2

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        return self(x), self._factor2 * (self._mean - np.asarray(x))

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.normal(self._mean, self._sigma, size=(n, 1))


class BetaLogPrior(pints.LogPrior):
    """
    Defines a beta (log) prior with given shape parameters ``a`` and ``b``.

    For example: ``p = BetaLogPrior(5, 1)`` for a shape parameters ``a=5`` and
    ``b=1``.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, a, b):
        # Parse input arguments
        self._a = float(a)
        self._b = float(b)

        # Validate inputs
        if self._a <= 0:
            raise ValueError('Shape parameter alpha must be positive')
        if self._b <= 0:
            raise ValueError('Shape parameter beta must be positive')

        # Cache constant
        self._log_beta = scipy.special.betaln(self._a, self._b)

    def __call__(self, x):
        if x[0] < 0.0 or x[0] > 1.0:
            return -float('inf')
        else:
            return scipy.special.xlogy(self._a - 1.0,
                                       x[0]) + scipy.special.xlog1py(
                self._b - 1.0, -x[0]) - self._log_beta

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)
        _x = x[0]

        # Account for pathological edges
        if _x == 0.0:
            _x = np.nextafter(0.0, 1.0)
        elif _x == 1.0:
            _x = np.nextafter(1.0, 0.0)

        if _x < 0.0 or _x > 1.0:
            return value, np.asarray([0.])
        else:
            # Use np.divide here to better handle possible v small denominators
            return value, np.asarray([np.divide(self._a - 1., _x) - np.divide(
                self._b - 1., 1. - _x)])

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.beta(self._a, self._b, size=(n, 1))


class ExponentialLogPrior(pints.LogPrior):
    """
    Defines an exponential (log) prior with given rate parameter ``rate`` with
    pdf f(x|rate) = rate * e^-{rate*x}.

    For example: ``p = ExponentialLogPrior(0.5, 1)`` for a rate ``rate=0.5``.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, rate):
        # Parse input arguments
        self._rate = float(rate)

        # Validate inputs
        if self._rate <= 0:
            raise ValueError('Rate parameter "scale" must be positive')

        # Cache constant
        self._log_scale = np.log(self._rate)

    def __call__(self, x):
        if x[0] < 0.0:
            return -float('inf')
        else:
            return self._log_scale - self._rate * x[0]

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)

        if x[0] < 0.0:
            return value, np.asarray([0.])
        else:
            return value, np.asarray([-self._rate])

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.exponential(scale=1. / self._rate, size=(n, 1))


class GammaLogPrior(pints.LogPrior):
    """
    Defines a gamma (log) prior with given shape parameter ``a`` and rate
    parameter ``b``, with pdf f(x|a,b)=b^a * x^(a-1) * e^{-bx} / Gamma(a)

    For example: ``p = GammaLogPrior(5, 1)`` for a shape parameter ``a=5`` and
    rate parameter ``b=1``.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, a, b):
        # Parse input arguments
        self._a = float(a)
        self._b = float(b)

        # Validate inputs
        if self._a <= 0:
            raise ValueError('Shape parameter alpha must be positive')
        if self._b <= 0:
            raise ValueError('Rate parameter beta must be positive')

        # Cache constant
        self._constant = scipy.special.xlogy(self._a,
                                             self._b) - scipy.special.gammaln(
            self._a)

    def __call__(self, x):
        if x[0] < 0.0:
            return -float('inf')
        else:
            return self._constant + scipy.special.xlogy(self._a - 1.,
                                                        x[0]) - self._b * x[0]

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)

        _x = x[0]

        # Account for pathological edge
        if _x == 0.0:
            _x = np.nextafter(0.0, 1.0)

        if _x < 0.0:
            return value, np.asarray([0.])
        else:
            # Use np.divide here to better handle possible v small denominators
            return value, np.asarray([np.divide(self._a - 1., _x) - self._b])

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.gamma(self._a, 1. / self._b, size=(n, 1))


class StudentTLogPrior(pints.LogPrior):
    """
    Defines a 1-d Student-t (log) prior with a given ``location``,
    ``degrees of freedom``,  and ``scale``.

    For example, to create a prior centered around 0 with 3 degrees of freedom
    and a scale of 1, use::

        p = pints.StudentTLogPrior(0, 3, 1)

    Arguments:

    ``location``
        The center of the distribution.
    ``df``
        The number of degrees of freedom of the distribution.
    ``scale``
        The scale of the distribution.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, location, df, scale):
        # Test inputs
        if float(df) <= 0:
            raise ValueError('Degrees of freedom must be positive')
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')

        # Parse input arguments
        self._df = float(df)
        self._location = float(location)
        self._scale = float(scale)

        # Cache constants
        self._log_df = np.log(self._df)

        self._1_sig_sq = 1. / (self._scale * self._scale)

        self._first = 0.5 * (1.0 + self._df)

        self._samp_const = scipy.special.xlogy(-0.5, self._df) - np.log(
            self._scale) - scipy.special.betaln(0.5 * self._df, 0.5)

        self._deriv_const = (-1. - self._df) * self._1_sig_sq

    def __call__(self, x):
        return self._samp_const + self._first * (self._log_df - np.log(
            self._df + self._1_sig_sq * (x[0] - self._location) ** 2))

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        offset = x[0] - self._location
        return self(x), np.asarray([offset * self._deriv_const / (
            self._df + offset * offset * self._1_sig_sq)])

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return scipy.stats.t.rvs(df=self._df, loc=self._location,
                                 scale=self._scale, size=n)


class CauchyLogPrior(pints.LogPrior):
    """
    Defines a 1-d Cauchy (log) prior with a given ``location``, and ``scale``.

    For example, to create a prior centered around 0 and a scale of 5, use::

        p = pints.CauchyLogPrior(0, 5)

    Arguments:

    ``location``
        The center of the distribution.
    ``scale``
        The scale of the distribution.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, location, scale):
        # Test inputs
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')

        # Parse input arguments
        # Cauchy is Student-t with 1 df
        self._df = 1
        self._location = float(location)
        self._scale = float(scale)

        # Cache constants
        self._first = 0.5 * (1.0 + self._df)
        self._log_df = np.log(self._df)
        self._log_scale = np.log(self._scale)
        self._log_beta = np.log(scipy.special.beta(0.5 * self._df, 0.5))

        # Cache scipy stats object
        self._t = scipy.stats.t(
            df=self._df, loc=self._location, scale=self._scale)

    def __call__(self, x):
        return (
            self._first * (
                self._log_df - np.log(
                    self._df + ((x[0] - self._location) / self._scale)**2))
            - 0.5 * self._log_df - self._log_scale - self._log_beta
        )

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return self._t.rvs(n)


class HalfCauchyLogPrior(pints.LogPrior):
    """
    Defines a 1-d half-Cauchy (log) prior with a given ``location`` and
    ``scale``. This is a Cauchy distribution that has been truncated to lie in
    between [0, inf].

    For example, to create a prior centered around 0 and a scale of 5, use::

        p = pints.HalfCauchyLogPrior(0, 5)

    Arguments:

    ``location``
        The center of the distribution.
    ``scale``
        The scale of the distribution.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, location, scale):
        # Test inputs
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')

        # Parse input arguments
        # set df for Student-t for sampling only
        self._df = 1
        self._location = float(location)
        self._scale = float(scale)

        # Cache constants
        self._first = np.log(2)
        self._log_scale = np.log(self._scale)
        self._log_weird = np.log(
            np.pi + 2 * np.arctan(self._location / self._scale))
        self._scale_sq = self._scale**2

        # Store scipy distribution object
        self._t = scipy.stats.t(
            df=self._df, loc=self._location, scale=self._scale)

    def __call__(self, x):
        if x[0] > 0:
            return (
                self._first + self._log_scale - self._log_weird
                - np.log((x[0] - self._location)**2 + self._scale_sq))
        else:
            return -float('inf')

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        samples = self._t.rvs(size=n)
        resample = samples <= 0
        n_resample = np.sum(resample)
        while n_resample:
            samples[resample] = self._t.rvs(size=n_resample)
            resample = samples <= 0
            n_resample = np.sum(resample)
        return samples


class UniformLogPrior(pints.LogPrior):
    """
    Defines a uniform prior over a given range.

    The range includes the lower, but not the upper boundaries, so that any
    point ``x`` with a non-zero prior must have ``lower <= x < upper``.

    For example: ``p = UniformLogPrior([1, 1, 1], [10, 10, 100])``, or
    ``p = UniformLogPrior(RectangularBoundaries([1, 1, 1], [10, 10, 100]))``.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, lower_or_boundaries, upper=None):
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries, pints.Boundaries):
                raise ValueError(
                    'UniformLogPrior requires a lower and an upper bound, or a'
                    ' single Boundaries object.')
            self._boundaries = lower_or_boundaries
        else:
            self._boundaries = pints.RectangularBoundaries(
                lower_or_boundaries, upper)

        # Cache dimension
        self._n_parameters = self._boundaries.n_parameters()

        # Minimum output value
        self._minf = -float('inf')

        # Maximum output value
        # Use normalised value (1/area) for rectangular boundaries,
        # otherwise just use 1.
        if isinstance(self._boundaries, pints.RectangularBoundaries):
            self._value = -np.log(np.product(self._boundaries.range()))
        else:
            self._value = 1

    def __call__(self, x):
        return self._value if self._boundaries.check(x) else self._minf

    def evaluateS1(self, x):
        """ See :meth:`LogPrior.evaluateS1()`. """
        # Ignoring points on the boundaries (i.e. on the surface of the
        # hypercube), because it's very unlikely and won't help the search
        # much...
        return self(x), np.zeros(self._n_parameters)

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._n_parameters

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return self._boundaries.sample(n)
