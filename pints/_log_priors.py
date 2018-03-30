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
import scipy.stats
import scipy.special


class ComposedLogPrior(pints.LogPrior):
    """
    *Extends:* :class:`LogPrior`

    LogPrior composed of one or more other LogPriors. The evaluation of the
    composed log-prior assumes the input log-priors are all independent from
    each other

    For example: ``p = ComposedLogPrior(log_prior1, log_prior2, log_prior3)``.
    """
    def __init__(self, *priors):
        # Check if sub-priors given
        if len(priors) < 1:
            raise ValueError('Must have at least one sub-prior')

        # Check if proper priors, count dimension
        self._dimension = 0
        for prior in priors:
            if not isinstance(prior, pints.LogPrior):
                raise ValueError('All sub-priors must extend pints.LogPrior.')
            self._dimension += prior.n_parameters()

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

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._dimension

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        output = np.zeros((n, self._dimension))
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            output[:, lo:hi] = prior.sample(n)
        return output


class MultivariateNormalLogPrior(pints.LogPrior):
    """
    *Extends:* :class:`LogPrior`

    Defines a multivariate normal (log)prior with a given ``mean`` and
    ``covariance`` matrix.

    For example::

        p = MultivariateNormalLogPrior(
                np.array([0, 0]), np.array([[1, 0],[0, 1]]))

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
        self._dimension = mean.shape[0]

    def __call__(self, x):
        return np.log(
            scipy.stats.multivariate_normal.pdf(
                x, mean=self._mean, cov=self._covariance))

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._dimension

    def sample(self, n=1):
        """ See :meth:`LogPrior.call()`. """
        # Note: size=n returns shape (n, d)
        return np.random.multivariate_normal(
            self._mean, self._covariance, size=n)


class NormalLogPrior(pints.LogPrior):
    """
    *Extends:* :class:`LogPrior`

    Defines a 1-d normal (log) prior with a given ``mean`` and
    ``standard_deviation``.

    For example: ``p = NormalLogPrior(0, 1)`` for a mean of ``0`` and standard
    deviation of ``1``.
    """
    def __init__(self, mean, standard_deviation):
        # Check that standard deviation is positive
        if float(standard_deviation) <= 0:
            raise ValueError('Standard deviation must be positve')

        # Parse input arguments
        self._mean = float(mean)
        self._sigma = float(standard_deviation)

        # Cache constants
        self._offset = 1 / np.sqrt(2 * np.pi * self._sigma ** 2)
        self._factor = 1 / (2 * self._sigma ** 2)

    def __call__(self, x):
        return self._offset - self._factor * (x[0] - self._mean)**2

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.normal(self._mean, self._sigma, size=(n, 1))

class StudentTLogPrior():
    """
    *Extends:* :class:`LogPrior`

    Defines a 1-d Student-t (log) prior with a given ``degrees of freedom``, ``location``,
    and ``scale``.

    For example: ``p = StudentTLogPrior(3, 0, 1)`` for degrees of freedom of ``3``, a location parameter of ``0``,
    and scale of ``1``.
    """
    def __init__(self, df, location, scale):
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
        self._first = 0.5 * (1.0 + self._df)
        self._log_df = np.log(self._df)
        self._log_scale = np.log(self._scale)
        self._log_beta = np.log(scipy.special.beta(0.5 * self._df, 0.5))

    def __call__(self, x):
        return self._first * (self._log_df - np.log(self._df + ((x[0] - self._location) / self._scale)**2)) - 0.5 * self._log_df - self._log_scale - self._log_beta

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return rvs(df=self._df, loc=self._location, scale=self._scale, size=1)

    def __call__(self, x):
        return self._first * (self._log_df - np.log(self._df + ((x[0] - self._location) / self._scale)**2)) - 0.5 * self_log_df - self._log_scale - self._log_beta

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return scipy.stats.t.rvs(df=self._df, loc=self._location, scale=self._scale, size=1)


class UniformLogPrior(pints.LogPrior):
    """
    *Extends:* :class:`LogPrior`

    Defines a uniform prior over a given range.

    The range includes the lower, but not the upper boundaries, so that any
    point ``x`` with a non-zero prior must have ``lower <= x < upper``.

    For example: ``p = UniformLogPrior([1, 1, 1], [10, 10, 100])``, or
    ``p = UniformLogPrior(Boundaries([1, 1, 1], [10, 10, 100]))``.

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
            self._boundaries = pints.Boundaries(lower_or_boundaries, upper)

        # Cache dimension
        self._dimension = self._boundaries.n_parameters()

        # Cache output value
        self._minf = -float('inf')
        self._value = -np.log(np.product(self._boundaries.range()))

    def __call__(self, x):
        return self._value if self._boundaries.check(x) else self._minf

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._dimension

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.uniform(
            self._boundaries.lower(),
            self._boundaries.upper(),
            size=(n, self._dimension))

