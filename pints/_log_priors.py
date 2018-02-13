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
            self._dimension += prior.dimension()

        # Store
        self._priors = priors

    def __call__(self, x):
        output = 0
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.dimension()
            output += prior(x[lo:hi])
        return output

    def dimension(self):
        """ See :meth:`LogPrior.dimension()`. """
        return self._dimension

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        output = np.zeros((n, self._dimension))
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.dimension()
            output[:, lo:hi] = prior.sample(n)
        return output


class MultivariateNormalLogPrior(pints.LogPrior):
    """
    *Extends:* :class:`LogPrior`

    Defines a multivariate normal (log)prior with a given `mean` and
    `covariance` matrix.

    For example::

        p = MultivariateNormalPrior(
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

    def dimension(self):
        """ See :meth:`LogPrior.dimension()`. """
        return self._dimension

    def sample(self, n=1):
        """ See :meth:`LogPrior.call()`. """
        # Note: size=n returns shape (n, d)
        return np.random.multivariate_normal(
            self._mean, self._covariance, size=n)


class NormalLogPrior(pints.LogPrior):
    """
    *Extends:* :class:`LogPrior`

    Defines a 1-d normal (log) prior with a given `mean` and `variance`.

    For example: ``p = NormalPrior(0, 1)`` for a mean of ``0`` and variance
    of ``1``.
    """
    def __init__(self, mean, variance):
        # Parse input arguments
        self._mean = float(mean)
        self._sigma = float(variance)

        # Cache constants
        self._offset = 1 / np.sqrt(2 * np.pi * self._sigma ** 2)
        self._factor = 1 / (2 * self._sigma ** 2)

    def __call__(self, x):
        return self._offset - self._factor * (x[0] - self._mean)**2

    def dimension(self):
        """ See :meth:`LogPrior.dimension()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.normal(self._mean, self._sigma, size=(n, 1))


class UniformLogPrior(pints.LogPrior):
    """
    *Extends:* :class:`LogPrior`

    Defines a uniform prior over a given range.

    The range includes the lower, but not the upper boundaries, so that any
    point ``x`` with a non-zero prior must have ``lower <= x < upper``.

    For example: ``p = UniformPrior([1,1,1], [10, 10, 100])``, or
    ``p = UniformPrior(Boundaries([1,1,1], [10, 10, 100]))``.
    """
    def __init__(self, lower_or_boundaries, upper=None):
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries, pints.Boundaries):
                raise ValueError(
                    'UniformPrior requires a lower and an upper bound, or a'
                    ' single Boundaries object.')
            self._boundaries = lower_or_boundaries
        else:
            self._boundaries = pints.Boundaries(lower_or_boundaries, upper)

        # Cache dimension
        self._dimension = self._boundaries.dimension()

        # Cache output value
        self._minf = -float('inf')
        self._value = -np.log(np.product(self._boundaries.range()))

    def __call__(self, x):
        return self._value if self._boundaries.check(x) else self._minf

    def dimension(self):
        """ See :meth:`LogPrior.dimension()`. """
        return self._dimension

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.uniform(
            self._boundaries.lower(),
            self._boundaries.upper(),
            size=(n, self._dimension))

