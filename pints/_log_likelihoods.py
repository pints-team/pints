#
# Log-likelihood functions
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


class KnownNoiseLogLikelihood(pints.ProblemLogLikelihood):
    """
    *Extends:* :class:`ProblemLogLikelihood`

    Calculates a log-likelihood assuming independent normally-distributed noise
    at each time point, using a known value for the standard deviation (sigma)
    of that noise.

    Arguments:

    ``problem``
        A :class:`SingleOutputProblem` or :class`MultiOutputProblem`.
    ``sigma``
        The standard devation(s) of the noise. Can be a single value or a
        sequence of sigma's for each output.

    """
    def __init__(self, problem, sigma):
        super(KnownNoiseLogLikelihood, self).__init__(problem)

        # Check sigma
        no = problem.n_outputs()
        if np.isscalar(sigma):
            sigma = np.ones(no) * float(sigma)
        else:
            sigma = pints.vector(sigma)
            if len(sigma) != no:
                raise ValueError(
                    'Sigma must be a scalar or a vector of length n_outputs.')
        if np.any(sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')

        # Pre-calculate parts
        self._offset = -0.5 * len(self._times) * np.log(2 * np.pi)
        self._offset -= len(self._times) * np.log(sigma)
        self._multip = -1 / (2.0 * sigma**2)

    def __call__(self, x):
        error = self._values - self._problem.evaluate(x)
        return np.sum(self._offset + self._multip * np.sum(error**2, axis=0))


class UnknownNoiseLogLikelihood(pints.ProblemLogLikelihood):
    """
    *Extends:* :class:`ProblemLogLikelihood`

    Calculates a log-likelihood assuming independent normally-distributed noise
    at each time point, and adds a parameter representing the standard
    deviation (sigma) of the noise on each output.

    For a noise level of ``sigma``, the likelihood becomes:

    .. math::
        L(\\theta, \sigma) = p(data | \\theta, \sigma) =
            \prod_{i=1}^N \\frac{1}{2\pi\sigma^2}\exp\left(
            -\\frac{(x_i - f_i(\\theta))^2}{2\sigma^2}\\right)

    leading to a log likelihood of

    .. math::
        \log{L(\\theta, \sigma)} =
            -\\frac{N}{2}\log{2\pi}
            -N\log{\sigma}
            -\\frac{1}{2\sigma^2}\sum_{i=1}^N{(x_i - f_i(\\theta))^2}

    Arguments:

    ``problem``
        A :class:`SingleOutputProblem` or :class`MultiOutputProblem`. For a
        single-output problem a single parameter is added, for a multi-output
        problem ``n_outputs`` parameters are added.

    """
    def __init__(self, problem):
        super(UnknownNoiseLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._dimension = problem.dimension() + self._no

        # Pre-calculate parts
        self._logn = 0.5 * len(self._times) * np.log(2 * np.pi)

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        error = self._values - self._problem.evaluate(x[:-self._no])
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))


class ScaledLogLikelihood(pints.ProblemLogLikelihood):
    """
    *Extends:* :class:`ProblemLogLikelihood`

    Calculates a log-likelihood based on a (conditional)
    :class:`ProblemLogLikelihood` divided by the number of time samples.

    The returned value will be ``(1 / n) * log_likelihood(x|problem)``, where
    ``n`` is the number of time samples multiplied by the number of outputs.

    Arguments:

    ``log_likelihood``
        A :class:`ProblemLogLikelihood`.

    This log-likelihood operates on both single and multi-output problems.
    """
    def __init__(self, log_likelihood):
        # Check arguments
        if not isinstance(log_likelihood, pints.ProblemLogLikelihood):
            raise ValueError(
                'Given log_likelihood must extend pints.ProblemLogLikelihood')

        # Call parent constructor
        super(ScaledLogLikelihood, self).__init__(log_likelihood._problem)

        # Store log-likelihood
        self._log_likelihood = log_likelihood

        # Pre-calculate parts
        self._f = 1.0 / np.product(self._values.shape)

    def __call__(self, x):
        return self._log_likelihood(x) * self._f


class SumOfIndependentLogLikelihoods(pints.LogLikelihood):
    """
    *Extends:* :class:`LogLikelihood`

    Calculates a sum of :class:`LogLikelihood` objects, all defined on the same
    parameter space.

    This is useful for e.g. Bayesian inference using a single model evaluated
    on two **independent** data sets ``D`` and ``E``. In this case,

    .. math::
        f(\\theta|D,E) &= \\frac{f(D, E|\\theta)f(\\theta)}{f(D, E)} \\\\
                       &= \\frac{f(D|\\theta)f(E|\\theta)f(\\theta)}{f(D, E)}

    Arguments:

    ``log_likelihoods``
        A sequence of :class:`LogLikelihood` objects.

    Example::

        log_likelihood = pints.SumOfIndependentLogLikelihoods([
            pints.UnknownNoiseLogLikelihood(problem1),
            pints.UnknownNoiseLogLikelihood(problem2),
        ])


    """
    def __init__(self, log_likelihoods):
        super(SumOfIndependentLogLikelihoods, self).__init__()

        # Check input arguments
        if len(log_likelihoods) < 2:
            raise ValueError(
                'SumOfIndependentLogLikelihoods requires at least two log'
                ' likelihoods.')
        for i, e in enumerate(log_likelihoods):
            if not isinstance(e, pints.LogLikelihood):
                raise ValueError(
                    'All objects passed to SumOfIndependentLogLikelihoods must'
                    ' be instances of pints.LogLikelihood (failed on argument '
                    + str(i) + ').')
        self._log_likelihoods = list(log_likelihoods)

        # Get and check dimension
        i = iter(self._log_likelihoods)
        self._dimension = next(i).dimension()
        for e in i:
            if e.dimension() != self._dimension:
                raise ValueError(
                    'All log-likelihoods passed to'
                    ' SumOfIndependentLogLikelihoods must have same'
                    ' dimension.')

    def dimension(self):
        """ See :meth:`LogPDF.dimension()`. """
        return self._dimension

    def __call__(self, x):
        total = 0
        for e in self._log_likelihoods:
            total += e(x)
        return total
