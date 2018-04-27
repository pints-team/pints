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


class LogPDF(object):
    """
    Callable class that represents the natural logarithm of a (not necessarily
    normalised) probability density function (PDF).

    When called with a vector argument ``x``, a ``LogPDF`` returns a value
    ``fx`` where ``fx = log(f(x))`` and ``f(x)`` is an unnormalised PDF.

    The size of the argument ``x`` is given by :meth:`n_parameters()`.
    """
    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the dimension of the space this :class:`LogPDF` is defined
        over.
        """
        raise NotImplementedError


class LogPDFS1(object):
    """
    Callable class that provides not just a :class:`LogPDF`, but also it's
    derivatives with respect to the parameters.

    When called with a vector argument ``x``, a ``LogPDFS1`` returns a
    tuple ``(fx, dfx)`` where ``fx = log(f(x))`` and ``f(x)`` is an
    unnormalised PDF, and where ``dfx = d/dp log(f(x))``.

    The size of the argument ``x`` is given by :meth:`n_parameters()`.
    """
    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the dimension of the space this :class:`LogPDF` is defined
        over.
        """
        raise NotImplementedError


class LogPrior(LogPDF):
    """
    *Extends:* :class:`LogPDF`

    Represents the natural logarithm ``log(f(theta))`` of a known probability
    density function ``f(theta)``.

    Priors are *usually* normalised (i.e. the integral ``f(theta)`` over all
    points ``theta`` in parameter space sums to 1), but this is not a strict
    requirement.
    """
    def sample(self, n=1):
        """
        Returns ``n`` random samples from the underlying prior distribution.

        The returned value is a numpy array with shape ``(n, d)`` where ``n``
        is the requested number of samples, and ``d`` is the dimension of the
        prior.

        Note: This method is optional, in the sense that only a subsets of
        inference methods require it.
        """
        raise NotImplementedError


class LogLikelihood(LogPDF):
    """
    *Extends:* :class:`LogPDF`

    Represents a log-likelihood defined on a parameter space.
    """


class LogLikelihoodS1(LogPDFS1):
    """
    *Extends:* :class:`LogPDFS1`

    Represents a log-likelihood defined on a parameter space, for which the
    derivatives of the log-likelihood with respect to the parameters are known.
    """


class ProblemLogLikelihood(LogLikelihood):
    """
    *Extends:* :class:`LogLikelihood`

    Represents a log-likelihood on a problem's parameter space, used to
    indicate the likelihood of an observed (fixed) time-series given a
    particular parameter set (variable).

    Arguments:

    ``problem``
        The time-series problem this log-likelihood is defined for. Must extend
        :class:`SingleOutputProblem` or :class:`MultiOutputProblem`.

    """
    def __init__(self, problem):
        super(LogLikelihood, self).__init__()

        # Check problem
        self._problem = problem
        if not (isinstance(problem, pints.SingleOutputProblem) or
                isinstance(problem, pints.MultiOutputProblem)):
            raise TypeError(
                'Expecting a single or multi-output problem without'
                ' sensitivities.')

        # Cache some problem variables
        self._values = problem.values()
        self._times = problem.times()
        self._n_parameters = problem.n_parameters()

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters


class ProblemLogLikelihoodS1(LogLikelihoodS1):
    """
    *Extends:* :class:`LogLikelihoodS1`

    Represents a log-likelihood on a problem's parameter space, used to
    indicate the likelihood of an observed (fixed) time-series given a
    particular parameter set (variable).

    Unlike the :class:`ProblemLogLikelihood` class, this class expects a
    problem that can provide first-order derivatives with respect to the
    parameter vector.

    Arguments:

    ``problem``
        The time-series problem this log-likelihood is defined for. Must extend
        :class:`SingleOutputProblemS1` or :class:`MultiOutputProblemS1`.

    """
    def __init__(self, problem):
        super(LogLikelihood, self).__init__()

        # Check problem
        self._problem = problem
        if not (isinstance(problem, pints.SingleOutputProblemS1) or
                isinstance(problem, pints.MultiOutputProblemS1)):
            raise TypeError(
                'Expecting a single or multi-output problem without'
                ' sensitivities.')

        # Cache some problem variables
        self._values = problem.values()
        self._times = problem.times()
        self._n_parameters = problem.n_parameters()

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters


class LogPosterior(LogPDF):
    """
    *Extends:* :class:`LogPDF`

    Represents the sum of a :class:`LogLikelihood` and a :class:`LogPrior`
    defined on the same parameter space.

    As an optimisation, if the :class:`LogPrior` evaluates as `-inf` for a
    particular point in parameter space, the corresponding
    :class:`LogLikelihood` won't be evaluated.

    Arguments:

    ``log_likelihood``
        A :class:`LogLikelihood`, defined on the same parameter space.
    ``log_prior``
        A :class:`LogPrior`, representing prior knowledge of the parameter
        space.


    """
    def __init__(self, log_likelihood, log_prior):
        super(LogPosterior, self).__init__()

        # Check arguments
        if not isinstance(log_prior, LogPrior):
            raise ValueError(
                'Given prior must extend pints.LogPrior.')
        if not isinstance(log_likelihood, LogLikelihood):
            raise ValueError(
                'Given log_likelihood must extend pints.LogLikelihood.')

        # Check number of parameters
        self._n_parameters = log_prior.n_parameters()
        if log_likelihood.n_parameters() != self._n_parameters:
            raise ValueError(
                'Given log_prior and log_likelihood must have same number of'
                ' parameters.')

        # Store prior and likelihood
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood

        # Store -inf, for later use
        self._minf = -float('inf')

    def __call__(self, x):
        # Evaluate log-prior first, assuming this is very cheap
        log_prior = self._log_prior(x)
        if log_prior == self._minf:
            return self._minf
        return log_prior + self._log_likelihood(x)

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters

