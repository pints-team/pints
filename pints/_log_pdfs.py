#
# Main Log PDF functions
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np


class LogPDF(object):
    """
    Represents the natural logarithm of a (not necessarily normalised)
    probability density function (PDF).

    All :class:`LogPDF` types are callable: when called with a vector argument
    ``x`` they return some value ``log(f(x))`` where ``f(x)`` is an
    unnormalised PDF. The size of the argument ``x`` is given by
    :meth:`n_parameters()`. In PINTS, all parameters must be continuous and
    real.

    All subclasses of ``LogPDF`` should provide an implementation of
    :meth:`__call__` and :meth:`n_parameters`. Providing :meth:`evaluateS1` is
    optional.
    """
    def __call__(self, x):
        """ Evaluates this LogPDF for parameters ``x``. """
        raise NotImplementedError

    def evaluateS1(self, x):
        """
        Evaluates this LogPDF, and returns the result plus the partial
        derivatives of the result with respect to the parameters.

        The returned data is a tuple ``(L, L')`` where ``L`` is a scalar value
        and ``L'`` is a sequence of length ``n_parameters``.

        Note that the derivative returned is of the log-pdf, so
        ``L' = d/dp log(f(p))``, evaluated at ``p=x``.

        *This is an optional method that is not always implemented.*
        """
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the dimension of the space this :class:`LogPDF` is defined
        over.
        """
        raise NotImplementedError


class LogPrior(LogPDF):
    """
    Represents the natural logarithm ``log(f(theta))`` of a known probability
    density function ``f(theta)``.

    Priors are *usually* normalised (i.e. the integral ``f(theta)`` over all
    points ``theta`` in parameter space sums to 1), but this is not a strict
    requirement.

    Extends :class:`LogPDF`.
    """
    def cdf(self, x):
        """
        Returns the cumulative density function at point(s) ``x``.

        ``x`` should be an ``n x d`` array, where ``n`` is the number of input
        samples and ``d`` is the dimension of the parameter space.
        """
        raise NotImplementedError

    def convert_from_unit_cube(self, u):
        """
        Converts samples ``u`` uniformly drawn from the unit cube into those
        drawn from the prior space, typically by transforming using
        :meth:`LogPrior.icdf()`.

        ``u`` should be an ``n x d`` array, where ``n`` is the number of input
        samples and ``d`` is the dimension of the parameter space.
        """
        return self.icdf(u)

    def convert_to_unit_cube(self, x):
        """
        Converts samples from the prior ``x`` to be drawn uniformly from the
        unit cube, typically by transforming using :meth:`LogPrior.cdf()`.

        ``x`` should be an ``n x d`` array, where ``n`` is the number of input
        samples and ``d`` is the dimension of the parameter space.
        """
        return self.cdf(x)

    def icdf(self, p):
        """
        Returns the inverse cumulative density function at cumulative
        probability/probabilities ``p``.

        ``p`` should be an ``n x d`` array, where ``n`` is the number of input
        samples and ``d`` is the dimension of the parameter space.
        """
        raise NotImplementedError

    def mean(self):
        """
        Returns the analytical value of the expectation of a random variable
        distributed according to this :class:`LogPDF`.
        """
        raise NotImplementedError

    def sample(self, n=1):
        """
        Returns ``n`` random samples from the underlying prior distribution.

        The returned value is a NumPy array with shape ``(n, d)`` where ``n``
        is the requested number of samples, and ``d`` is the dimension of the
        prior.
        """
        raise NotImplementedError


class LogLikelihood(LogPDF):
    """
    Represents a log-likelihood defined on a parameter space.

    This class adds no new functionality, but exists to indicate when a LogPDF
    represents the probability of a data set *given* a set of parameters,
    rather than a probability of those parameters.

    *Extends:* :class:`LogPDF`
    """


class ProblemLogLikelihood(LogLikelihood):
    """
    Represents a log-likelihood on a problem's parameter space, used to
    indicate the likelihood of an observed (fixed) time-series given a
    particular parameter set (variable).

    Extends :class:`LogLikelihood`.

    Parameters
    ----------
    problem
        The time-series problem this log-likelihood is defined for.

    """
    def __init__(self, problem):
        super().__init__()
        self._problem = problem
        # Cache some problem variables
        self._values = problem.values()
        self._times = problem.times()
        self._n_parameters = problem.n_parameters()

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters

    def problem(self):
        return self._problem


class LogPosterior(LogPDF):
    """
    Represents the sum of a :class:`LogLikelihood` and a :class:`LogPrior`
    defined on the same parameter space.

    As an optimisation, if the :class:`LogPrior` evaluates as `-inf` for a
    particular point in parameter space, the corresponding :class:`LogPDF` will
    not be evaluated.

    Extends :class:`LogPDF`.

    Parameters
    ----------
    log_likelihood
        A :class:`LogLikelihood`, defined on the same parameter space.
    log_prior
        A :class:`LogPrior`, representing prior knowledge of the parameter
        space.
    """
    def __init__(self, log_likelihood, log_prior):
        super().__init__()

        # Check arguments
        if not isinstance(log_prior, LogPrior):
            raise ValueError(
                'Given prior must extend pints.LogPrior.')
        if not isinstance(log_likelihood, LogLikelihood):
            raise ValueError(
                'Given log_likelihood must extend pints.LogLikelihood.')

        # Check dimensions
        self._n_parameters = log_prior.n_parameters()
        if log_likelihood.n_parameters() != self._n_parameters:
            raise ValueError(
                'Given log_prior and log_likelihood must have same dimension.')

        # Store prior and likelihood
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood

        # Store -inf, for later use
        self._minf = -np.inf

    def __call__(self, x):
        # Evaluate log-prior first, assuming this is very cheap
        log_prior = self._log_prior(x)
        if log_prior == self._minf:
            return self._minf
        return log_prior + self._log_likelihood(x)

    def evaluateS1(self, x):
        """
        Evaluates this LogPDF, and returns the result plus the partial
        derivatives of the result with respect to the parameters.

        The returned data has the shape ``(L, L')`` where ``L`` is a scalar
        value and ``L'`` is a sequence of length ``n_parameters``.

        *This method only works if the underlying :class:`LogPDF` and
        :class:`LogPrior` implement the optional method
        :meth:`LogPDF.evaluateS1()`!*
        """
        #TODO: Is there an optimisation to be made here?
        a, da = self._log_prior.evaluateS1(x)
        b, db = self._log_likelihood.evaluateS1(x)
        return a + b, da + db

    def log_likelihood(self):
        """ Returns the :class:`LogLikelihood` used by this posterior. """
        return self._log_likelihood

    def log_prior(self):
        """ Returns the :class:`LogPrior` used by this posterior. """
        return self._log_prior

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters

