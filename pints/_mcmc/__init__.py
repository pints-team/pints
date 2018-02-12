#
# Sub-module containing MCMC inference routines
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class MCMCSampler(object):
    """
    Abstract base class for (single or multi-chain) MCMC methods.
    """
    def name(self):
        """
        Returns this method's full name.
        """
        raise NotImplementedError


class SingleChainMCMC(MCMCSampler):
    """
    *Extends:* :class:`MCMCSampler`

    Abstract base class for MCMC methods that generate a single markov chain,
    via an ask-and-tell interface.

    Arguments:

    ``x0``
        An starting point in the parameter space.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the covariance
        of the distribution to estimate, around ``x0``.

    """
    def __init__(self, x0, sigma0=None):

        # Check initial position
        self._x0 = pints.vector(x0)

        # Get dimension
        self._dimension = len(self._x0)

        # Check initial standard deviation
        if sigma0 is None:
            # Get representative parameter value for each parameter
            self._sigma0 = np.abs(self._x0)
            self._sigma0[self._sigma0 == 0] = 1
            # Use to create diagonal matrix
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0)
            if np.product(self._sigma0.shape) == self._dimension:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._dimension,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape(
                    (self._dimension, self._dimension))

    def ask(self):
        """
        Returns a position in the search space to evaluate.
        """
        raise NotImplementedError

    def tell(self, fx):
        """
        Performs an iteration of the MCMC algorithm, using the evaluation
        ``fx`` of the point previously specified by ``ask``. Returns the next
        sample in the chain.
        """
        raise NotImplementedError


class SingleChainAdaptiveMCMC(SingleChainMCMC):
    """
    *Extends:* :class:`SingleChainMCMC`

    Abstract base class for adaptive single chain MCMC methods.

    ``x0``
        An starting point in the parameter space.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the covariance
        of the distribution to estimate, around ``x0``.

    """
    def __init__(self, x0, sigma0=None):
        super(SingleChainAdaptiveMCMC, self).__init__(x0, sigma0)

        # Adaptation enabled/disabled
        self._adaptation = True

    def adaptation(self):
        """
        Returns ``True`` if this sampler is in adaptive mode.
        """
        return self._adaptation

    def set_adaptation(self, adaptation):
        """
        Switches adaptation on/off for this sampler.

        Arguments:

        ``adaptation``
            A boolean specifying whether this method should (``True``) or
            should not (``False``) run in adaptive mode for the next
            iterations.

        """
        self._adaptation = bool(adaptation)


class MultiChainMCMC(MCMCSampler):
    """
    *Extends:* :class:`MCMCSampler`

    Abstract base class for MCMC methods that generate multiple markov chains,
    via an ask-and-tell interface.

    Arguments:

    ``chains``
        The number of MCMC chains to generate.
    ``x0``
        A sequence of starting points. Can be a list of lists, a 2-dimensional
        array, or any other structure such that ``x0[i]`` is the starting point
        for chain ``i``.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.

    """
    def __init__(self, chains, x0, sigma0=None):

        # Check number of chains
        self._chains = int(chains)
        if self._chains < 1:
            raise ValueError('Number of chains must be at least 1.')

        # Check initial position(s)
        if len(x0) != chains:
            raise ValueError(
                'Number of initial positions must be equal to number of'
                ' chains.')
        self._x0 = np.array([pints.vector(x) for x in x0])
        self._x0.setflags(write=False)

        # Get dimension
        self._dimension = len(self._x0[0])

        # Check initial points all have correct dimension
        if not all([len(x) == self._dimension for x in self._x0]):
            raise ValueError('All initial points must have same dimension.')

        # Check initial standard deviation
        if sigma0 is None:
            # Get representative parameter value for each parameter
            self._sigma0 = np.max(np.abs(self._x0), axis=0)
            self._sigma0[self._sigma0 == 0] = 1
            # Use to create diagonal matrix
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0, copy=True)
            if np.product(self._sigma0.shape) == self._dimension:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._dimension,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape(
                    (self._dimension, self._dimension))

    def ask(self):
        """
        Returns a sequence of positions in the search space to evaluate.
        """
        raise NotImplementedError

    def tell(self, fxs):
        """
        Performs an iteration of the MCMC algorithm, using the evaluations
        ``fxs`` of the points previously specified by ``ask``. Returns the next
        samples in the chains.
        """
        raise NotImplementedError


class MCMCSampling(object):
    """
    Samples from a :class:`pints.LogPDF` using a Markov Chain Monte Carlo
    (MCMC) method.

    Arguments:

    ``log_pdf``
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    ``chains``
        The number of MCMC chains to generate.
    ``x0``
        A sequence of starting points. Can be a list of lists, a 2-dimensional
        array, or any other structure such that ``x0[i]`` is the starting point
        for chain ``i``.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.
    ``method``
        The class of :class:`MCMCSampler` to use. If no method is specified,
        :class:`AdaptiveCovarianceMCMC` is used.

    """
    def __init__(self, log_pdf, chains, x0, sigma0=None, method=None):

        # Store function
        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError('Given function must extend pints.LogPDF')
        self._log_pdf = log_pdf

        # Get dimension
        self._dimension = self._log_pdf.dimension()

        # Check number of chains
        self._chains = int(chains)
        if self._chains < 1:
            raise ValueError('Number of chains must be at least 1.')

        # Check initial position(s): Most checking is done by samplers!
        if len(x0) != chains:
            raise ValueError(
                'Number of initial positions must be equal to number of'
                ' chains.')
        if not all([len(x) == self._dimension for x in x0]):
            raise ValueError(
                'All initial positions must have the same dimension as the'
                ' given LogPDF.')

        # Don't check initial standard deviation: done by samplers!

        # Set default method
        if method is None:
            method = pints.AdaptiveCovarianceMCMC
        elif not issubclass(method, pints.MCMCSampler):
            raise ValueError('Given method must extend pints.MCMCSampler.')

        # Using single chain samplers?
        self._single_chain = issubclass(method, pints.SingleChainMCMC)

        # Create sampler(s)
        if self._single_chain:
            # Using n individual samplers (Note that it is possible to have
            # _single_chain=True and _n_samplers=1)
            self._n_samplers = self._chains
            self._samplers = [method(x, sigma0) for x in x0]
        else:
            # Using a single sampler that samples multiple chains
            self._n_samplers = 1
            self._samplers = [method(self._chains, x0, sigma0)]

        # Print info to console
        self._verbose = True

        # Parallelisation
        self._parallel = None
        self.set_parallel()

        # Adaptive methods
        self._adaptation_free_iterations = None
        self.set_adaptation_free_iterations()

        #
        # Stopping criteria
        #

        # Maximum iterations
        self._max_iterations = None
        self.set_max_iterations()

        #TODO: Add more stopping criteria

    def adaptation_free_iterations(self):
        """
        For adaptive methods, returns the number of adaptation free iterations
        to use at the start of the run. Runs ``None`` for non-adaptive methods
        or when no adaptation free period is configured
        """
        return self._adaptation_free_iterations

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations`.
        """
        return self._max_iterations

    def parallel(self):
        """
        Returns ``True`` if this sampling is set to use parallelisation.
        """
        return self._parallel

    def run(self):
        """
        Runs the MCMC sampler(s) and returns a number of markov chains, each
        representing the distribution of the given log-pdf.
        """
        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Iterations
        iteration = 0

        # Create evaluator object
        if self._parallel:
            self._evaluator = pints.ParallelEvaluator(self._log_pdf)
        else:
            self._evaluator = pints.SequentialEvaluator(self._log_pdf)

        # Set up progress reporting
        next_message = 0
        message_warm_up = 3
        message_interval = 20

        # Print configuration
        if self._verbose:
            print('Using ' + str(self._samplers[0].name()))
            if self._parallel:
                print('Running in parallel mode.')
            else:
                print('Running in sequential mode.')

        # Create chains
        #TODO Pre-allocate?
        #TODO Thinning
        #TODO Advanced logging
        chains = []

        # Start searching
        running = True
        while running:
            # Get points
            if self._single_chain:
                xs = [sampler.ask() for sampler in self._samplers]
            else:
                xs = self._samplers[0].ask()

            # Calculate scores
            fxs = self._evaluator.evaluate(xs)

            # Perform iteration(s)
            if self._single_chain:
                samples = np.array([
                    s.tell(fxs[i]) for i, s in enumerate(self._samplers)])
            else:
                samples = self._samplers[0].tell(fxs)
            chains.append(samples)

            # Show progress in verbose mode:
            if self._verbose and iteration >= next_message:
                # TODO: Add some sort of status printing here
                print('' + str(iteration))
                if iteration < message_warm_up:
                    next_message = iteration + 1
                else:
                    next_message = message_interval * (
                        1 + iteration // message_interval)

            # Update iteration count
            iteration += 1

            #
            # Check stopping criteria
            #

            # Maximum number of iterations
            if (self._max_iterations is not None and
                    iteration >= self._max_iterations):
                running = False
                if self._verbose:
                    print('Halting: Maximum number of iterations ('
                          + str(iteration) + ') reached.')

            #TODO Add more stopping criteria

            #
            # Adaptive methods
            #

            # Start adapting?
            if iteration == self._adaptation_free_iterations:
                for sampler in self._samplers:
                    sampler.set_adaptation(True)

        # Swap axes in chains, to get indices
        #  [chain, iteration, parameter]
        chains = np.array(chains)
        chains = chains.swapaxes(0, 1)

        # Return generated chains
        return chains

    def set_adaptation_free_iterations(self, iterations=200):
        """
        For adaptive methods, sets the number of adaptation free ``iterations``
        at the start of a run.
        Returns ``False`` if the chosen method doesn't support adaptation.
        """
        self._adaptation_free_iterations = None

        # Check input
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError(
                'Number of adaptation free iterations cannot be negative.')

        # Attempt to configure samplers
        try:
            for sampler in self._samplers:
                sampler.set_adaptation(False)
        except AttributeError:
            # No adaptation method? Return False
            return False

        # Store number of iterations, return True
        self._adaptation_free_iterations = iterations
        return True

    def set_max_iterations(self, iterations=10000):
        """
        Adds a stopping criterion, allowing the routine to halt after the
        given number of `iterations`.

        This criterion is enabled by default. To disable it, use
        `set_max_iterations(None)`.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError(
                    'Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

    def set_parallel(self, parallel=False):
        """
        Enables/disables parallel evaluation.
        """
        self._parallel = bool(parallel)

    def set_verbose(self, value):
        """
        Enables or disables verbose mode for this MCMC sampling. In verbose
        mode, lots of output is generated during a run.
        """
        self._verbose = bool(value)

    def verbose(self):
        """
        Returns ``True`` if the MCMC sampling is set to run in verbose mode.
        """
        return self._verbose


def mcmc_sample(log_pdf, chains, x0, sigma0=None, method=None):
    """
    Sample from a :class:`pints.LogPDF` using a Markov Chain Monte Carlo
    (MCMC) method.

    Arguments:

    ``log_pdf``
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    ``chains``
        The number of MCMC chains to generate.
    ``x0``
        A sequence of starting points. Can be a list of lists, a 2-dimensional
        array, or any other structure such that ``x0[i]`` is the starting point
        for chain ``i``.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.
    ``method``
        The class of :class:`MCMCSampler` to use. If no method is specified,
        :class:`AdaptiveCovarianceMCMC` is used.
    """
    return MCMCSampling(log_pdf, chains, x0, sigma0, method).run()
