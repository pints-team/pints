#
# Sub-module containing ABC inference routines
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
import pints
import numpy as np

class ABCSampler(pints.Loggable, pints.TunableMethod):
    """
    Abstract base class for ABC methods.
    All ABC samplers implement the :class:`pints.Loggable` and
    :class:`pints.TunableMethod` interfaces.
    """
    def name(self):
        """
        Returns this method's full name.
        """
        raise NotImplementedError

    def ask(self):
        """
        Returns a parameter vector sampled from a prior
        """
        raise NotImplementedError

    def tell(self, fx):
        """
        Performs an iteration of the ABC-rejection algorithm, using the parameters specified by ask.

        Returns the accepted parameter values, or ``None`` to indicate
        that no parameters were accepted (tell allows for multiple evaluations per iteration).

        """
        raise NotImplementedError

class ABCController(object):
    """
    Samples from a :class:`pints.LogPrior`

    Properties related to the number of iterations, parallelisation,
    threshold, and number of parameters to sample can be set directly on the ``ABCController`` object, e.g.::

        abc.set_max_iterations(1000)

    Finally, to run an ABC routine, call::

        posterior_estimate = abc.run()

    Constructor arguments:
    ``error_measure``
        An error measure to evaluate on a problem, given a forward model, simulated and observed data, and times

    ``log_prior``
        A :class:`LogPrior` function from which parameter values are sampled

    ``method``
        The class of :class:`ABCSampler` to use. If no method is specified,
        :class:`ABCRejection` is used.
    """
    def __init__(self, error_measure, log_prior, method=None):

        # Store function
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given function must extend pints.LogPrior')
        self._log_prior = log_prior

        # Check error_measure
        # if not isinstance(error_measure, pints.ErrorMeasure):
            # raise ValueError('Given error_measure must extend pints.ErrorMeasure')
        self._error_measure = error_measure

        # Check if number of parameters from prior matches that of error measure
        if self._log_prior.n_parameters() != self._error_measure.n_parameters():
            raise ValueError('Number of parameters in prior must match number of parameters in model')

        # Get number of parameters
        self._n_parameters = self._log_prior.n_parameters()

        # Don't check initial standard deviation: done by samplers!

        # Set default method
        if method is None:
            method = pints.ABCRejection
        else:
            try:
                ok = issubclass(method, pints.ABCSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError('Given method must extend pints.ABCSampler.')

        # Parallelisation
        self._parallel = False
        self._n_workers = 1
        self.set_parallel()

        #
        # Stopping criteria
        #

        # Maximum iterations
        self._max_iterations = None
        self.set_max_iterations()

        # Threshold value
        self._threshold = 1.5
        self.set_threshold()

        # Number of parameter samples in posterior estimate
        self._n_target = 500
        self.set_n_target()

        # Number of draws per iteration
        self._n_draws = 1
        self.set_n_draws()

        # TODO: Add more stopping criteria

        # Create sampler(s)

        # Using n individual samplers (Note that it is possible to have
        # _n_samplers=1)
        self._samplers = method(log_prior, self._threshold)

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations()`.
        """
        return self._max_iterations

    def threshold(self):
        """
        Returns the threshold below which values are accepted
        """
        return self._threshold

    def n_target(self):
        """
        Returns the target number of samples to obtain in the estimated posterior
        """
        return self._n_target

    def n_draws(self):
        """
        Returns the number of draws per iteration
        """
        return self._n_draws

    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False

    def run(self):
        """
        Runs the ABC sampler and returns the accepted parameter values which make up the posterior estimate
        """
        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Iteration and evaluation counting
        iteration = 0
        evaluations = 0

        # Choose method to evaluate
        f = self._error_measure

        # Create evaluator object
        if self._parallel:
            # Use n_workers
            n_workers = self._n_workers
            evaluator = pints.ParallelEvaluator(f, n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Initialize samples
        samples = []

        # Start sampling
        timer = pints.Timer()

        while len(samples) < self._n_target:

            # Get parameter values sampled from prior
            xs = self._samplers.ask(self._n_draws)

            # Simulate datasets based on sampled parameters and calculate root-mean-squared-error
            fxs = evaluator.evaluate(xs)

            # Update evaluation count
            evaluations += len(fxs)

            # Check RMSE values against a threshold and accept parameters below the threshold
            accepted_vals = self._samplers.tell(fxs)

            # Add new accepted parameters to the estimated posterior
            samples.extend(accepted_vals)

            # Update iteration count
            iteration += 1

            #
            # Check stopping criteria
            #

            # Maximum number of iterations
            if (self._max_iterations is not None and
                    iteration >= self._max_iterations):

                halt_message = ('Halting: Maximum number of iterations ('
                                + str(iteration) + ') reached.')

                print(halt_message)
                break

        timer.time()
        return samples

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

    def set_threshold(self, threshold=1.5):
        """
        Sets a threshold below which to accept simulated values
        """
        if threshold <= 0:
            raise ValueError('Threshold must be positive')
        self._threshold = threshold

    def set_n_target(self, n_target=500):
        """
        Sets a target number of samples
        """
        self._n_target = n_target

    def set_n_draws(self, n_draws=1):
        """
        Sets the number of draws per iteration
        """
        self._n_draws = n_draws

    def set_parallel(self, parallel=False):
        """
        Enables/disables parallel evaluation.

        If ``parallel=True``, the method will run using a number of worker
        processes equal to the detected cpu core count. The number of workers
        can be set explicitly by setting ``parallel`` to an integer greater
        than 0.
        Parallelisation can be disabled by setting ``parallel`` to ``0`` or
        ``False``.
        """
        if parallel is True:
            self._parallel = True
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1


