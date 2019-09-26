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
import pints


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
        Performs an iteration of the ABC-rejection algorithm, using the
        parameters specified by ask.

        Returns the accepted parameter values, or ``None`` to indicate
        that no parameters were accepted (tell allows for multiple evaluations
        per iteration).

        """
        raise NotImplementedError


class ABCController(object):
    """
    Samples from a :class:`pints.LogPrior`

    Properties related to the number of iterations, parallelisation,
    threshold, and number of parameters to sample can be set directly on the
    ``ABCController`` object, e.g.::

        abc.set_max_iterations(1000)

    Finally, to run an ABC routine, call::

        posterior_estimate = abc.run()

    Constructor arguments:
    ``error_measure``
        An error measure to evaluate on a problem, given a forward model,
        simulated and observed data, and times

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
        # raise ValueError('Given error_measure must extend
        # pints.ErrorMeasure')
        self._error_measure = error_measure

        # Check if number of parameters from prior matches that of error
        # measure
        if self._log_prior.n_parameters() != \
                self._error_measure.n_parameters():
            raise ValueError('Number of parameters in prior must match number '
                             'of parameters in model')

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

    def n_target(self):
        """
        Returns the target number of samples to obtain in the estimated
        posterior.
        """
        return self._n_target

    def n_draws(self):
        """
        Returns the number of draws per iteration.
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
        Runs the ABC sampler.
        """
        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Iteration and evaluation counting
        iteration = 0
        evaluations = 0
        accepted_count = 0

        # Choose method to evaluate
        f = self._error_measure

        # Create evaluator object
        if self._parallel:
            # Use at most n_workers workers
            n_workers = min(self._n_workers, self._chains)
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Initial phase
        if self._needs_initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)

        # Write chains to disk
        chain_loggers = []
        if self._chain_files:
            for filename in self._chain_files:
                cl = pints.Logger()
                cl.set_stream(None)
                cl.set_filename(filename, True)
                for k in range(self._n_parameters):
                    cl.add_float('p' + str(k))
                chain_loggers.append(cl)

        # Write evaluations to disk
        eval_loggers = []
        if self._evaluation_files:
            # Bayesian inference on a log-posterior? Then separate out the
            # prior so we can calculate the loglikelihood
            prior = None
            if isinstance(self._log_pdf, pints.LogPosterior):
                prior = self._log_pdf.log_prior()

            # Set up loggers
            for filename in self._evaluation_files:
                cl = pints.Logger()
                cl.set_stream(None)
                cl.set_filename(filename, True)
                if prior:
                    # Logposterior in first column, to be consistent with the
                    # non-bayesian case
                    cl.add_float('logposterior')
                    cl.add_float('loglikelihood')
                    cl.add_float('logprior')
                else:
                    cl.add_float('logpdf')
                eval_loggers.append(cl)

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                print('Using ' + str(self._samplers[0].name()))
                print('Generating ' + str(self._chains) + ' chains.')
                if self._parallel:
                    print('Running in parallel with ' + str(n_workers) +
                          ' worker processess.')
                else:
                    print('Running in sequential mode.')
                if self._chain_files:
                    print(
                        'Writing chains to ' + self._chain_files[0] + ' etc.')
                if self._evaluation_files:
                    print(
                        'Writing evaluations to ' + self._evaluation_files[0]
                        + ' etc.')

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            max_iter_guess = max(self._max_iterations or 0, 10000)
            max_eval_guess = max_iter_guess * self._chains
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            for sampler in self._samplers:
                sampler._log_init(logger)
            logger.add_time('Time m:s')

        # Start sampling
        timer = pints.Timer()
        running = True

        # Initialize samples
        samples = []

        while running:
            # Sample until a given sample is accepted
            xs = self._samplers.ask(self._n_draws)
            fxs = evaluator.evaluate(xs)
            evaluations += self._n_draws
            accepted_vals = self._samplers.tell(fxs)
            if accepted_vals is not None:
                accepted_count += len(accepted_vals)
            while accepted_vals is None:
                xs = self._samplers.ask(self._n_draws)
                fxs = evaluator.evaluate(xs)
                accepted_vals = self._samplers.tell(fxs)
                accepted_count += len(accepted_vals)

            # Add new accepted parameters to the estimated posterior
            samples.append(accepted_vals)

            # Update iteration count
            iteration += 1

            # Show progress
            if logging and iteration >= next_message:
                # Log state
                logger.log(iteration, evaluations)
                for sampler in self._samplers:
                    sampler._log_write(logger)
                logger.log(timer.time())

                # Choose next logging point
                if iteration < self._message_warm_up:
                    next_message = iteration + 1
                else:
                    next_message = self._message_interval * (
                        1 + iteration // self._message_interval)

            # Check requested number of samples
            if (self._max_iterations is not None and
                    iteration >= self._max_iterations):
                running = False
                halt_message = ('Halting: Maximum number of iterations ('
                                + str(iteration) + ') reached.')
            elif accepted_count >= self._n_target:
                running = False
                halt_message = ('Halting: target number of samples ('
                                + str(accepted_count) + ') reached.')

            # Log final state and show halt message
            if logging:
                logger.log(iteration, evaluations)
                for sampler in self._samplers:
                    sampler._log_write(logger)
                logger.log(timer.time())
                if self._log_to_screen:
                    print(halt_message)

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
