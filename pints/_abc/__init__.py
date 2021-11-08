#
# Sub-module containing ABC inference routines
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np

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
        Returns a parameter vector sampled from the LogPrior.
        """
        raise NotImplementedError

    def tell(self, x):
        """
        Performs an iteration of the ABC algorithm, using the
        parameters specified by ask.
        Returns the accepted parameter values, or ``None`` to indicate
        that no parameters were accepted (tell allows for multiple evaluations
        per iteration).
        """
        raise NotImplementedError



class ABCController(object):
    """
    Samples from a :class:`pints.LogPrior`.
    
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
    :class:`RejectionABC` is used.
    """

    def __init__(self, error_measure, log_prior, method=None):

        # Store function
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given function must extend pints.LogPrior.')
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
                             'of parameters in model.')

        # Get number of parameters
        self._n_parameters = self._log_prior.n_parameters()

        # Set rejection ABC as default method
        if method is None:
            method = pints.RejectionABC
        else:
            try:
                ok = issubclass(method, ABCSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError('Given method must extend ABCSampler.')

        # Initialisation
        self._parallel = False
        self._n_workers = 1
        self._max_iterations = 10000
        self._nr_samples = 500
        self._sampler = method(log_prior)
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()
        self._acceptance_rate = 0


    def set_log_interval(self, iters=20, warm_up=3):
        """
        Changes the frequency with which messages are logged.

        Arguments:
        
        ``interval``
            A log message will be shown every ``iters`` iterations.
        ``warm_up``
            A log message will be shown every iteration, for the first
            ``warm_up`` iterations.

        """
        iters = int(iters)
        if iters < 1:
            raise ValueError("Interval must be greater than 0.")

        warm_up = max(0, int(warm_up))
        self._message_interval = iters
        self._message_warm_up = warm_up

    def set_log_to_file(self, filename=None, csv=False):
        """
        Enables progress logging to file when a filename is passed in, disables
        it if ``filename`` is ``False`` or ``None``.

        The argument ``csv`` can be set to ``True`` to write the file in comma
        separated value (CSV) format. By default, the file contents will be
        similar to the output on screen.
        """
        if filename:
            self._log_filename = str(filename)
            self._log_csv = True if csv else False
        else:
            self._log_filename = None
            self._log_csv = False

    def set_log_to_screen(self, enabled):
        """
        Enables or disables progress logging to screen.
        """
        self._log_to_screen = True if enabled else False

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations()`.
        """
        return self._max_iterations

    def nr_samples(self):
        """
        Returns the target number of samples to obtain in the estimated
        posterior.
        """
        return self._nr_samples

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
        if self._max_iterations is None:
            raise ValueError("At least one stopping criterion must be set.")

        has_stopping_criterion = self._max_iterations

        # Iteration and evaluation counting
        iteration = 0
        evaluations = 0
        accepted_count = 0

        # Choose method to evaluate
        f = self._error_measure

        # Create evaluator
        if self._parallel:
            n_workers = self._n_workers
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                print('Using ' + str(self._sampler.name()))
                if self._parallel:
                    print('Running in parallel with ' + str(n_workers) +
                          ' worker processess.')
                else:
                    print('Running in sequential mode.')

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            max_iter_guess = max(self._max_iterations or 0, 10000)
            max_eval_guess = max_iter_guess
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            logger.add_float('Acceptance rate')
            self._sampler._log_init(logger)
            logger.add_time('Time m:s')

        # Start sampling
        timer = pints.Timer()
        running = True

        samples = []
        # Sample until we find an acceptable sample
        while running:
            accepted_vals = None
            while accepted_vals is None:
                # Get points from prior
                xs = self._sampler.ask(self._n_workers)

                # Simulate and get error
                fxs = evaluator.evaluate(xs)
                evaluations += self._n_workers

                # Tell sampler errors and get list of acceptable parameters
                accepted_vals = self._sampler.tell(fxs)

            accepted_count += len(accepted_vals)
            for val in accepted_vals:
                samples.append(val)
            
            iteration += 1
            
             # Log progress
            if logging and iteration >= next_message:
                # Log state
                logger.log(iteration, evaluations, (
                           accepted_count / evaluations))
                self._sampler._log_write(logger)
                logger.log(timer.time())

                # Choose next logging point
                if iteration < self._message_warm_up:
                    next_message = iteration + 1
                else:
                    next_message = self._message_interval * (
                        1 + iteration // self._message_interval)
            
            if iteration >= self._max_iterations:
                running = False
                halt_message = ('Halting: Maximum number of iterations ('
                                + str(iteration) + ') reached.')
            elif accepted_count >= self._nr_samples:
                running = False
                halt_message = ('Halting: target number of samples ('
                                + str(accepted_count) + ') reached.')

        # Log final state and show halt message
        if logging:
            logger.log(iteration, evaluations)
            self._sampler._log_write(logger)
            logger.log(timer.time())
            if self._log_to_screen:
                print(halt_message)
        samples = np.array(samples)
        return samples

    def log_filename(self):
        """
        Returns log filename.
        """
        return self._log_filename

    def sampler(self):
        """
        Returns the underlying sampler object.
        """
        return self._sampler

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
                raise ValueError('Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

    def set_nr_samples(self, nr_samples=500):
        """
        Sets a target number of samples
        """
        self._nr_samples = nr_samples

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
            self._n_workers = pints.ParallelEvaluator.cpu_count()
            self._parallel = True

        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1
