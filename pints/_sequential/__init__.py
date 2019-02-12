#
# Sub-module containing sequential MC inference routines
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pints
import numpy as np


class SMCSampler(object):
    """
    Abstract base class for Sequential Monte Carlo (SMC) samplers.

    This class provides fine-grained control over smc sampling. Users who don't
    require this may prefer to use the :class:`SMCController` class instead.

    Arguments:

    ``log_prior``
        A :class:`LogPrior` on the same parameter space, used to draw proposals
        from.
    ``sigma0``
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.

    """
    def __init__(self, log_prior, sigma0=None):

        # Store log prior
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError(
                'Given log_prior function must extend pints.LogPrior')
        self._log_prior = log_prior

        # Set number of parameters
        self._n_parameters = log_prior.n_parameters()

        # Don't check sigma0, will be handled by internal MCMC objects!

        # Internal state
        self._running = False

        # Total number of particles
        self._n_particles = 1000

        # Temperature schedule
        self._schedule = None
        self.set_schedule()

        # Number of MCMC steps per temperature

    def ask(self):
        """
        Returns an array of samples to calculate log pdf values for.
        """
        raise NotImplementedError

    def name(self):
        """
        Returns this method's full name.
        """
        raise NotImplementedError

    def n_temperatures(self):
        """
        Returns the number of temperatures in this sampler's schedule.
        """
        return len(self._schedule)

    def n_kernel_samples(self):
        """
        Returns the number of kernel samples to perform per temperature.
        """
        return self._n_kernel_samples

    def n_particles(self):
        """
        Returns the number of particles used in this method.
        """
        return self._n_particles

    def set_n_kernel_samples(self, n):  # TODO CHANGE NAME?
        """
        Sets the number of MCMC samples to run for each temperature.
        """
        if self._running:
            raise RuntimeError(
                'Number of kernel samples cannot be changed during run.')

        n = int(n)
        if n < 1:
            raise ValueError('Number of samples per temperature must be >= 1.')
        self._n_mcmc_samples = n

    def set_n_particles(self, n):
        """
        Sets the number of particles, must be 10 or greater.
        """
        if self._running:
            raise RuntimeError(
                'Number of particles cannot be changed during run.')

        n = int(n)
        if n < 10:
            raise ValueError('Number of particles must be at least 10.')
        self._n_particles = n

    def set_temperature_schedule(self, schedule=10):
        """
        Sets a temperature schedule.

        If ``schedule`` is an ``int`` it is interpreted as the number of
        temperatures and a schedule is generated that is uniformly spaced on
        the log scale.

        If ``schedule`` is a list (or array) it is interpreted as a custom
        temperature schedule.
        """
        if self._running:
            raise RuntimeError(
                'The temperature schedule cannot be changed during run.')

        # Check type of schedule argument
        if np.isscalar(schedule):

            # Set using int
            schedule = int(schedule)
            if schedule < 2:
                raise ValueError(
                    'A schedule must contain at least two temperatures.')

            # Set a temperature schedule that is uniform on log(T)
            a_max = np.log(1)
            a_min = np.log(0.0001)
            #diff = (a_max - a_min) / schedule
            log_schedule = np.linspace(a_min, a_max, schedule)
            self._schedule = np.exp(log_schedule)

        else:

            # Set to custom schedule
            schedule = pints.vector(schedule)
            if len(schedule) < 2:
                raise ValueError(
                    'A schedule must contain at least two temperatures.')
            if schedule[0] != 0:
                raise ValueError(
                    'First element of temperature schedule must be 0.')

            # Check vector elements all between 0 and 1 (inclusive)
            if np.any(schedule < 0):
                raise ValueError('Temperatures must be non-negative.')
            if np.any(schedule > 1):
                raise ValueError('Temperatures cannot exceed 1.')

            # Store
            self._schedule = schedule

    def tell(self, log_pdfs):
        """
        Performs an iteration of the sampler's algorithm, using the given log
        pdf values.
        """
        raise NotImplementedError

    def temperature(self):
        """
        Returns the current sampler temperature.
        """
        raise NotImplementedError


class SMCController(object):
    """
    Samples from a :class:`pints.LogPDF` using a :class:`Sequential Markov
    Chain Monte Carlo (SMC)<pints.SMCSampler>` method.

    The method to use is specified at runtime. For example::

        mcmc = pints.SMCSampling(
            log_pdf, log_prior, x0, method=pints.SMC)

    Arguments:

    ``log_pdf``
        A :class:`pints.LogPDF` to sample.
    ``log_prior``
        A :class:`pints.LogPrior to draw proposal samples from. The
        implementation assumes that the ``log_pdf`` is expensive to evaluate,
        while evaluating the ``log_prior`` is very cheap.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.
    ``method``
        The method to use, must be a subclass of :class:`pints.SMCSampler`.

    """
    def __init__(self, log_pdf, log_prior, sigma0=None, method=None):

        # Store functions
        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError('Given log_pdf must extend pints.LogPDF.')
        self._log_pdf = log_pdf

        # Get number of parameters
        self._n_parameters = self._log_pdf.n_parameters()

        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given log_prior must extend pints.LogPrior.')
        if log_prior.n_parameters() != self._n_parameters:
            raise ValueError(
                'Given log_pdf and log_prior must have same number of'
                ' parameters.')

        # Check method
        if method is None:
            method = pints.SMC
        else:
            try:
                ok = issubclass(method, pints.SMCSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError('Given method must extend pints.SMCSampler.')

        # Create sampler
        self._sampler = method(log_prior, sigma0)

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()

        # Writing to disk
        #TODO?

        # Parallelisation
        self._parallel = False
        self._n_workers = 1
        #self.set_parallel()    #TODO

        # Controller state
        self._has_run = False

    '''
    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False
    '''

    def run(self):
        """
        Runs the SMC sampler(s) and returns a list of samples from the target
        distribution.
        """
        if self._has_run:
            raise RuntimeError('A controller can only be run once.')

        # Create evaluator object
        #TODO Parallelisation
        evaluator = pints.SequentialEvaluator(self._log_pdf)

        # Count evaluations
        evaluations = 0

        # Number of samples, number of iterations
        n_particles = self._sampler.n_particles()
        n_mcmc_steps = self._sampler.n_mcmc_steps()
        n_temperatures = self._sampler.n_temperatures()
        n_iter = (1 + (n_temperatures - 1) * n_mcmc_steps)

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                print('Using ' + self._sampler.name())
                print('Total number of particles: ' + str(n_particles))
                print('Number of temperatures: ' + str(n_temperatures))
                print('Number of MCMC steps at each temperature: '
                      + str(self.n_mcmc_steps))
                #if self._resample_end_2_3:
                #    print('Resampling at end of each iteration')
                #else:
                #    print('Not resampling at end of each iteration')

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            logger.add_counter('i', max_value=n_temperatures)
            logger.add_float('Temp.')
            logger.add_counter('Eval.', max_value=n_particles * n_iter)
            #TODO: LET SAMPLER LOG ITS OWN STUFF
            logger.add_time('Time m:s')

        # Start sampling
        timer = pints.Timer()
        for iteration in range(n_iter):

            # Get points
            xs = self._sampler.ask()

            # Calculate log pdfs
            fxs = evaluator.evaluate(xs)

            # Update evaluation count
            evaluations += n_particles

            # Update sampler
            samples = self._sampler.tell(fxs)

            # Show progress
            if logging and iteration >= next_message:

                # Log state
                logger.log(iteration, self._sampler.temperature(), evaluations)
                #TODO: LET SAMPLER LOG ITS OWN STUFF
                #self._sampler._log_write(logger)
                logger.log(timer.time())

                # Choose next logging point
                if iteration < self._message_warm_up:
                    next_message = iteration + 1
                else:
                    next_message = self._message_interval * (
                        1 + iteration // self._message_interval)
        # Log final state and show halt message
        if logging:
            logger.log(iteration, self._sampler.temperature(), evaluations)
            #TODO: LET SAMPLER LOG ITS OWN STUFF
            #self._sampler._log_write(logger)
            logger.log(timer.time())

        # Return generated samples
        return samples

    '''
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
        if self._has_run:
            raise RuntimeError('Log interval cannot be changed post-run.')

        iters = int(iters)
        if iters < 1:
            raise ValueError('Interval must be greater than zero.')
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
        if self._has_run:
            raise RuntimeError('Logging cannot be configured post-run.')

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
        if self._has_run:
            raise RuntimeError('Logging cannot be configured post-run.')

        self._log_to_screen = True if enabled else False

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
        if self._has_run:
            raise RuntimeError(
                'Parallelisation cannot be configured post-run.')

        if parallel is True:
            self._parallel = True
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1
    '''
