#
# Sub-module containing sequential MC inference routines
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
from scipy.special import logsumexp


class SMCSampler(object):
    """
    Abstract base class for sequential Monte Carlo samplers.

    Arguments:

    ``log_posterior``
        A :class:`LogPosterior` function that evaluates points in the parameter
        space.

    """
    def __init__(self, log_posterior, x0, sigma0=None):

        # Store log_likelihood and log_prior
        if not isinstance(log_posterior, pints.LogPDF):
            raise ValueError(
                'Given posterior function must extend pints.LogPDF')
        self._log_posterior = log_posterior

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

        # Get dimension
        self._dimension = self._log_posterior.n_parameters()

        if not isinstance(log_posterior, pints.LogPosterior):
            lower = np.repeat(-100, self._dimension)
            upper = np.repeat(+100, self._dimension)
            self._log_prior = pints.UniformLogPrior(lower, upper)
        else:
            self._log_prior = self._log_posterior._log_prior

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_rate()
        self._evaluations = 0

        # Parallelisation
        self._parallel = False
        self._n_workers = 1
        self.set_parallel()

        # Initial starting parameters
        self._mu = self._x0
        self._sigma = self._sigma0
        self._method = pints.AdaptiveCovarianceMCMC
        self._chain = self._method(self._x0, self._sigma0)

        # Initial phase (needed for e.g. adaptive covariance)
        self._initial_phase_iterations = 0
        # self._needs_initial_phase = self._method.needs_initial_phase()
        # if self._needs_initial_phase:
        #     self.set_initial_phase_iterations()

        # Temperature schedule
        self._schedule = None
        self.set_temperature_schedule()

        # Set run params
        self._particles = 1000
        self._initialise()

        # ESS threshold (default from Del Moral et al.)
        self._ess_threshold = self._particles / 2

        # Determines whether to resample particles at end of
        # steps 2 and 3 from Del Moral et al. (2006)
        self._resample_end_2_3 = True

        # Set number of MCMC steps to do for each distribution
        self._kernel_samples = 1

    def _initialise(self):
        """
        Initialises SMC
        """
        self._samples = np.random.multivariate_normal(
            mean=self._mu, cov=self._sigma, size=self._particles)
        self._samples_log_pdf = np.zeros(self._particles)
        self._weights = np.zeros(self._particles)
        for i in range(self._particles):
            self._samples_log_pdf[i] = self._log_posterior(self._samples[i])
            log_prior_pdf = self._log_prior(self._samples[i])
            self._weights[i] = (
                self._schedule[1] * self._samples_log_pdf[i] +
                (1 - self._schedule[1]) * log_prior_pdf -
                log_prior_pdf
            )
            self._evaluations += 1
        self._weights = np.exp(self._weights - logsumexp(self._weights))

    def set_particles(self, particles):
        """
        Sets the number of particles
        """
        if particles < 10:
            raise ValueError('Must have more than 10 particles in SMC.')
        self._particles = particles
        self._initialise()

    def set_resample_end_2_3(self, resample_end_2_3):
        """
        Determines whether a resampling step is performed at end of
        steps 2 and 3 in Del Moral et al. Algorithm 3.1.1
        """
        if not isinstance(resample_end_2_3, bool):
            raise ValueError('Resample_end_2_3 should be boolean.')
        self._resample_end_2_3 = resample_end_2_3

    def set_ess_threshold(self, ess_threshold):
        """
        Sets the threshold ESS (effective sample size)
        """
        if ess_threshold > self._particles:
            raise ValueError('ESS threshold must be below actual sample size.')
        if ess_threshold < 0:
            raise ValueError('ESS must be greater than zero.')
        self._ess_threshold = ess_threshold

    def set_kernel_samples(self, kernel_samples):
        """
        Sets number of MCMC samples to do for each temperature
        """
        if kernel_samples < 1:
            raise ValueError('Number of samples per temperature must be >= 1.')
        if not isinstance(kernel_samples, int):
            raise ValueError('Number of samples per temperature must be int.')
        self._kernel_samples = kernel_samples

    def set_temperature_schedule(self, schedule=10):
        """
        Sets a temperature schedule.

        If ``schedule`` is an ``int`` it is interpreted as the number of
        temperatures and a schedule is generated that is uniformly spaced on
        the log scale.

        If ``schedule`` is a list (or array) it is interpreted as a custom
        temperature schedule.
        """

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
            self._schedule.setflags(write=False)

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
        self._iterations = len(self._schedule)

    def ask(self):
        """
        Returns a position in the search space to evaluate.
        """
        self._proposed = self._chain.ask()
        return self._proposed

    def tell(self, fx):
        """
        Performs an iteration of the MCMC algorithm, using the evaluation
        ``fx`` of the point previously specified by ``ask``. Returns the next
        sample in the chain.
        """
        return self._chain.tell(fx)

    def run(self):
        """
        Runs the SMC sampling routine.
        """

        # Create evaluator object
        if self._parallel:
            # Use at most n_workers workers
            n_workers = min(self._n_workers, self._chains)
            evaluator = pints.ParallelEvaluator(
                self._log_posterior, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(self._log_posterior)

        # Set up progress reporting
        next_message = 0
        message_warm_up = 0
        message_interval = 1

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            # Create timer
            timer = pints.Timer()

            if self._log_to_screen:
                # Show current settings
                print('Running ' + self.name())
                print('Total number of particles: ' + str(self._particles))
                print('Number of temperatures: ' + str(len(self._schedule)))
                if self._resample_end_2_3:
                    print('Resampling at end of each iteration')
                else:
                    print('Not resampling at end of each iteration')
                print(
                    'Number of MCMC steps at each temperature: '
                    + str(self._kernel_samples))

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            logger.add_float('Temperature')
            logger.add_counter('Eval.', max_value=len(self._schedule) *
                               self._particles)
            #TODO: Add other informative fields ?
            logger.add_time('Time m:s')

            i_message = 1

        # Run!
        for i in range(0, self._iterations - 1):
            # Set current temperature
            self._current_beta = self._schedule[i + 1]

            # If ESS < threshold then resample to avoid degeneracies
            if self.ess() < self._ess_threshold:
                self._samples, self._weights, self._samples_log_pdf = (
                    self._resample()
                )

            for j in range(self._particles):
                for k in range(self._kernel_samples):
                    self._current = np.copy(self._samples[j])
                    self._chain._current = np.copy(self._current)
                    # Use some method to propose new samples
                    self._proposed = self.ask()

                    # Evaluate their fit
                    fx = evaluator.evaluate([self._proposed])[0]

                    # prior evaluation
                    f_prior = self._log_prior(self._proposed)

                    # Use tell from adaptive covariance MCMC
                    f_prior_current = self._log_prior(self._current)
                    self._current_log_pdf = self._tempered_distribution(
                        self._samples_log_pdf[j],
                        f_prior_current,
                        self._current_beta)

                    self._chain._proposed = np.copy(self._proposed)
                    self._chain._current_log_pdf = np.copy(
                        self._current_log_pdf)
                    self._samples[j] = self.tell(
                        self._tempered_distribution(fx,
                                                    f_prior,
                                                    self._current_beta)
                    )
                    # translate _current_log_pdf back into posterior pdf value
                    self._samples_log_pdf[j] = (
                        (1.0 / self._current_beta) *
                        (self._chain._current_log_pdf -
                         (1 - self._current_beta) *
                         self._log_prior(self._samples[j]))
                    )

                    self._evaluations += 1

            # Store old samples
            self._samples_old = np.copy(self._samples)
            self._samples_log_pdf_old = np.copy(self._samples_log_pdf)
            # Update weights
            self._new_weights(self._schedule[i], self._current_beta)

            # # Conditional resampling step
            if self._resample_end_2_3:
                self._samples, weights_discard, self._samples_log_pdf = (
                    self._resample()
                )

            # Show progress
            if logging:
                i_message += 1
                if i_message >= next_message:
                    # Log state
                    logger.log(1 - self._current_beta, self._evaluations,
                               timer.time())

                    # Choose next logging point
                    if i_message > message_warm_up:
                        next_message = message_interval * (
                            1 + i_message // message_interval)

    def _tempered_distribution(self, fx, f_prior, beta):
        """
        Returns beta * fx + (1-beta) * f_prior
        """
        return beta * fx + (1 - beta) * f_prior

    def _resample(self):
        """
        Returns samples according to the weights vector from the multinomial
        distribution.
        """
        selected = np.random.multinomial(self._particles, self._weights)
        new_samples = np.zeros((self._particles, self._dimension))
        new_log_prob = np.zeros(self._particles)
        a_start = 0
        a_end = 0
        for i in range(0, self._particles):
            a_end = a_end + selected[i]
            new_samples[a_start:a_end, :] = self._samples[i]
            new_log_prob[a_start:a_end] = self._samples_log_pdf[i]
            a_start = a_start + selected[i]

        assert \
            np.count_nonzero(new_samples == 0) == 0, \
            "Zero elements appearing in samples matrix."

        return (new_samples, np.repeat(1.0 / self._particles, self._particles),
                new_log_prob)

    def _w_tilde(self, fx_old, f_prior_old, beta_old, beta_new):
        """
        Calculates the log unnormalised incremental weight as per eq. (31) in
        Del Moral.
        """
        return (
            self._tempered_distribution(fx_old, f_prior_old, beta_new)
            - self._tempered_distribution(fx_old, f_prior_old, beta_old)
        )

    def _new_weight(self, w_old, fx_old, f_prior_old, beta_old, beta_new):
        """
        Calculates the log new weights as per algorithm 3.1.1. in Del Moral et
        al. (2006).
        """
        w_tilde_value = self._w_tilde(fx_old, f_prior_old, beta_old, beta_new)
        return np.log(w_old) + w_tilde_value

    def _new_weights(self, beta_old, beta_new):
        """
        Calculates the new weights as per algorithm 3.1.1 in Del Moral et al.
        (2006).
        """
        for i, w in enumerate(self._weights):
            fx_old = self._samples_log_pdf_old[i]
            f_prior_old = self._log_prior(self._samples_old[i])
            self._weights[i] = self._new_weight(w, fx_old,
                                                f_prior_old,
                                                beta_old, beta_new)
        self._weights = np.exp(self._weights - logsumexp(self._weights))

    def ess(self):
        """
        Calculates the effective sample size.
        """
        return 1.0 / np.sum(self._weights**2)

    def weights(self):
        """
        Returns weights from last run of SMC.
        """
        return self._weights

    def set_log_rate(self, rate=20, warm_up=3):
        """
        Changes the frequency with which messages are logged.

        Arguments:

        ``rate``
            A log message will be shown every ``rate`` iterations.
        ``warm_up``
            A log message will be shown every iteration, for the first
            ``warm_up`` iterations.

        """
        rate = int(rate)
        if rate < 1:
            raise ValueError('Rate must be greater than zero.')
        warm_up = max(0, int(warm_up))

        self._message_rate = rate
        self._message_warm_up = warm_up

    def set_log_to_file(self, filename=None, csv=False):
        """
        Enables logging to file when a filename is passed in, disables it if
        ``filename`` is ``False`` or ``None``.

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
        Enables or disables logging to screen.
        """
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
        if parallel is True:
            self._parallel = True
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1
