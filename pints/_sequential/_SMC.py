#
# Sequential Monte Carlo following Del Moral et al. 2006
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
from scipy.special import logsumexp


class SMC(pints.SMCSampler):
    """
    Samples from a density using sequential Monte Carlo sampling [1], although
    allows multiple MCMC steps per temperature, if desired.

    Algorithm 3.1.1 using equation (31) for ``w_tilde``.

    [1] "Sequential Monte Carlo Samplers", Del Moral et al. 2006,
    Journal of the Royal Statistical Society. Series B.
    """
    def __init__(self, log_pdf, log_prior, x0, sigma0=None):
        super(SMC, self).__init__(log_pdf, log_prior, x0, sigma0)


        # MCMC Method used for kernel steps
        self._method = pints.AdaptiveCovarianceMCMC

        # Temperature schedule
        self._schedule = None
        self._n_iterations = None
        self.set_temperature_schedule()

        # Set run params
        self._n_particles = 1000

        # ESS threshold (default from Del Moral et al.)
        self._ess_threshold = self._n_particles / 2

        # Determines whether to resample particles at end of
        # steps 2 and 3 from Del Moral et al. (2006)
        self._resample_end_2_3 = True

        # Set number of MCMC steps to do for each distribution
        self._kernel_samples = 1

        #TODO
        self._running = False

        # vvvvvvvvvvvvvvvvvvvvv MOVE TO SMCSampling vvvvvvvvvvvvvvvvvvvvv

        self._n_evaluations = 0

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()

        # Writing to disk
        #TODO?

        # Stopping criteria
        #TODO?

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def name(self):
        """ See :meth:`SMCSampler.name()`. """
        return 'Sequential Monte Carlo'

    def set_n_particles(self, n):
        """
        Sets the number of particles.
        """
        #TODO Check if not running
        n = int(n)
        if n < 10:
            raise ValueError('Must have more than 10 particles in SMC.')
        self._n_particles = n

    def set_resample_end_2_3(self, resample_end_2_3):
        """
        Determines whether a resampling step is performed at end of steps 2 and
        3 in Del Moral et al. Algorithm 3.1.1.
        """
        #TODO Can this be changed while running?
        if not isinstance(resample_end_2_3, bool):
            raise ValueError('Resample_end_2_3 should be boolean.')
        self._resample_end_2_3 = resample_end_2_3

    def set_ess_threshold(self, ess_threshold):
        """
        Sets the threshold effective sample size (ESS).
        """
        #TODO Can this be changed while running?
        #TODO This check should maybe go in initialise, since it involves 2
        #     parameters?
        if ess_threshold > self._n_particles:
            raise ValueError(
                'ESS threshold must be lower than or equal to number of'
                ' particles.')
        if ess_threshold < 0:
            raise ValueError('ESS must be greater than zero.')
        self._ess_threshold = ess_threshold

    def set_n_kernel_samples(self, kernel_samples):
        """
        Sets number of MCMC samples to do for each temperature.
        """
        #TODO Can this be changed while running?
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
        #TODO Can this be changed while running?

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

        #TODO: This bit is important
        self._n_iterations = len(self._schedule)


    def ask(self):
        pass

    def tell(self):
        pass


    def run(self):
        """
        Runs the SMC sampling routine.
        """
        #TODO: Change to ask-tell
        if self._running:
            raise Exception('Single use')
        self._running = True

        # Create evaluator object
        evaluator = pints.SequentialEvaluator(self._log_pdf)

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
                print('Total number of particles: ' + str(self._n_particles))
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
                               self._n_particles)
            #TODO: Add other informative fields ?
            logger.add_time('Time m:s')

            i_message = 1








        # Initialise

        # Sample from the prior
        self._samples = self._log_prior.sample(self._n_particles)

        # Create chains
        self._chains = [self._method(p, self._sigma0) for p in self._samples]
        if self._chains[0].needs_initial_phase():
            for chain in self._chains:
                chain.set_initial_phase(False)


        # Get LogPDF of initial samples
        # TODO: Change to ask-and-tell
        self._samples_log_pdf = evaluator.evaluate(self._samples)
        for i, f in enumerate(self._samples_log_pdf):
            self._chains[i].ask()
            self._chains[i].tell(f)
        self._n_evaluations += self._n_particles



        # Set weights based on next temperature
        beta = self._schedule[1]
        self._weights = np.zeros(self._n_particles)
        for i, sample in enumerate(self._samples):
            prior = self._log_prior(sample)
            self._weights[i] = beta * (self._samples_log_pdf[i] - prior)
        self._weights = np.exp(self._weights - logsumexp(self._weights))

        # Run!
        for i in range(0, self._n_iterations - 1):

            # Set temperature
            beta = self._schedule[i + 1]

            # If ESS < threshold then resample to avoid degeneracies
            if 1 / np.sum(self._weights**2) < self._ess_threshold:
                self._resample()

            # Update chains with log pdfs tempered with current beta
            for j, chain in enumerate(self._chains):
                current = self._samples[j]
                current_prior = self._log_prior(current)
                current_log_pdf = self._samples_log_pdf[j]
                current_tempered = self._temper(
                    current_log_pdf, current_prior, beta)
                chain.replace(current, current_tempered)

            # Perform MCMC step(s)
            #TODO: Move to ask and tell
            for k in range(self._kernel_samples):

                # Ask                
                proposals = [chain.ask() for chain in self._chains]
                
                # Evaluate
                proposed_log_pdfs = evaluator.evaluate(proposals)
                self._n_evaluations += self._n_particles
          
                # Tell
                for j, proposed in enumerate(proposals):
                    proposed_prior = self._log_prior(proposed)
                    proposed_log_pdf = proposed_log_pdfs[j]
                    proposed_tempered = self._temper(
                        proposed_log_pdf, proposed_prior, beta)
                    
                    updated = self._chains[j].tell(proposed_tempered)
                    if np.all(updated == proposed):  # TODO: use accepted()
                        self._samples[j] = proposed
                        self._samples_log_pdf[j] = proposed_log_pdf

            # Update weights
            # TODO: Write as matrix expression?
            for j, w in enumerate(self._weights):
                self._weights[j] = np.log(w) + self._w_tilde(
                    self._samples_log_pdf[j],
                    self._log_prior(self._samples[j]),
                    self._schedule[i],
                    self._schedule[i + 1])
            self._weights = np.exp(self._weights - logsumexp(self._weights))

            # Conditional resampling step
            if self._resample_end_2_3:
                update_weights = (i != (self._n_iterations - 2))
                self._resample(update_weights)




            # Show progress
            if logging:
                i_message += 1
                if i_message >= next_message:
                    # Log state
                    logger.log(1 - beta, self._n_evaluations, timer.time())

                    # Choose next logging point
                    if i_message > message_warm_up:
                        next_message = message_interval * (
                            1 + i_message // message_interval)

        return self._samples

    def _resample(self, update_weights=True):
        """
        Resamples (and updates the weights and log_pdfs) according to the
        weights vector from the multinomial distribution.
        """
        selected = np.random.multinomial(self._n_particles, self._weights)
        new_samples = np.zeros((self._n_particles, self._n_parameters))
        new_log_prob = np.zeros(self._n_particles)
        a_start = 0
        a_end = 0
        for i in range(0, self._n_particles):
            a_end = a_end + selected[i]
            new_samples[a_start:a_end, :] = self._samples[i]
            new_log_prob[a_start:a_end] = self._samples_log_pdf[i]
            a_start = a_start + selected[i]

        if np.count_nonzero(new_samples == 0) > 0:
            raise RuntimeError('Zero elements appearing in samples matrix.')

        self._samples = new_samples
        self._samples_log_pdf = new_log_prob
        if update_weights:
            self._weights = np.repeat(1 / self._n_particles, self._n_particles)

    def _temper(self, fx, f_prior, beta):
        """
        Returns beta * fx + (1-beta) * f_prior
        """
        return beta * fx + (1 - beta) * f_prior

    def _w_tilde(self, fx_old, f_prior_old, beta_old, beta_new):
        """
        Calculates the log unnormalised incremental weight as per eq. (31) in
        Del Moral.
        """
        return (
            self._temper(fx_old, f_prior_old, beta_new)
            - self._temper(fx_old, f_prior_old, beta_old)
        )

    def weights(self):
        """
        Returns weights from last run of SMC.
        """
        #TODO: Ensure these are set in __init__
        return self._weights

    def ess(self):
        """
        Returns ess from last run of SMC.
        """
        return 1.0 / np.sum(self._weights**2)


    #
    # vvvvvvvvvvvvv ALL THIS SHOULD BE IN SMCSampling vvvvvvvvvvvvvvv
    #

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

