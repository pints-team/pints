#
# Nested rejection sampler implementation.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np

try:
    # Import logsumexp from its new location in scipy.special
    from scipy.special import logsumexp
except ImportError:  # pragma: no cover
    from scipy.misc import logsumexp


class NestedRejectionSampler(pints.NestedSampler):
    """
    Creates a nested sampler that estimates the marginal likelihood
    and generates samples from the posterior.

    This is the simplest form of nested sampler and involves using
    rejection sampling from the prior as described in the algorithm on page 839
    in [1] to estimate the marginal likelihood and generate weights,
    preliminary samples (with their respective likelihoods), required to
    generate posterior samples.

    The posterior samples are generated as described in [1] on page 849 by
    randomly sampling the preliminary point, accounting for their weights and
    likelihoods.

    Extends :class:`NestedSampler`.

    [1] "Nested Sampling for General Bayesian Computation", John Skilling,
    Bayesian Analysis 1:4 (2006).
    """

    def __init__(self, log_likelihood, log_prior):
        super(NestedRejectionSampler, self).__init__(log_likelihood, log_prior)

        # Target acceptance rate
        self._active_points = 1000

        # Total number of iterations
        self._iterations = 1000

        # Total number of posterior samples
        self._posterior_samples = 1000

        # Total number of likelihood evaluations made
        self._n_evals = 0

    def active_points_rate(self):
        """
        Returns the number of active points that will be used in next run.
        """
        return self._active_points

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run.
        """
        return self._iterations

    def posterior_samples(self):
        """
        Returns the number of posterior samples that will be returned (see
        :meth:`set_posterior_samples()`).
        """
        return self._posterior_samples

    def run(self):
        """ See :meth:`pints.MCMC.run()`. """

        # Check if settings are sensible
        max_post = 0.25 * (self._iterations + self._active_points)
        if self._posterior_samples > max_post:
            raise ValueError(
                'Number of posterior samples must not exceed 0.25 times (the'
                ' number of iterations + the number of active points).')

        # Set up progress reporting
        next_message = 0
        message_warm_up = 3
        message_interval = 20

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            # Create timer
            timer = pints.Timer()

            if self._log_to_screen:
                # Show current settings
                print('Running nested rejection sampling')
                print('Number of active points: ' + str(self._active_points))
                print('Total number of iterations: ' + str(self._iterations))
                print('Total number of posterior samples: ' + str(
                    self._posterior_samples))

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            logger.add_counter('Iter.', max_value=self._iterations)
            logger.add_counter('Eval.', max_value=self._iterations * 10)
            # TODO: Add other informative fields ?
            logger.add_time('Time m:s')

        # Problem dimension
        d = self._n_parameters

        # Generate initial random points by sampling from the prior
        m_active = np.zeros((self._active_points, d + 1))
        m_initial = self._log_prior.sample(self._active_points)
        for i in range(0, self._active_points):
            # Calculate likelihood
            m_active[i, d] = self._log_likelihood(m_initial[i, :])
            self._n_evals += 1

            # Show progress
            if logging and i >= next_message:
                # Log state
                logger.log(0, self._n_evals, timer.time())

                # Choose next logging point
                if i > message_warm_up:
                    next_message = message_interval * (
                        1 + i // message_interval)

        m_active[:, :-1] = m_initial

        # store all inactive points, along with their respective
        # log-likelihoods (hence, d+1)
        m_inactive = np.zeros((self._iterations, d + 1))

        # store weights
        w = np.zeros(self._active_points + self._iterations)

        # store X values (defined in [1])
        X = np.zeros(self._iterations + 1)
        X[0] = 1

        # log marginal likelihood holder
        v_log_Z = np.zeros(self._iterations + 1)

        # Run
        i_message = self._active_points - 1
        for i in range(0, self._iterations):
            a_running_log_likelihood = np.min(m_active[:, d])
            a_min_index = np.argmin(m_active[:, d])
            X[i + 1] = np.exp(-(i + 1) / self._active_points)
            w[i] = X[i] - X[i + 1]
            v_log_Z[i] = a_running_log_likelihood
            m_inactive[i, :] = m_active[a_min_index, :]

            # Independently samples params from the prior until
            # log_likelihood(params) > threshold.
            # Note a_running_log_likelihood can be -inf, so while is never run
            proposed = self._log_prior.sample()[0]
            log_likelihood = self._log_likelihood(proposed)
            self._n_evals += 1
            while log_likelihood < a_running_log_likelihood:
                proposed = self._log_prior.sample()[0]
                log_likelihood = self._log_likelihood(proposed)
                self._n_evals += 1
            m_active[a_min_index, :] = np.concatenate(
                (proposed, np.array([log_likelihood])))

            # Show progress
            if logging:
                i_message += 1
                if i_message >= next_message:
                    # Log state
                    logger.log(i_message, self._n_evals, timer.time())

                    # Choose next logging point
                    if i_message > message_warm_up:
                        next_message = message_interval * (
                            1 + i_message // message_interval)

        v_log_Z[self._iterations] = logsumexp(m_active[:, d])
        w[self._iterations:] = float(X[self._iterations]) / float(
            self._active_points)
        m_samples_all = np.vstack((m_inactive, m_active))
        log_Z = logsumexp(v_log_Z, b=w[0:(self._iterations + 1)])

        vP = np.exp(m_samples_all[:, d] - log_Z) * w
        m_theta = m_samples_all[:, :-1]
        vIndex = np.random.choice(
            range(0, self._iterations + self._active_points),
            self._posterior_samples, p=vP)
        m_posterior_samples = m_theta[vIndex, :]

        return m_posterior_samples, log_Z

    def set_active_points_rate(self, active_points):
        """
        Sets the number of active points for the next run.
        """
        active_points = int(active_points)
        if active_points <= 5:
            raise ValueError('Number of active points must be greater than 5.')
        self._active_points = active_points

    def n_hyper_parameters(self):
        """
        Returns the number of hyper-parameters for this method (see
        :class:`TunableMethod`).
        """
        return 1

    def set_hyper_parameters(self, x):
        """
        Sets the hyper-parameters for the method with the given vector of
        values (see :class:`TunableMethod`).

        Hyper-parameter vector is: ``[active_points_rate]``

        Arguments:

        ``x`` an array of length ``n_hyper_parameters`` used to set the
              hyper-parameters
        """

        self.set_active_points_rate(x[0])

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run.
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_posterior_samples(self, posterior_samples):
        """
        Sets the number of posterior samples to generate from points proposed
        by the nested sampling algorithm.
        """
        posterior_samples = int(posterior_samples)
        if posterior_samples < 1:
            raise ValueError(
                'Number of posterior samples must be greater than zero.')
        self._posterior_samples = posterior_samples
