#
# Sequential Monte Carlo
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from scipy import stats
from scipy.special import logsumexp


class SMC(pints.SMCSampler):
    """
    Samples from a density using sequential Monte Carlo sampling [1].

    Algorithm 3.1.1 using equation (31) for ``w_tilde``.

    [1] "Sequential Monte Carlo Samplers", Del Moral et al. 2006,
    Journal of the Royal Statistical Society. Series B.
    """
    def __init__(self, log_posterior, x0, sigma0=None, log_prior=None):
        super(SMC, self).__init__(log_posterior, x0, sigma0, log_prior)

        # Number of particles
        self._particles = 1000

        # Thinning: Store only one sample per X
        self._thinning_rate = 1

        # Temperature schedule
        self._schedule = None
        self.set_temperature_schedule()

        # ESS threshold (default from Del Moral et al.)
        self._ess_threshold = self._particles / 2
        
        # Determines whether to resample particles at end of
        # steps 2 and 3 from Del Moral et al. (2006)
        self._resample_end_2_3 = True

    def set_particles(self, particles):
        """
        Sets the number of particles
        """
        if particles < 10:
            raise ValueError('Must have more than 10 particles in SMC.')
        self._particles = particles
        
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

    def set_temperature_schedule(self, schedule=10):
        """
        Sets a temperature schedule.

        If ``schedule`` is an ``int`` it is interpreted as the number of
        temperatures and a schedule is generated that is uniformly spaced
        on the log scale.

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
            a_max = 0
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

    def run(self):
        """
        Runs the SMC sampler.
        """
        # Report the current settings
        if self._verbose:
            print('Running sequential Monte Carlo')
            print('Total number of particles: ' + str(self._particles))
            print('Number of temperatures: ' + str(len(self._schedule)))
            if self._resample_end_2_3:
              print('Resampling at end of each iteration')
            else:
              print('Not resampling at end of each iteration')
            print('Storing 1 sample per ' + str(self._thinning_rate)
                  + ' particle')

        # Initial starting parameters
        mu = self._x0
        sigma = self._sigma0

        # Starting parameters
        samples = np.random.multivariate_normal(mean=mu, cov=sigma,
                                                size=self._particles)

        # Starting weights
        weights = np.zeros(self._particles)
        for i in range(0, self._particles):
            weights[i] = (
                self.tempered_distribution(samples[i], self._schedule[1])
                - self.tempered_distribution(samples[i], 0.0)
            )
        weights = np.exp(weights - logsumexp(weights))

        # Iterate steps 2 and 3 in Del Moral 3.1.1.
        num_iterates = len(self._schedule)
        m_samples = np.zeros((self._particles, self._dimension, num_iterates))
        m_samples[:, :, 0] = samples
        weights_old = weights
        for i in range(0, num_iterates - 1):
            if self._verbose:
                print(
                    'Sampling from distribution of temperature: '
                    + str(self._schedule[i + 1]))
            samples_new, weights_new = self.steps_2_and_3(
                m_samples[:, :, i], weights_old, self._schedule[i],
                self._schedule[i + 1])
            weights_old = weights_new
            m_samples[:, :, i + 1] = samples_new

        return m_samples[:, :, -1]

    def set_thinning_rate(self, thinning):
        """
        Sets the thinning rate. With a thinning rate of *n*, only every *n-th*
        sample will be stored.
        """
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError('Thinning rate must be greater than zero.')
        self._thinning_rate = thinning

    def thinning_rate(self):
        """
        Returns the thinning rate that will be used in the next run. A thinning
        rate of *n* indicates that only every *n-th* sample will be stored.
        """
        return self._thinning_rate

    def tempered_distribution(self, x, beta):
        """
        Returns the tempered log-pdf:
        beta * log pi(x) + (1 - beta) * log prior(x)
        If not explicitly given prior is assumed to be
        multivariate normal
        """
        return beta * self._log_posterior(x) + (1 - beta) * self._log_prior(x)

    def w_tilde(self, x_old, x_new, beta_old, beta_new):
        """
        Calculates the log unnormalised incremental weight as per eq. (31) in
        Del Moral.
        """
        return (
            self.tempered_distribution(x_old, beta_new)
            - self.tempered_distribution(x_old, beta_old)
        )

    def new_weight(self, w_old, x_old, x_new, beta_old, beta_new):
        """
        Calculates the log new weights as per algorithm 3.1.1.
        in Del Moral et al. (2006).
        """
        w_tilde_value = self.w_tilde(x_old, x_new, beta_old, beta_new)
        return w_old + w_tilde_value

    def new_weights(self, w_old, samples_old, samples_new, beta_old, beta_new):
        """
        Calculates the new weights as per algorithm 3.1.1.
        in Del Moral et al. (2006).
        """
        w_new = np.zeros(self._particles)
        for i in range(0, self._particles):
            w_new[i] = self.new_weight(
                w_old[i], samples_old[i], samples_new[i], beta_old, beta_new)
        return np.exp(w_new - logsumexp(w_new))

    def kernel_sample(self, samples, beta):
        """
        Generates a new sample by using a Metropolis kernel for a distribution
        with log pdf::

            beta * log pi(x) + (1 - beta) * log N(0, sigma).

        """
        proposed = np.zeros((self._particles, self._dimension))
        for i in range(0, self._particles):
            proposed[i] = np.random.multivariate_normal(
                mean=samples[i], cov=self._sigma0, size=1)[0]
        assert \
            np.count_nonzero(proposed == 0) == 0, \
            "Zero elements appearing in proposals matrix."
        samples_new = np.zeros((self._particles, self._dimension))
        for i in range(0, self._particles):
            r = np.exp(
                self.tempered_distribution(proposed[i], beta)
                - self.tempered_distribution(samples[i], beta))
            if r <= np.random.uniform(size=1):
                samples_new[i] = samples[i]
            else:
                samples_new[i] = proposed[i]

        assert \
            np.count_nonzero(samples_new == 0) == 0, \
            "Zero elements appearing in samples matrix."

        return samples_new

    def ess(self, weights):
        """
        Calculates the effective sample size.
        """
        return 1.0 / np.sum(weights**2)

    def resample(self, weights, samples):
        """
        Returns samples according to the weights vector
        from the multinomial distribution.
        """
        selected = np.random.multinomial(self._particles, weights)
        new_samples = np.zeros((self._particles, self._dimension))
        a_start = 0
        a_end = 0
        for i in range(0, self._particles):
            a_end = a_end + selected[i]
            new_samples[a_start:a_end, :] = samples[i]
            a_start = a_start + selected[i]

        assert \
            np.count_nonzero(new_samples == 0) == 0, \
            "Zero elements appearing in samples matrix."

        return new_samples, np.repeat(1.0 / self._particles, self._particles)

    def steps_2_and_3(self, samples_old, weights_old, beta_old, beta_new):
        """
        Undertakes steps 2 and 3 from algorithm 3.1.1. in
        Del Moral et al. (2006).
        """
        if self.ess(weights_old) < self._ess_threshold:
            resamples, weights_new = self.resample(weights_old, samples_old)
        else:
            resamples, weights_new = samples_old, weights_old
        samples_new = self.kernel_sample(resamples, beta_new)
        weights_new = self.new_weights(
            weights_old, samples_old, samples_new, beta_old, beta_new)

        # Either resample again or don't: algorithm 3.1.1. due to the form of L used
        # (eqn. 30 and 31) resample again
        if self._resample_end_2_3:
          samples_new, weights_discard = self.resample(weights_new, samples_new)
        return samples_new, weights_new

