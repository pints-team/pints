#
# Sequential Monte Carlo
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


class SMC(pints.SMCSampler):
    """
    Samples from a density using sequential Monte Carlo sampling [1], although
    allows multiple MCMC steps per temperature, if desired.

    Algorithm 3.1.1 using equation (31) for ``w_tilde``.

    [1] "Sequential Monte Carlo Samplers", Del Moral et al. 2006,
    Journal of the Royal Statistical Society. Series B.
    """
    def __init__(self, log_posterior, x0, sigma0=None):
        super(SMC, self).__init__(log_posterior, x0, sigma0)

    def run(self):
        """ See :meth:`SMCSampler`. """

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
            samples_new, weights_new = self._steps_2_and_3(
                m_samples[:, :, i], weights_old, self._schedule[i],
                self._schedule[i + 1])
            weights_old = weights_new
            m_samples[:, :, i + 1] = samples_new
        self._weights = weights_new
        return m_samples[:, :, -1]

    def _kernel_sample(self, samples, beta):
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
            r = (self._tempered_distribution(proposed[i], beta)
                 - self._tempered_distribution(samples[i], beta))
            if r <= np.log(np.random.uniform(0, 1)):
                samples_new[i] = samples[i]
            else:
                samples_new[i] = proposed[i]

        assert \
            np.count_nonzero(samples_new == 0) == 0, \
            "Zero elements appearing in samples matrix."

        return samples_new

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


    def _steps_2_and_3(self, samples_old, weights_old, beta_old, beta_new):
        """
        Undertakes steps 2 and 3 from algorithm 3.1.1. in Del Moral et al.
        (2006) except allow multiple MCMC steps per temperature if desired.
        """
        self._weights = weights_old
        if self.ess() < self._ess_threshold:
            resamples, weights_new = self._resample(weights_old, samples_old)
        else:
            resamples, weights_new = samples_old, weights_old

        samples_new = self._kernel_sample(resamples, beta_new)
        # Perform multiple MCMC steps per temperature
        if self._kernel_samples > 1:
            for i in range(0, self._kernel_samples - 1):
                samples_new = self._kernel_sample(samples_new, beta_new)
        weights_new = self._new_weights(
            weights_old, samples_old, samples_new, beta_old, beta_new)

        # Either resample again or don't: algorithm 3.1.1. due to the form of
        # L used (eqn. 30 and 31) resample again
        if self._resample_end_2_3:
            samples_new, weights_discard = self._resample(
                weights_new, samples_new)
        return samples_new, weights_new
