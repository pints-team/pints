#
# Annealed importance sampling
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


class AnnealedImportanceSampler(pints.MarginalLikelihoodSampler):
    """
    *Extends:* :class:`MarginalLikelihoodSampler`
    
    Uses annealed importance sampling [1] to estimate the
    marginal likelihood. Here we use Metropolis updates to
    move around parameter space.
    
    [1] "Annealed Importance Sampling", Radford M. Neal, 1998, Technical Report
    No. 9805.
    """
    def __init__(self, log_likelihood, log_prior):
        super(MarginalLikelihoodSampler, self).__init__(
            log_likelihood, log_prior)

        # Total number of iterations
        self._iterations = self._dimension * 2000
        
        # Number of beta divisions to consider 0 = beta_n <
        # beta_n-1 < ... < beta_0 = 1
        self._num_beta = 20

        # Thinning: Store only one sample per X
        self._thinning_rate = 1

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run.
        """
        return self._iterations
    
    def set_num_beta(self, num_beta):
        """
        Sets the number of beta point values to consider.
        """
        self._num_beta = num_beta

    def run(self):

        # Report the current settings
        if self._verbose:
            print('Running annealed importance sampling')
            print('Total number of iterations: ' + str(self._iterations))
            print('Number of beta values to consider: ' + str(self._num_beta))
            print('Storing one sample per ' + str(self._thinning_rate))

        # Initial starting parameters
        mu = self._x0
        sigma = self._sigma0

        # Beta values
        beta = np.linspace(0, 1, self._num_beta)
        
        # log_w values
        log_w = np.zeros(self._iterations)
        
        # Go!
        for i in range(self._iterations):
            x_vec = np.zeros(self._num_beta)
            f_vec = np.zeros(self._num_beta)
            f_lagged_vec = np.zeros(self._num_beta)
            current = self._x0
            if not np.isfinite(current_f):
                raise ValueError(
                                 'Suggested starting position has a non-finite log-likelihood.')
            for j in range(self._num_beta + 1):
              
                proposed = np.random.multivariate_normal(
                    current, sigma)
                    
                if j == 0:
                    current = self._log_prior.randomSample()
                    current_f = self._log_prior(current)
                    f_lagged_vec[j] = current_f
                # Use Metropolis to step
                elif j < self._num_beta:
                    # Check if the point can be accepted
                    current_f = self._log_prior(current) + beta[j] * self._log_likelihood(current)
                    proposed_f = self._log_prior(proposed) + beta[j] * self._log_likelihood(proposed)
                    f_vec[j - 1] = current_f
                    if np.isfinite(proposed_f):
                        u = np.log(np.random.rand())
                        if u < proposed_f - current_f:
                            current = proposed
                            current_f = proposed_f
                    f_lagged_vec[j] = current_f
                else:
                    f_vec[j - 1] = self._log_prior(current) + self._log_likelihood(current)
                    
            log_w[i] = np.sum(f_vec) - np.sum(f_lagged_vec)

            # Report
            if self._verbose and i % 50 == 0:
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                print('  In burn-in: ' + str(i < self._burn_in))
                print('  Adapting: ' + str(i >= self._adaptation))
                print('  Acceptance rate: ' + str(acceptance))

        # Return generated chain
        return np.sum(w) / self._iterations

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run
        (including burn-in and non-adaptive iterations).
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

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
        
    def num_beta(self):
        """
        Returns the number of beta points to consider in annealed
        importance sampling.
        """
        return self._num_beta
