#
# emcee hammer
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
import emcee


class EmceeHammerMCMC(pints.MCMC):
    """
    Creates chains of samples from a target distribution, 
    using the routine described in [1].

    [1] "emcee: The MCMC Hammer", Daniel Foreman-Mackey, David W. Hogg, 
    Dustin Lang, Jonathan Goodman, https://arxiv.org/abs/1202.3665.
    """
    def __init__(self, log_likelihood, x0, sigma0=None):
        super(EmceeHammerMCMC, self).__init__(
            log_likelihood, x0, sigma0)

        # Total number of iterations
        self._iterations = self._dimension * 2000

        # Number of iterations to discard as burn-in
        self._burn_in = int(0.5 * self._iterations)

        # Thinning: Store only one sample per X
        self._thinning_rate = 1

        # Number of walkers to evolve
        self._walkers = 100

    def burn_in(self):
        """
        Returns the number of iterations that will be discarded as burn-in in
        the next run.
        """
        return self._burn_in

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run, including the non-adaptive and burn-in iterations.
        """
        return self._iterations

    def run(self):
        """See: :meth:`pints.MCMC.run()`."""

        # Report the current settings
        if self._verbose:
            print('Running emcee hammer MCMC')
            print('Total number of iterations per walker: ' + str(self._iterations))
            print('Number of walkers: ' + str(self._walkers))
            print(
                'Number of iterations to discard as burn-in: '
                + str(self._burn_in))
            print('Storing one sample per ' + str(self._thinning_rate))

        # Problem dimension
        d = self._dimension

        # Initial starting parameters
        mu = self._x0
        sigma = self._sigma0
        current = self._x0
        current_log_likelihood = self._log_likelihood(current)
        if not np.isfinite(current_log_likelihood):
            raise ValueError(
                'Suggested starting position has a non-finite log-likelihood.')

        # Chain of stored samples
        stored = int((self._iterations - self._burn_in) / self._thinning_rate)
        chain = np.zeros((stored, d))

        # Set initial values
        p0 = [np.random.normal(loc = mu,scale = mu / 100.0,size = len(mu))
              for i in range(self._walkers)]

        # Run
        sampler = emcee.EnsembleSampler(self._walkers, self._dimension, self._log_likelihood)
        pos, prob, state = sampler.run_mcmc(p0, self._iterations)

        # Remove burn-in
        samples = sampler.chain[:, self._burn_in:, :]

        # Thin samples
        samples = samples[:,::self._thinning_rate,:]

        # Return generated chain
        return samples

    def set_burn_in(self, burn_in):
        """
        Sets the number of iterations to discard as burn-in in the next run.
        """
        burn_in = int(burn_in)
        if burn_in < 0:
            raise ValueError('Burn-in rate cannot be negative.')
        self._burn_in = burn_in

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

    def set_walkers(self, walkers):
        """
        Sets the number of walkers to evolve in emcee algorithm.
        """
        self._walkers = walkers
    

    def thinning_rate(self):
        """
        Returns the thinning rate that will be used in the next run. A thinning
        rate of *n* indicates that only every *n-th* sample will be stored.
        """
        return self._thinning_rate


def emcee_hammer_MCMC(log_likelihood, x0, sigma0=None):
    """
    Runs an adaptive covariance MCMC routine with the default parameters.
    """
    return emcee_hammer_MCMC(log_likelihood, x0, sigma0).run()

