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
    
        # Number of threads to evolve in parallel
        self._threads = 1

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
            print('Storing 1 sample per ' + str(self._thinning_rate) + ' iteration')

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
        if self._threads < 2:
            sampler = emcee.EnsembleSampler(self._walkers, self._dimension, self._log_likelihood)
        else:
            sampler = emcee.EnsembleSampler(self._walkers, self._dimension, self._log_likelihood,
                                            threads = self._threads)
        pos, prob, state = sampler.run_mcmc(p0, self._iterations, thin=self._thinning_rate)

        # Remove burn-in
        samples = sampler.chain[:, self._burn_in:, :]

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


class EmceePTMCMC(pints.MCMC):
    """
    Uses parallel tempering to sample from the posterior with an
    implementation from the makers of the emcee software, described:
    http://dfm.io/emcee/current/user/pt/
    """
    def __init__(self, log_likelihood, x0, sigma0=None):
        super(EmceePTMCMC, self).__init__(
            log_likelihood, x0, sigma0)

        # Total number of iterations
        self._iterations = self._dimension * 2000

        # Number of iterations to discard as burn-in
        self._burn_in = int(0.5 * self._iterations)

        # Thinning: Store only one sample per X
        self._thinning_rate = 1

        # Number of walkers to evolve
        self._walkers = 100
    
        # Number of threads to evolve in parallel
        self._threads = 1
        
        # Number of temperatures to consider
        self._num_temps = 20
        
        # Assume flat prior (likelihood can equal likelihood * prior)
        def log_prior(x):
          return 0.0
        self._log_prior = log_prior

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
            print('Number of temperatures: ' + str(self._num_temps))
            print('Number of walkers: ' + str(self._walkers))
            print(
                'Number of iterations to discard as burn-in: '
                + str(self._burn_in))
            print('Storing 1 sample per ' + str(self._thinning_rate) + ' iteration')

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
        p0 = [[np.random.normal(loc = mu,scale = mu / 100.0,size = len(mu))
              for i in range(self._walkers)] for j in range(self._num_temps)]

        # Run
        if self._threads < 2:
            sampler = emcee.PTSampler(ntemps=self._num_temps, nwalkers=self._walkers, dim=self._dimension, 
                                      logl=self._log_likelihood, logp=self._log_prior)
        else:
            sampler = emcee.PTSampler(ntemps=self._num_temps, nwalkers=self._walkers, dim=self._dimension, 
                                      logl=self._log_likelihood, logp=self._log_prior, threads = self._threads)
        pos, prob, state = sampler.run_mcmc(pos0=p0, N=self._iterations, thin=self._thinning_rate)

        # Remove burn-in
        samples = sampler.chain[:, :, self._burn_in:, :]
        
        # Consider only zero temperature chains
        samples = samples[0, :, :, :]

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
        
    def set_num_temps(self, num_temps):
        """
        Sets the number of different temperature schedules to consider.
        """
        self._num_temps = num_temps
    
    def set_log_prior(self, log_prior):
        """
        Sets the log prior
        """
        self._log_prior = log_prior
