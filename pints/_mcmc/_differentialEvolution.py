#
# Differential evolution MCMC
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


class DifferentialEvolutionMCMC(pints.MCMC):
    """
    *Extends:* :class:`MCMC`

    Uses differential evolution MCMC as described in [1]
    to do posterior sampling from the posterior.
    
    In each step of the algorithm N chains are evolved
    using the evolution equation,
    
    x_proposed = x[i,r] + gamma * (X[i,r1] - x[i,r2]) + e
    
    where r1 and r2 are random chain indices chosen (without
    replacement) from the N available chains, and i indicates
    the current time step. e ~ N(0,b) in d dimensions (where
    d is the dimensionality of the parameter vector).
    
    If x_proposed / x[i,r] > u ~ U(0,1), then 
    x[i+1,r] = x_proposed; otherwise, x[i+1,r] = x[i].
    
    [1] "A Markov Chain Monte Carlo version of the genetic
    algorithm Differential Evolution: easy Bayesian computing
    for real parameter spaces", 2006, Cajo J. F. Ter Braak,
    Statistical Computing.
    """
    def __init__(self, log_likelihood, x0, sigma0=None):
        super(DifferentialEvolutionMCMC, self).__init__(
            log_likelihood, x0, sigma0)

        # Total number of iterations
        self._iterations = self._dimension * 2000

        # Gamma
        self._gamma = 2.38 / np.sqrt(2 * self._dimension)
        
        # Normal proposal std.
        self._b = 0.01
        
        # Number of chains to evolve
        self._num_chains = 100

        # Number of iterations to discard as burn-in
        self._burn_in = int(0.5 * self._iterations)

        # Thinning: Store only one sample per X
        self._thinning_rate = 1

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
            print('Running differential evolution MCMC')
            print('gamma = ' + str(self._gamma))
            print('normal proposal std. = ' + str(self._b))
            print('Total number of iterations: ' + str(self._iterations))
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

        # chains of stored samples
        chains = np.zeros((self._iterations, self._num_chains, self._dimension))
        current_log_likelihood = np.zeros(self._num_chains)

        # Set initial values
        for j in range(self._num_chains):
          chains[0, j, :] = np.random.normal(loc = mu, scale = mu / 100.0, size = len(mu))
          current_log_likelihood[j] = self._log_likelihood(chains[0, j, :])

        # Go!
        for i in range(1, self._iterations):
          r1 = np.random.choice(self._num_chains, self._num_chains, replace = False)
          r2 = np.random.choice(self._num_chains, self._num_chains, replace = False)
          
          for j in range(self._num_chains):
            proposed = chains[i - 1, j, :] + self._gamma * (chains[i - 1, r1[j], :] - chains[i - 1, r2[j], :]) + np.random.normal(loc = 0, scale = self._b * mu, size = len(mu))
            u = np.log(np.random.rand())
            proposed_log_likelihood = self._log_likelihood(proposed)
            
            if u < proposed_log_likelihood - current_log_likelihood[j]:
              chains[i, j, :] = proposed
              current_log_likelihood[j] = proposed_log_likelihood
            else:
              chains[i, j, :] = chains[i - 1, j, :]

          # Report
          if self._verbose and i % 50 == 0:
              print('Iteration ' + str(i) + ' of ' + str(self._iterations))
              print('  In burn-in: ' + str(i < self._burn_in))
        
        non_burn_in = self._iterations - self._burn_in
        chains = chains[non_burn_in:, :, :]
        chains = chains[::self._thinning_rate, :, :]
        
        # Convert 3d array to list of lists
        samples = [chains[:, i, :] for i in range(0,self._num_chains)]
        
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

    def thinning_rate(self):
        """
        Returns the thinning rate that will be used in the next run. A thinning
        rate of *n* indicates that only every *n-th* sample will be stored.
        """
        return self._thinning_rate
        
    def set_gamma(self, gamma):
        """
        Sets the gamma coefficient used in updating the position of each
        chain.
        """
        if gamma < 0:
          raise ValueError('Gamma must be non-negative.')
        self._gamma = gamma
    
    def set_b(self, b):
        """
        Sets the normal scale coefficient used in updating the position of each
        chain.
        """
        if b < 0:
          raise ValueError('normal scale coefficient must be non-negative.')
        self._b = b
    
    def set_num_chains(self, num_chains):
      """
      Sets the number of chains to evolve
      """
      if num_chains < 10:
        raise ValueError('This method works best with many chains (>>10, typically).')
      self._num_chains = num_chains


def differential_evolution_mcmc(log_likelihood, x0, sigma0=None):
    """
    Runs an differential evolution MCMC routine with the default parameters.
    """
    return DifferentialEvolutionMCMC(log_likelihood, x0, sigma0).run()

