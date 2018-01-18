#
# Population MCMC
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


class PopulationMCMC(pints.MCMC):
    """
    *Extends:* :class:`MCMC`

    Creates a chain of samples from a target distribution, using the population
    MCMC routine described in algorithm 1 in [1].
    
    The algorithm goes through the following steps (after initialising
    N chains):
    
    1. Mutation: randomly select chain i and update the chain using
    a Markov kernel that admits pi_i as its invariant distribution.
    
    2. Select another chain j at random from the remaining and
    make a random choice between steps 3 and 4.
    
    3. Exchange: swap the parameter vector of i and j with probability min(1,A),
    
    A = pi_i(x_j) * pi_j(x_i) / (pi_i(x_i) * pi_j(x_j)),
    
    where x_i and x_j are the current values of chains i and j, respectively.    
    
    4. Crossover: select a parameter index at random and swap the current
    values of i and j at that position, with probability min(1,A),
    
    A = pi_i(x_j') * pi_j(x_i') / (pi_i(x_i) * pi_j(x_j)),
    
    where x_i' and x_j' are the parameter vectors of chains i and j, respectively,
    after the proposed swap has taken place. For example, suppose,
    x_i = (a,b,c) and x_j = (d,e,f) before the swap. Suppose we then randomly
    select index 2 this would mean x_i' = (a,e,c) and x_j' = (d,b,f).
    
    Here pi_i = p(theta|data) ^ (1 - T_i), where p(theta|data) is the target
    distribution and T_i is bounded between [0,1] and represents a tempering
    parameter. We use a range of T = (0,delta_T,...,1), where 
    delta_T = 1 / num_temperatures, and the chain with
    T_i = 0 is the one whose target distribution we want to sample.

    [1] "On population-based simulation for static inference", Ajay Jasra,
    David A. Stephens and Christopher C. Holmes, Statistical Computing, 2007.
    """
    def __init__(self, log_likelihood, x0, sigma0=None):
        super(PopulationMCMC, self).__init__(
            log_likelihood, x0, sigma0)

        # Total number of iterations
        self._iterations = self._dimension * 2000

        # Number of iterations before adapation
        self._num_temperatures = 10
        self.set_temperature_schedule()

        # Number of iterations to discard as burn-in
        self._burn_in = int(0.5 * self._iterations)

        # Thinning: Store only one sample per X
        self._thinning_rate = 1
        
        # Probability of choosing an exchange versus
        # crossover step in each iteration
        self._prob_exchange = 0.5

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

    def num_temperatures(self):
        """
        Returns the number of temperatures to use in the tempering,
        which are equally spaced between 0 and 1 unless a custom
        temperature schedule has been specified.
        """
        return self._num_temperatures
    
    def prob_exchange(self):
        """
        Returns the Probability of choosing an exchange versus
        a crossover step in each iteration.
        """
        return self._prob_exchange
    
    def set_prob_exchange(self, prob):
        """
        Sets the probability of choosing exchange an exchange versus
        a crossover step in each iteration.
        """
        if prob < 0:
            raise ValueError('Probability of exchange step must be non-negative.')
        if prob > 1:
            raise ValueError('Probability of exchange step must be less than 1.')
        
        self._prob_exchange = prob
        
    def temperature_schedule(self):
        """
        Returns the temperature schedule used in the tempering
        algorithm. Each temperature T pertains to particular chain
        whose stationary distribution is p(theta|data) ^ (1 - T).
        """
        return self._temperature_schedule
        
    def set_num_temperatures(self, num_temperatures):
        """
        Returns the number of temperatures to use in the tempering,
        which are equally spaced between 0 and 1.
        """
        if num_temperatures < 1:
                raise ValueError('Number of temperatures must be equal to or greater than 1.')
        self._num_temperatures = int(num_temperatures)
        self.set_temperature_schedule()
        
    def set_temperature_schedule(self, schedule=None):
        """
        If, schedule=None, this sets the temperature schedule for
        tempering to be at uniform intervals in [0,1].
        Else, sets the temperature schedule according to the
        vector 'schedule'.
        """
        if schedule is None:
            self._temperature_schedule = np.linspace(0, 0.95, self._num_temperatures)
        else:
            self.set_num_temperatures(len(schedule))
            
            # Check vector elements all between 0 and 1 (inclusive)
            if schedule[0] != 0:
                raise ValueError('First element of temperature schedule must be 0.')
            for i in range(self._num_temperatures):
                if schedule[i] < 0:
                    raise ValueError('Temperature schedule elements must be non-negative')
                if schedule[i] > 0:
                    raise ValueError('Temperature schedule elements must be less than or equal to 1')

            self._temperature_schedule = schedule

    def run(self):
        """See: :meth:`pints.MCMC.run()`."""

        # Report the current settings
        if self._verbose:
            print('Running population MCMC')
            print('Total number of iterations: ' + str(self._iterations))
            print('Number of temperatures: ' + str(self._num_temperatures))
            print(
                'Number of iterations to discard as burn-in: '
                + str(self._burn_in))
            print('Storing 1 sample per ' + str(self._thinning_rate) + ' iteration')

        # Problem dimension
        d = self._dimension
        
        # Number of temperatures
        T = self._num_temperatures
        
        # Temperature schedule
        temperatures = self._temperature_schedule

        # Initial starting parameters
        mu = self._x0
        sigma = self._sigma0
        current = self._x0
        current_log_likelihood = self._log_likelihood(current)
        if not np.isfinite(current_log_likelihood):
            raise ValueError(
                'Suggested starting position has a non-finite log-likelihood.')
                
        # Initialise
        # chains of stored samples
        chains = np.zeros((self._iterations, T, d))
        current = np.zeros((T, d))
        current_log_likelihood = np.zeros(T)

        # Set initial values
        for i in range(T):
            current[i, :] = np.random.normal(loc=mu, scale=mu / 20.0, size=len(mu))
            current_log_likelihood[i] = (1.0 - temperatures[i]) * self._log_likelihood(current[i, :])
            chains[0, i, :] = current[i, :]
            
        # Initial acceptance rate (value doesn't matter)
        loga = np.zeros(T)
        acceptance = np.zeros(T)
        counts = np.zeros(T)
        sigma = []
        # print(self._sigma0)
        for i in range(T):
            sigma.append(self._sigma0)

        # Go!
        for t in range(self._iterations):
          
            # Select random chain i and another chain j
            i, j = np.random.choice(T, 2, replace=False)

            # Update using Markov kernel for chain i
            accepted = 0
            proposed = np.random.multivariate_normal(current[i, :], np.exp(loga[i]) * sigma[i])

             # Check if the point can be accepted
            proposed_log_likelihood = (1.0 - temperatures[i]) * self._log_likelihood(proposed)
            if np.isfinite(proposed_log_likelihood):
                u = np.log(np.random.rand())
                if u < proposed_log_likelihood - current_log_likelihood[i]:
                    current_log_likelihood[i] = proposed_log_likelihood
                    current[i, :] = proposed
                    accepted = 1
            
            counts[i] += 1
            # Adapt covariance matrix
            # if counts[i] >= 50:
            #     gamma = (counts[i] - 50 + 2) ** - 0.6
            #     mu = (1 - gamma) * mu + gamma * current[i, :]
            #     loga[i] += gamma * (accepted - 0.25)
            #     dsigm = np.reshape(current[i, :] - mu, (d, 1))
            #     sigma[i] = (1 - gamma) * sigma[i] + gamma * np.dot(dsigm, dsigm.T)
            #     
            # print(sigma[i])

            # # Determine whether to do an exchange or crossover step
            u1 = np.random.rand()

            # Propose exchange step
            if self._prob_exchange > u1:
                proposed_i = current[j, :]
                proposed_j = current[i, :]

            # Propose crossover step
            else:
                # Select random element of parameter vector to swap
                k = np.random.randint(d)
                proposed_i = current[i, :]
                proposed_i[k] = current[j, k]
                proposed_j = current[j, :]
                proposed_j[k] = current[i, k]

            # Calculate proposed log likelihoods
            proposed_log_likelihood_ij = (1.0 - temperatures[i]) * self._log_likelihood(proposed_i)
            proposed_log_likelihood_ji = (1.0 - temperatures[j]) * self._log_likelihood(proposed_j)

            # Accept/reject chosen step
            log_A = proposed_log_likelihood_ij + proposed_log_likelihood_ji - (current_log_likelihood[i]
                                                                                   + current_log_likelihood[j])
            u2 = np.log(np.random.rand())
            if u2 < log_A:
                current[i, :] = proposed_i
                current[j, :] = proposed_j
                current_log_likelihood[i] = proposed_log_likelihood_ij
                current_log_likelihood[j] = proposed_log_likelihood_ji
            
            # Update all chains
            chains[t, :, :] = current
                
            # Report
            if self._verbose and t % 50 == 0:
                print('Iteration ' + str(t) + ' of ' + str(self._iterations))
                print('  In burn-in: ' + str(t < self._burn_in))
        
        # Keep chains whose target distribution is desired
        keep_index = []
        # for i in range(self._num_temperatures):
        #     if self._temperature_schedule[i] == 0:
        #         keep_index.append(i)
        # chains = chains[:, keep_index, :]
        
        # Thin chains
        non_burn_in = self._iterations - self._burn_in
        chains = chains[non_burn_in:, :, :]
        chains = chains[::self._thinning_rate, :, :]
        
        # Convert 3d array to list of lists
        num_remaining = chains.shape[1]
        samples = [chains[:, i, :] for i in range(0, num_remaining)]

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

