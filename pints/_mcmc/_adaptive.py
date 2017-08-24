#
# Exponential natural evolution strategy optimizer: xNES
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
import pints
import numpy as np
import scipy
import scipy.linalg
import multiprocessing

class AdaptiveCovarianceMCMC(pints.MCMC):
    """
    Creates a chain of samples from a target distribution, using the adaptive
    covariance machine described in [1, 2].
        
    [1] Glasmachers, Schaul, Schmidhuber et al. (2010) Exponential natural
    evolution strategies.
    Proceedings of the 12th annual conference on Genetic and evolutionary
    computation
    
    [2] PyBrain: The Python machine learning library (http://pybrain.org)
    
    """
    def run(self):

        # Target acceptance rate
        acceptance_target = 0.25

        # Total number of iterations
        #TODO Allow changing before run() with method call
        iterations = 2000 * self._dimension

        # Number of iterations to use adaptation in
        #TODO Allow changing before run() with method call
        adaptation = int(iterations / 2)

        # Number of iterations to discard as burn-in
        #TODO Allow changing before run() with method call
        burn_in = int(iterations / 2)

        # Thinning: Store only one sample per X
        #TODO Allow changing before run() with method call
        thinning = 10

        # Initial starting parameters
        mu = self._x0
        sigma = self._sigma0
        current = self._x0
        current_log_likelihood = self._log_likelihood(current)
        if not np.isfinite(current_log_likelihood):
            raise ValueError('Suggested starting position has a non-finite'
                ' log-likelihood')
        
        # Problem dimension
        d = self._dimension

        # Chain of stored samples
        stored = int((iterations - burn_in) / thinning)
        chain = np.zeros((stored, self._dimension))

        # Initial acceptance rate (value doesn't matter)
        loga = 0
        acceptance = 0

        # Go!
        for i in xrange(iterations):
            # Propose new point
            # Note: Normal distribution is symmetric
            #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
            #  distribution term from the acceptance criterion
            proposed = np.random.multivariate_normal(current,
                np.exp(loga) * sigma)
            
            # Check if the point can be accepted
            accepted = 0
            proposed_log_likelihood = self._log_likelihood(proposed)
            if np.isfinite(proposed_log_likelihood):
                u = np.log(np.random.rand())
                if u < proposed_log_likelihood - current_log_likelihood:
                    accepted = 1
                    current = proposed
                    current_log_likelihood = proposed_log_likelihood
            
            # Adapt covariance matrix
            if i > adaptation:
                gamma = (i - adaptation + 1) ** -0.6
                dsigm = np.reshape(current - mu, (d, 1))
                sigma = (1 - gamma) * sigma + gamma * np.dot(dsigm, dsigm.T)
                mu = (1 - gamma) * mu + gamma * current
                loga += gamma * (accepted - acceptance_target)

            # Update acceptance rate
            acceptance = (i * acceptance + accepted) / (i + 1)
            
            # Add point to chain
            ilog = i - burn_in
            if ilog >= 0 and ilog % thinning == 0:
                chain[ilog // thinning, :] = current

        # Check that chain fully filled
        if ilog < len(chain):
            raise Exception('Unexpected error: Chain not fully generated.')

        # Return generated chain
        return chain

def adaptive_covariance_mcmc(log_likelihood, x0, sigma0=None):
    """
    Runs an adaptive covariance MCMC routine with the default parameters.
    """
    return AdaptiveCovarianceMCMC(log_likelihood, x0, sigma0).run() 

