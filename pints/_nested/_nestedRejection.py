#
# Nested rejection sampler implementation.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy
import scipy.linalg
from scipy.misc import logsumexp

class NestedRejectionSampler(pints.NestedSampler):
    """
    *Extends:* :class:`nestedSampler`

    Creates a nested sampler that estimates the marginal likelihood
    and generates samples from the posterior.

    This is the simplest form of nested sampler and involves using
    rejection sampling from the prior as described in the algorithm on page 839
    in [1] to estimate the marginal likelihood and generate weights, preliminary samples (with their
    respective likelihoods), required to generate posterior samples.

    The posterior samples are generated as described in [1] on page 849 by randomly sampling the preliminary point,
    accounting for their weights and likelihoods.

    [1] "Nested Sampling for General Bayesian Computation", John Skilling, Bayesian Analysis 1:4 (2006).

    """
    def __init__(self, log_likelihood, aPrior):
        super(NestedRejectionSampler, self).__init__(log_likelihood,aPrior)

        # Target acceptance rate
        self._active_points = 1000

        # Total number of iterations
        self._iterations = 1000

        # Total number of posterior samples
        self._posterior_samples = 1000

    def active_points_rate(self):
        """
        Returns the number of active points that will be used in next run.
        """
        return self._acceptance_target

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
            print('Running nested rejection sampling')
            print('Number of active points: ' + str(self._active_points))
            print('Total number of iterations: ' + str(self._iterations))
            print('Total number of posterior samples: ' + str(self._posterior_samples))

        # Problem dimension
        d = self._dimension

        # go!
        ## generate initial random points by sampling from the prior
        m_active = np.zeros((self._active_points, d + 1))
        m_initial = self._prior.random_sample(self._active_points)
        for i in range(0,self._active_points):
            m_active[i,d] = self._log_likelihood(m_initial[i,:])
        m_active[:,:-1] = m_initial

        ## store all inactive points, along with their respective log-likelihoods (hence, d+1)
        m_inactive = np.zeros((self._iterations, d + 1))

        ## store weights
        w = np.zeros(self._active_points + self._iterations)

        ## store X values (defined in [1])
        X = np.zeros(self._iterations+1)
        X[0] = 1

        ## log marginal likelihood holder
        v_log_Z = np.zeros(self._iterations+1)

        ## run iter
        for i in range(0,self._iterations):
            a_running_log_likelihood = np.min(m_active[:,d])
            a_min_index = np.argmin(m_active[:,d])
            X[i+1] = np.exp(-(i+1.0)/self._active_points)
            w[i] = X[i] - X[i+1]
            v_log_Z[i] = a_running_log_likelihood
            m_inactive[i,:] = m_active[a_min_index,:]
            m_active[a_min_index,:] = reject_sample_prior(a_running_log_likelihood,self._log_likelihood,self._prior)

        v_log_Z[self._iterations] = logsumexp(m_active[:,d])
        w[self._iterations:] = float(X[self._iterations]) / float(self._active_points)
        m_samples_all = np.vstack((m_inactive,m_active))
        log_Z = logsumexp(v_log_Z, b = w[0:(self._iterations+1)])
        vLogP = m_samples_all[:,d] - log_Z + np.log(w)
        vP = np.exp(m_samples_all[:,d] - log_Z) * w
        m_theta = m_samples_all[:,:-1]
        vIndex = np.random.choice(range(0,(self._iterations + self._active_points)), self._posterior_samples, p=vP)
        m_posterior_samples = m_theta[vIndex,:]
        return [m_posterior_samples, log_Z]

    def set_active_points_rate(self, active_points):
        """
        Sets the number of active points for the next run.
        """
        active_points = float(active_points)
        if active_points <= 5:
            raise ValueError('Number of active points must be greater than 5.')
        self._active_points = active_points

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run
        (including burn-in and non-adaptive iterations).
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_posterior_samples(self,posterior_samples):

        if posterior_samples > np.round(0.25 * (self._iterations + self._active_points)):
            raise ValueError('Number of posterior samples must be fewer than 25% the total number of preminary points')
        self._posterior_samples = posterior_samples

## independently samples params from the prior until logLikelihood(params) > aThreshold
def reject_sample_prior(aThreshold,aLogLikelihood,aPrior):
    v_proposed = aPrior.random_sample()[0]
    while aLogLikelihood(v_proposed) < aThreshold:
        v_proposed = aPrior.random_sample()[0]
    return np.concatenate((v_proposed,np.array([aLogLikelihood(v_proposed)])))

