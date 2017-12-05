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
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy
import scipy.linalg
import multiprocessing
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
        self._activePoints = 1000

        # Total number of iterations
        self._iterations = 1000

        # Total number of posterior samples
        self._numPosteriorSamples = 1000

    def activePoints_rate(self):
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
            print('Number of active points: ' + str(self._activePoints))
            print('Total number of iterations: ' + str(self._iterations))
            print('Total number of posterior samples: ' + str(self._numPosteriorSamples))

        # Problem dimension
        d = self._dimension

        # go!
        ## generate initial random points by sampling from the prior
        mActive = np.zeros((self._activePoints, d + 1))
        mInitial = self._prior.randomSample(self._activePoints)
        for i in range(0,self._activePoints):
            mActive[i,d] = self._log_likelihood(mInitial[i,:])
        mActive[:,:-1] = mInitial

        ## store all inactive points, along with their respective log-likelihoods (hence, d+1)
        mInactive = np.zeros((self._iterations, d + 1))

        ## store weights
        w = np.zeros(self._activePoints + self._iterations)

        ## store X values (defined in [1])
        X = np.zeros(self._iterations+1)
        X[0] = 1

        ## log marginal likelihood holder
        vLogZ = np.zeros(self._iterations+1)

        ## run iter
        for i in range(0,self._iterations):
            aRunningLogLikelihood = np.min(mActive[:,d])
            aMinIndex = np.argmin(mActive[:,d])
            X[i+1] = np.exp(-(i+1.0)/self._activePoints)
            w[i] = X[i] - X[i+1]
            vLogZ[i] = aRunningLogLikelihood
            mInactive[i,:] = mActive[aMinIndex,:]
            mActive[aMinIndex,:] = rejectSamplePrior(aRunningLogLikelihood,self._log_likelihood,self._prior)

        vLogZ[self._iterations] = logsumexp(mActive[:,d])
        w[self._iterations:] = float(X[self._iterations]) / float(self._activePoints)
        mSamplesAll = np.vstack((mInactive,mActive))
        logZ = logsumexp(vLogZ, b = w[0:(self._iterations+1)])
        vLogP = mSamplesAll[:,d] - logZ + np.log(w)
        vP = np.exp(mSamplesAll[:,d] - logZ) * w
        mTheta = mSamplesAll[:,:-1]
        vIndex = np.random.choice(range(0,(self._iterations + self._activePoints)), self._numPosteriorSamples, p=vP)
        mPosteriorSamples = mTheta[vIndex,:]
        return [mPosteriorSamples, logZ]

    def set_activePoints_rate(self, activePoints):
        """
        Sets the number of active points for the next run.
        """
        activePoints = float(activePoints)
        if activePoints <= 5:
            raise ValueError('Number of active points must be greater than 5.')
        self._activePoints = activePoints

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run
        (including burn-in and non-adaptive iterations).
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_numPosteriorSamples(self,numPosteriorSamples):

        if numPosteriorSamples > np.round(0.25 * (self._iterations + self._activePoints)):
            raise ValueError('Number of posterior samples must be fewer than 25% the total number of preminary points')
        self._numPosteriorSamples = numPosteriorSamples

## independently samples params from the prior until logLikelihood(params) > aThreshold
def rejectSamplePrior(aThreshold,aLogLikelihood,aPrior):
    vProposed = aPrior.randomSample()[0]
    while aLogLikelihood(vProposed) < aThreshold:
        vProposed = aPrior.randomSample()[0]
    return np.concatenate((vProposed,np.array([aLogLikelihood(vProposed)])))

