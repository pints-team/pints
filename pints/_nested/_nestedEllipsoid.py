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
import numpy.linalg as la
from pints._nested._nestedRejection import rejectSamplePrior

class NestedEllipsoidSampler(pints.NestedSampler):
    """
    *Extends:* :class:`nestedSampler`

    Creates a nested sampler that estimates the marginal likelihood
    and generates samples from the posterior.

    This is the form of nested sampler described in [1], where an ellipsoid is drawn around surviving
    particles (typically with an enlargement factor to avoid missing prior mass), and then random samples
    are drawn from within the bounds of the ellipsoid. By sampling in the space of surviving particles, the
    efficiency of this algorithm should be better than simple rejection sampling.

    [1] "A NESTED SAMPLING ALGORITHM FOR COSMOLOGICAL MODEL SELECTION", Pia Mukherjee, David Parkinson, Andrew R. Liddle, 2008. arXiv: arXiv:astro-ph/0508461v2 11 Jan 2006

    """
    def __init__(self, log_likelihood, aPrior):
        super(NestedEllipsoidSampler, self).__init__(log_likelihood,aPrior)

        # Target acceptance rate
        self._activePoints = 1000

        # Total number of iterations
        self._iterations = 1000

        # Total number of posterior samples
        self._numPosteriorSamples = 1000

        # Enlargement factor for ellipsoid
        self._enlargementFactor = 1.5

        # Number of nested rejection samples before starting ellipsoidal sampling
        self._rejectionSamples = 1000

        # Gaps between updating ellipsoid
        self._ellipsoidUpdateGap = 20

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
            print('Enlargement factor: ' + str(self._enlargementFactor))
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

        print 'Starting rejection sampling...'
        ## run iter
        for i in range(0,self._iterations):
            aRunningLogLikelihood = np.min(mActive[:,d])
            aMinIndex = np.argmin(mActive[:,d])
            X[i+1] = np.exp(-(i+1.0)/self._activePoints)
            w[i] = X[i] - X[i+1]
            vLogZ[i] = aRunningLogLikelihood
            mInactive[i,:] = mActive[aMinIndex,:]

            if (i+1) % self._rejectionSamples == 0:
                print 'Starting ellipsoidal sampling (finished rejection)...'
                A, centroid = mvee(mActive[:,:d])
            if i > self._rejectionSamples:
                if (i + 1 -self._rejectionSamples) % self._ellipsoidUpdateGap == 0:
                    print str(i+1 - self._rejectionSamples) + ' ellipsoidal samples completed (updating ellipsoid)...'
                    A, centroid = mvee(mActive[:,:d])
            if i < self._rejectionSamples:
                mActive[aMinIndex,:] = rejectSamplePrior(aRunningLogLikelihood,self._log_likelihood,self._prior)
            else:
                mActive[aMinIndex,:] = rejectEllipsoidSample_faster(aRunningLogLikelihood,self._log_likelihood,mActive[:,:d],self._enlargementFactor,A,centroid)

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

    def set_enlargementFactor(self,enlargementFactor):
        if enlargementFactor < 1:
            raise ValueError('Enlargement factor must exceed 1')
        self._enlargementFactor = enlargementFactor

    def set_rejectionSamples(self,rejectionSamples):
        if rejectionSamples < 0:
            raise ValueError('Must have non-negative rejection samples')
        if rejectionSamples > self._iterations:
            raise ValueError('Must have fewer rejection samples than total samples')
        self._rejectionSamples = rejectionSamples
    def set_ellipsoidUpdateGap(self,ellipsoidUpdateGap):
        if ellipsoidUpdateGap < 1:
            raise ValueError('Ellipsoid update gap must exceed 1')
        if isinstance(ellipsoidUpdateGap,int) == False:
            raise ValueError('Ellipsoid update gap must be an integer')
        self._ellipsoidUpdateGap = ellipsoidUpdateGap


def rejectDrawFromEllipsoid(A,centroid,aLogLikelihood,aThreshold):
    """draws a random point from within ellipsoid and accepts it if log-likelihood
        exceeds threshold"""
    vProposed = drawFromEllipsoid(A, centroid, 1)[0]
    aTempLogLikelihood = aLogLikelihood(vProposed)
    while aTempLogLikelihood < aThreshold:
        vProposed = drawFromEllipsoid(A, centroid, 1)[0]
        aTempLogLikelihood = aLogLikelihood(vProposed)
    return np.concatenate((vProposed,np.array([aTempLogLikelihood])))

def rejectUniformDraw(aMin,aMax,aLogLikelihood,aThreshold):
    """ equivalent to rejectDrawFromEllipsoid but in 1D"""
    aProposed = np.random.uniform(aMin,aMax,1)
    while aLogLikelihood(aProposed) < aThreshold:
        aProposed = np.random.uniform(aMin,aMax,1)
    return np.concatenate((aProposed,np.array([aLogLikelihood(aProposed)])))

def rejectEllipsoidSample(aThreshold,aLogLikelihood,mSamplesPrevious,enlargementFactor):
    """independently samples params from the prior until
        logLikelihood(params) > aThreshold"""
    aDim = len(mSamplesPrevious.shape)
    if aDim > 1:
        A, centroid = mvee(mSamplesPrevious)
        A = (1.0 / enlargementFactor) * A
        vSample = rejectDrawFromEllipsoid(la.inv(A),centroid,aLogLikelihood,aThreshold)
    else:
        aMin = np.min(mSamplesPrevious)
        aMax = np.max(mSamplesPrevious)
        aMiddle = (aMin + aMax) / 2.0
        aDiff = aMax - aMin
        aDiff = aDiff * enlargementFactor
        aMin = aMiddle - (aDiff / 2.0)
        aMax = aMiddle + (aDiff / 2.0)
        vSample = rejectUniformDraw(aMin,aMax,aLikelihood,aX,aN)
    return vSample

def rejectEllipsoidSample_faster(aThreshold,aLogLikelihood,mSamplesPrevious,enlargementFactor,A,centroid):
    """independently samples params from the prior until
        logLikelihood(params) > aThreshold. Accepts A as input (which is only updated every N steps)."""
    aDim = len(mSamplesPrevious.shape)
    if aDim > 1:
        A = (1.0 / enlargementFactor) * A
        vSample = rejectDrawFromEllipsoid(la.inv(A),centroid,aLogLikelihood,aThreshold)
    else:
        aMin = np.min(mSamplesPrevious)
        aMax = np.max(mSamplesPrevious)
        aMiddle = (aMin + aMax) / 2.0
        aDiff = aMax - aMin
        aDiff = aDiff * enlargementFactor
        aMin = aMiddle - (aDiff / 2.0)
        aMax = aMiddle + (aDiff / 2.0)
        vSample = rejectUniformDraw(aMin,aMax,aLikelihood,aX,aN)
    return vSample


def drawFromEllipsoid( covmat, cent, npts):
    """ draw npts random uniform points from within an ellipsoid
        with a covariance matrix covmat and a centroid cent, as per: http://www.astro.gla.ac.uk/~matthew/blog/?p=368"""
    try:
        ndims = covmat.shape[0]
    except IndexError:
        ndims = 1

    # calculate eigenvalues (e) and eigenvectors (v)
    eigenValues, eigenVectors = np.linalg.eig(covmat)
    idx = (-eigenValues).argsort()[::-1][:ndims]
    e = eigenValues[idx]
    v = eigenVectors[:,idx]
    e = np.diag(e)

    # generate radii of hyperspheres
    rs = np.random.uniform(0,1,npts)

    # generate points
    pt = np.random.normal(0,1,[npts,ndims]);

    # get scalings for each point onto the surface of a unit hypersphere
    fac = np.sum(pt**2,axis=1)

    # calculate scaling for each point to be within the unit hypersphere
    # with radii rs
    fac = (rs**(1.0/ndims)) / np.sqrt(fac)

    pnts = np.zeros((npts,ndims));

    # scale points to the ellipsoid using the eigenvalues and rotate with
    # the eigenvectors and add centroid
    d = np.sqrt(np.diag(e))
    d.shape = (ndims,1)

    for i in range(0,npts):
        # scale points to a uniform distribution within unit hypersphere
        pnts[i,:] = fac[i]*pt[i,:]
        pnts[i,:] = np.dot(np.multiply(pnts[i,:],np.transpose(d)),np.transpose(v)) + cent

    return pnts


def mvee(points, tol = 0.001):
    """
        Finds the ellipse equation in "center form"
        (x-c).T * A * (x-c) = 1
        """
    try:
        N, d = points.shape
    except ValueError:
        N = points.shape[0]
        d = 1

    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u,points)
    if d > 1:
        A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
                   - np.multiply.outer(c,c))/d
    else:
        A = (np.dot(np.dot(points.T, np.diag(u)), points)
             - np.multiply.outer(c,c))**(-1.0)
    return A, c

