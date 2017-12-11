#
# Nested ellipsoidal sampler implementation.
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
import numpy.linalg as la
from scipy.misc import logsumexp
from pints._nested import reject_sample_prior


class NestedEllipsoidSampler(pints.NestedSampler):
    """
    *Extends:* :class:`nestedSampler`

    Creates a nested sampler that estimates the marginal likelihood
    and generates samples from the posterior.

    This is the form of nested sampler described in [1], where an ellipsoid is
    drawn around surviving particles (typically with an enlargement factor to
    avoid missing prior mass), and then random samples are drawn from within
    the bounds of the ellipsoid. By sampling in the space of surviving
    particles, the efficiency of this algorithm should be better than simple
    rejection sampling.

    [1] "A NESTED SAMPLING ALGORITHM FOR COSMOLOGICAL MODEL SELECTION", Pia
        Mukherjee, David Parkinson, Andrew R. Liddle, 2008.
        arXiv: arXiv:astro-ph/0508461v2 11 Jan 2006

    """
    def __init__(self, log_likelihood, prior):
        super(NestedEllipsoidSampler, self).__init__(log_likelihood, prior)

        # Target acceptance rate
        self._active_points = 1000

        # Total number of iterations
        self._iterations = 1000

        # Total number of posterior samples
        self._posterior_samples = 1000

        # Enlargement factor for ellipsoid
        self._enlargement_factor = 1.5

        # Number of nested rejection samples before starting ellipsoidal
        # sampling
        self._rejection_samples = 1000

        # Gaps between updating ellipsoid
        self._ellipsoid_update_gap = 20

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
            print('Enlargement factor: ' + str(self._enlargement_factor))
            print('Total number of posterior samples: ' + str(
                self._posterior_samples))

        # Problem dimension
        d = self._dimension

        # go!
        # generate initial random points by sampling from the prior
        m_active = np.zeros((self._active_points, d + 1))
        m_initial = self._prior.random_sample(self._active_points)
        for i in range(0, self._active_points):
            m_active[i, d] = self._log_likelihood(m_initial[i, :])
        m_active[:, :-1] = m_initial

        # store all inactive points, along with their respective
        # log-likelihoods (hence, d+1)
        m_inactive = np.zeros((self._iterations, d + 1))

        # store weights
        w = np.zeros(self._active_points + self._iterations)

        # store X values (defined in [1])
        X = np.zeros(self._iterations + 1)
        X[0] = 1

        # log marginal likelihood holder
        v_log_Z = np.zeros(self._iterations + 1)
        if self._verbose:
            print('Starting rejection sampling...')

        # run iter
        for i in range(0, self._iterations):
            a_running_log_likelihood = np.min(m_active[:, d])
            a_min_index = np.argmin(m_active[:, d])
            X[i + 1] = np.exp(-(i + 1.0) / self._active_points)
            w[i] = X[i] - X[i + 1]
            v_log_Z[i] = a_running_log_likelihood
            m_inactive[i, :] = m_active[a_min_index, :]

            if (i + 1) % self._rejection_samples == 0:
                if self._verbose:
                    print('Starting ellipsoidal sampling (finished'
                          ' rejection)...')
                A, centroid = minimum_volume_ellipsoid(m_active[:, :d])
            if i > self._rejection_samples:
                if ((i + 1 - self._rejection_samples)
                        % self._ellipsoid_update_gap == 0):
                    if self._verbose:
                        print(str(i + 1 - self._rejection_samples)
                              + ' ellipsoidal samples completed (updating'
                                ' ellipsoid)...')
                    A, centroid = minimum_volume_ellipsoid(m_active[:, :d])

            if i < self._rejection_samples:
                m_active[a_min_index, :] = reject_sample_prior(
                    a_running_log_likelihood, self._log_likelihood,
                    self._prior)
            else:
                m_active[a_min_index, :] = reject_ellipsoid_sample_faster(
                    a_running_log_likelihood, self._log_likelihood,
                    m_active[:, :d], self._enlargement_factor, A, centroid)

        v_log_Z[self._iterations] = logsumexp(m_active[:, d])
        w[self._iterations:] = float(X[self._iterations]) / float(
            self._active_points)
        m_samples_all = np.vstack((m_inactive, m_active))
        logZ = logsumexp(v_log_Z, b=w[0:(self._iterations + 1)])
        # vLogP = m_samples_all[:, d] - logZ + np.log(w)
        vP = np.exp(m_samples_all[:, d] - logZ) * w
        mTheta = m_samples_all[:, :-1]
        vIndex = np.random.choice(
            range(0, self._iterations + self._active_points),
            self._posterior_samples, p=vP)
        m_posterior_samples = mTheta[vIndex, :]
        return [m_posterior_samples, logZ]

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

    def set_posterior_samples(self, posterior_samples):
        """
        Sets the number of posterior samples to generate from points proposed
        from nested sampling algorithm.
        """
        if posterior_samples > np.round(0.25 * (self._iterations
                + self._active_points)):
            raise ValueError('Number of posterior samples must be fewer than'
                ' 25% the total number of preminary points')
        self._posterior_samples = posterior_samples

    def set_enlargement_factor(self, enlargement_factor):
        """
        Sets the factor (>1) by which to increase the minimal volume
        ellipsoidal in rejection sampling. A higher value means it is less
        likely that areas of high probability mass will be missed. A low value
        means that rejection sampling is more efficient.
        """
        if enlargement_factor < 1:
            raise ValueError('Enlargement factor must exceed 1')
        self._enlargement_factor = enlargement_factor

    def set_rejection_samples(self, rejection_samples):
        """
        Sets the number of rejection samples to take, which will be assigned
        weights and ultimately produce a set of posterior samples.
        """
        if rejection_samples < 0:
            raise ValueError('Must have non-negative rejection samples')
        if rejection_samples > self._iterations:
            raise ValueError('Must have fewer rejection samples than total'
                ' samples')
        self._rejection_samples = rejection_samples

    def set_ellipsoid_update_gap(self, ellipsoid_update_gap):
        """
        Sets the frequency with which the minimum volume ellipsoid is
        re-estimated as part of the nested rejection sampling algorithm. A
        higher rate of this parameter means each sample will be more
        efficiently produced, yet the cost of re-computing the ellipsoid means
        it is often desirable to compute this every n iterates.
        """
        if ellipsoid_update_gap < 1:
            raise ValueError('Ellipsoid update gap must exceed 1')
        if isinstance(ellipsoid_update_gap,int) == False:
            raise ValueError('Ellipsoid update gap must be an integer')
        self._ellipsoid_update_gap = ellipsoid_update_gap


def reject_draw_from_ellipsoid(A, centroid, log_likelihood, threshold):
    """
    Draws a random point from within ellipsoid and accepts it if log-likelihood
    exceeds threshold.
    """
    v_proposed = draw_from_ellipsoid(A, centroid, 1)[0]
    temp_log_likelihood = log_likelihood(v_proposed)
    while temp_log_likelihood < threshold:
        v_proposed = draw_from_ellipsoid(A, centroid, 1)[0]
        temp_log_likelihood = log_likelihood(v_proposed)
    return np.concatenate((v_proposed, np.array([temp_log_likelihood])))


def reject_uniform_draw(a_min,a_max,aLogLikelihood,threshold):
    """
    Equivalent to reject_draw_from_ellipsoid but in 1D.
    """
    a_proposed = np.random.uniform(a_min, a_max, 1)
    while aLogLikelihood(a_proposed) < threshold:
        a_proposed = np.random.uniform(a_min, a_max, 1)
    return np.concatenate((a_proposed,np.array([aLogLikelihood(a_proposed)])))


def reject_ellipsoid_sample(threshold, log_likelihood, m_samples_previous,
        enlargement_factor):
    """
    Independently samples params from the prior until
    `logLikelihood(params) > threshold`.
    """
    aDim = len(m_samples_previous.shape)
    if aDim > 1:
        A, centroid = minimum_volume_ellipsoid(m_samples_previous)
        A = (1.0 / enlargement_factor) * A
        v_sample = reject_draw_from_ellipsoid(la.inv(A),centroid,
            log_likelihood, threshold)
    else:
        a_min = np.min(m_samples_previous)
        a_max = np.max(m_samples_previous)
        a_middle = (a_min + a_max) / 2
        a_diff = a_max - a_min
        a_diff = a_diff * enlargement_factor
        a_min = a_middle - (a_diff / 2)
        a_max = a_middle + (a_diff / 2)
        v_sample = reject_uniform_draw(a_min, a_max, a_likelihood, aX, aN)
    return v_sample


def reject_ellipsoid_sample_faster(threshold, log_likelihood,
        m_samples_previous, enlargement_factor, A, centroid):
    """
    Independently samples params from the prior until
    `logLikelihood(params) > threshold`. Accepts A as input (which is only
    updated every N steps).
    """
    aDim = len(m_samples_previous.shape)
    if aDim > 1:
        A = (1.0 / enlargement_factor) * A
        v_sample = reject_draw_from_ellipsoid(la.inv(A), centroid,
            log_likelihood, threshold)
    else:
        a_min = np.min(m_samples_previous)
        a_max = np.max(m_samples_previous)
        a_middle = (a_min + a_max) / 2
        a_diff = a_max - a_min
        a_diff = a_diff * enlargement_factor
        a_min = a_middle - (a_diff / 2)
        a_max = a_middle + (a_diff / 2)
        v_sample = reject_uniform_draw(a_min, a_max, a_likelihood, aX, aN)
    return v_sample


def draw_from_ellipsoid( covmat, cent, npts):
    """
    Draw `npts` random uniform points from within an ellipsoid with a
    covariance matrix covmat and a centroid cent, as per:
    http://www.astro.gla.ac.uk/~matthew/blog/?p=368
    """
    try:
        ndims = covmat.shape[0]
    except IndexError:
        ndims = 1

    # calculate eigen_values (e) and eigen_vectors (v)
    eigen_values, eigen_vectors = la.eig(covmat)
    idx = (-eigen_values).argsort()[::-1][:ndims]
    e = eigen_values[idx]
    v = eigen_vectors[:,idx]
    e = np.diag(e)

    # generate radii of hyperspheres
    rs = np.random.uniform(0, 1, npts)

    # generate points
    pt = np.random.normal(0, 1, [npts, ndims]);

    # get scalings for each point onto the surface of a unit hypersphere
    fac = np.sum(pt**2, axis=1)

    # calculate scaling for each point to be within the unit hypersphere
    # with radii rs
    fac = (rs**(1 / ndims)) / np.sqrt(fac)

    pnts = np.zeros((npts,ndims));

    # scale points to the ellipsoid using the eigen_values and rotate with
    # the eigen_vectors and add centroid
    d = np.sqrt(np.diag(e))
    d.shape = (ndims, 1)

    for i in range(0, npts):
        # scale points to a uniform distribution within unit hypersphere
        pnts[i,:] = fac[i] * pt[i, :]
        pnts[i,:] = np.dot(np.multiply(pnts[i, :], np.transpose(d)),
            np.transpose(v)) + cent

    return pnts


def minimum_volume_ellipsoid(points, tol = 0.001):
    """
    Finds the ellipse equation in "center form": `(x-c).T * A * (x-c) = 1`.
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
        step_size = (M[jdx] - d - 1) / ((d + 1) * (M[jdx] - 1))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = la.norm(new_u - u)
        u = new_u
    c = np.dot(u,points)
    if d > 1:
        A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
                   - np.multiply.outer(c, c))/d
    else:
        A = 1 / (np.dot(np.dot(points.T, np.diag(u)), points)
             - np.multiply.outer(c, c))
    return A, c

