#
# Nested ellipsoidal sampler implementation.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
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


class NestedEllipsoidSampler(pints.NestedSampler):
    """
    Creates a nested sampler that estimates the marginal likelihood
    and generates samples from the posterior.

    This is the form of nested sampler described in [1], where an ellipsoid is
    drawn around surviving particles (typically with an enlargement factor to
    avoid missing prior mass), and then random samples are drawn from within
    the bounds of the ellipsoid. By sampling in the space of surviving
    particles, the efficiency of this algorithm should be better than simple
    rejection sampling.

    *Extends:* :class:`NestedSampler`

    [1] "A nested sampling algorithm for cosmological model selection",
    Pia Mukherjee, David Parkinson, Andrew R. Liddle, 2008.
    arXiv: arXiv:astro-ph/0508461v2 11 Jan 2006
    """
    def __init__(self, log_likelihood, log_prior):
        super(NestedEllipsoidSampler, self).__init__(log_likelihood, log_prior)

        # Target acceptance rate
        self._active_points = 0
        self.set_active_points_rate()

        # Total number of iterations
        self._iterations = 0
        self.set_iterations()

        # Total number of posterior samples
        self._posterior_samples = 0
        self.set_posterior_samples()

        # Number of nested rejection samples before starting ellipsoidal
        # sampling
        self._rejection_samples = 0
        self.set_rejection_samples()

        # Gaps between updating ellipsoid
        self._ellipsoid_update_gap = 0
        self.set_ellipsoid_update_gap()

        # Enlargement factor for ellipsoid
        self._enlargement_factor = 0
        self.set_enlargement_factor()

        # Total number of log_likelihood evaluations
        self._n_evals = 0

    def active_points_rate(self):
        """
        Returns the number of active points that will be used in next run (see
        :meth:`set_active_points_rate()`).
        """
        return self._active_points

    def ellipsoid_update_gap(self):
        """
        Returns the ellipsoid update gap used in the algorithm (see
        :meth:`set_ellipsoid_update_gap()`).
        """
        return self._ellipsoid_update_gap

    def enlargement_factor(self):
        """
        Returns the enlargement factor used in the algorithm (see
        :meth:`set_enlargement_factor()`).
        """
        return self._enlargement_factor

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run (see :meth:`set_iterations()`).
        """
        return self._iterations

    def posterior_samples(self):
        """
        Returns the number of posterior samples that will be returned (see
        :meth:`set_posterior_samples()`).
        """
        return self._posterior_samples

    def rejection_samples(self):
        """
        Returns the number of rejection sample used in the algorithm (see
        :meth:`set_rejection_samples()`).
        """
        return self._rejection_samples

    def run(self):
        """ See :meth:`pints.MCMC.run()`. """

        # Reset total number of log_likelihood evaluations
        self._n_evals = 0

        # Check if settings make sense
        max_post = 0.25 * (self._iterations + self._active_points)
        if self._posterior_samples > max_post:
            raise ValueError(
                'Number of posterior samples must not exceed 0.25 times (the'
                ' number of iterations + the number of active points).')
        if self._rejection_samples > self._iterations:
            raise ValueError(
                'Number of rejection samples must not exceed number of'
                ' iterations.')

        # Set up progress reporting
        next_message = 0
        message_warm_up = 3
        message_interval = 50

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            # Create timer
            timer = pints.Timer()

            if self._log_to_screen:
                # Show current settings
                print('Running nested rejection sampling')
                print('Number of active points: ' + str(self._active_points))
                print('Total number of iterations: ' + str(self._iterations))
                print('Enlargement factor: ' + str(self._enlargement_factor))
                print('Total number of posterior samples: ' + str(
                    self._posterior_samples))

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            logger.add_counter('Iter.', max_value=self._iterations)
            logger.add_counter('Eval.', max_value=self._iterations * 10)
            #TODO: Add other informative fields ?
            logger.add_time('Time m:s')

        # Problem dimension
        d = self._dimension

        # Generate initial random points by sampling from the prior
        m_active = np.zeros((self._active_points, d + 1))
        m_initial = self._log_prior.sample(self._active_points)
        for i in range(0, self._active_points):
            # Evaluate log likelihood
            m_active[i, d] = self._log_likelihood(m_initial[i, :])
            self._n_evals += 1

            # Show progress
            if logging and i >= next_message:
                # Log state
                logger.log(0, self._n_evals, timer.time())

                # Choose next logging point
                if i > message_warm_up:
                    next_message = message_interval * (
                        1 + i // message_interval)

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

        # Run
        i_message = self._active_points - 1
        for i in range(0, self._iterations):

            a_running_log_likelihood = np.min(m_active[:, d])
            a_min_index = np.argmin(m_active[:, d])
            X[i + 1] = np.exp(-(i + 1.0) / self._active_points)
            w[i] = X[i] - X[i + 1]
            v_log_Z[i] = a_running_log_likelihood
            m_inactive[i, :] = m_active[a_min_index, :]

            if (i + 1) % self._rejection_samples == 0:
                A, centroid = self._minimum_volume_ellipsoid(m_active[:, :d])

            if i > self._rejection_samples:
                if ((i + 1 - self._rejection_samples)
                        % self._ellipsoid_update_gap == 0):
                    A, centroid = self._minimum_volume_ellipsoid(
                        m_active[:, :d])

            if i < self._rejection_samples:
                # Start off with rejection sampling, while this is still very
                # efficient.
                m_active[a_min_index, :] = self._reject_sample_prior(
                    a_running_log_likelihood)
            else:
                # After a number of samples, switch to ellipsoid sampling.
                m_active[a_min_index, :] = \
                    self._reject_ellipsoid_sample_faster(
                        a_running_log_likelihood, m_active[:, :d],
                        self._enlargement_factor, A, centroid)

            # Show progress
            if logging:
                i_message += 1
                if i_message >= next_message:
                    # Log state
                    logger.log(i_message, self._n_evals, timer.time())

                    # Choose next logging point
                    if i_message > message_warm_up:
                        next_message = message_interval * (
                            1 + i_message // message_interval)

        v_log_Z[self._iterations] = logsumexp(m_active[:, d])
        w[self._iterations:] = \
            float(X[self._iterations]) / float(self._active_points)
        m_samples_all = np.vstack((m_inactive, m_active))
        logZ = logsumexp(v_log_Z, b=w[0:(self._iterations + 1)])

        vP = np.exp(m_samples_all[:, d] - logZ) * w
        mTheta = m_samples_all[:, :-1]
        vIndex = np.random.choice(
            range(0, self._iterations + self._active_points),
            self._posterior_samples, p=vP)
        m_posterior_samples = mTheta[vIndex, :]

        return m_posterior_samples, logZ

    def set_active_points_rate(self, active_points=1000):
        """
        Sets the number of active points for the next run.
        """
        active_points = int(active_points)
        if active_points <= 5:
            raise ValueError('Number of active points must be greater than 5.')
        self._active_points = active_points

    def set_iterations(self, iterations=1000):
        """
        Sets the total number of iterations to be performed in the next run.
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_posterior_samples(self, posterior_samples=1000):
        """
        Sets the number of posterior samples to generate from points proposed
        by the nested sampling algorithm.
        """
        posterior_samples = int(posterior_samples)
        if posterior_samples < 1:
            raise ValueError(
                'Number of posterior samples must be greater than zero.')
        self._posterior_samples = posterior_samples

    def set_enlargement_factor(self, enlargement_factor=1.5):
        """
        Sets the factor (>1) by which to increase the minimal volume
        ellipsoidal in rejection sampling. A higher value means it is less
        likely that areas of high probability mass will be missed. A low value
        means that rejection sampling is more efficient.
        """
        if enlargement_factor <= 1:
            raise ValueError('Enlargement factor must exceed 1.')
        self._enlargement_factor = enlargement_factor

    def set_rejection_samples(self, rejection_samples=1000):
        """
        Sets the number of rejection samples to take, which will be assigned
        weights and ultimately produce a set of posterior samples.
        """
        if rejection_samples < 0:
            raise ValueError('Must have non-negative rejection samples.')
        self._rejection_samples = rejection_samples

    def set_ellipsoid_update_gap(self, ellipsoid_update_gap=20):
        """
        Sets the frequency with which the minimum volume ellipsoid is
        re-estimated as part of the nested rejection sampling algorithm. A
        higher rate of this parameter means each sample will be more
        efficiently produced, yet the cost of re-computing the ellipsoid means
        it is often desirable to compute this every n iterates.
        """
        ellipsoid_update_gap = int(ellipsoid_update_gap)
        if ellipsoid_update_gap <= 1:
            raise ValueError('Ellipsoid update gap must exceed 1.')
        self._ellipsoid_update_gap = ellipsoid_update_gap

    def _minimum_volume_ellipsoid(self, points, tol=0.001):
        """
        Finds the ellipse equation in "center form":
        ``(x-c).T * A * (x-c) = 1``.
        """
        N, d = points.shape
        Q = np.column_stack((points, np.ones(N))).T
        err = tol + 1
        u = np.ones(N) / N
        while err > tol:
            # assert(u.sum() == 1) # invariant
            X = np.dot(np.dot(Q, np.diag(u)), Q.T)
            M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1) / ((d + 1) * (M[jdx] - 1))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u
        c = np.dot(u, points)
        A = la.inv(
            + np.dot(np.dot(points.T, np.diag(u)), points)
            - np.multiply.outer(c, c)
        ) / d
        return A, c

    def _reject_sample_prior(self, threshold):
        """
        Independently samples params from the prior until
        ``log_likelihood(params) > threshold``.
        """
        # Note: threshold can be -inf, so that while loop is never run.
        proposed = self._log_prior.sample()[0]
        log_likelihood = self._log_likelihood(proposed)
        self._n_evals += 1
        while log_likelihood < threshold:
            proposed = self._log_prior.sample()[0]
            log_likelihood = self._log_likelihood(proposed)
            self._n_evals += 1
        return np.concatenate((proposed, np.array([log_likelihood])))

    def _reject_ellipsoid_sample_faster(
            self, threshold, m_samples_previous, enlargement_factor, A,
            centroid):
        """
        Independently samples params from the prior until
        ``logLikelihood(params) > threshold``. Accepts ``A`` as input (which is
        only updated every ``N`` steps).
        """
        return self._reject_draw_from_ellipsoid(
            la.inv((1 / enlargement_factor) * A), centroid, threshold)

    def _reject_draw_from_ellipsoid(self, A, centroid, threshold):
        """
        Draws a random point from within ellipsoid and accepts it if
        log-likelihood exceeds threshold.
        """
        # Note: threshold can be -inf, so that while loop is never run.
        proposed = self._draw_from_ellipsoid(A, centroid, 1)[0]
        log_likelihood = self._log_likelihood(proposed)
        self._n_evals += 1
        while log_likelihood < threshold:
            proposed = self._draw_from_ellipsoid(A, centroid, 1)[0]
            log_likelihood = self._log_likelihood(proposed)
            self._n_evals += 1
        return np.concatenate((proposed, np.array([log_likelihood])))

    def _draw_from_ellipsoid(self, covmat, cent, npts):
        """
        Draw `npts` random uniform points from within an ellipsoid with a
        covariance matrix covmat and a centroid cent, as per:
        http://www.astro.gla.ac.uk/~matthew/blog/?p=368
        """
        try:
            ndims = covmat.shape[0]
        except IndexError:  # pragma: no cover
            ndims = 1

        # calculate eigen_values (e) and eigen_vectors (v)
        eigen_values, eigen_vectors = la.eig(covmat)
        idx = (-eigen_values).argsort()[::-1][:ndims]
        e = eigen_values[idx]
        v = eigen_vectors[:, idx]
        e = np.diag(e)

        # generate radii of hyperspheres
        rs = np.random.uniform(0, 1, npts)

        # generate points
        pt = np.random.normal(0, 1, [npts, ndims])

        # get scalings for each point onto the surface of a unit hypersphere
        fac = np.sum(pt**2, axis=1)

        # calculate scaling for each point to be within the unit hypersphere
        # with radii rs
        fac = (rs**(1 / ndims)) / np.sqrt(fac)
        pnts = np.zeros((npts, ndims))

        # scale points to the ellipsoid using the eigen_values and rotate with
        # the eigen_vectors and add centroid
        d = np.sqrt(np.diag(e))
        d.shape = (ndims, 1)

        for i in range(0, npts):
            # scale points to a uniform distribution within unit hypersphere
            pnts[i, :] = fac[i] * pt[i, :]
            pnts[i, :] = np.dot(
                np.multiply(pnts[i, :], np.transpose(d)),
                np.transpose(v)
            ) + cent

        return pnts


#TODO: THIS METHOD IS NEVER USED
'''
def _reject_ellipsoid_sample(
        threshold, log_likelihood, m_samples_previous, enlargement_factor):
    """
    Independently samples params from the prior until
    `logLikelihood(params) > threshold`.
    """
    aDim = len(m_samples_previous.shape)
    if aDim > 1:
        A, centroid = _minimum_volume_ellipsoid(m_samples_previous)
        A = (1 / enlargement_factor) * A
        return _reject_draw_from_ellipsoid(
            la.inv(A), centroid, log_likelihood, threshold)
    else:
        a_min = np.min(m_samples_previous)
        a_max = np.max(m_samples_previous)
        a_middle = (a_min + a_max) / 2
        a_diff = a_max - a_min
        a_diff = a_diff * enlargement_factor
        a_min = a_middle - (a_diff / 2)
        a_max = a_middle + (a_diff / 2)
        return _reject_uniform_draw(a_min, a_max, log_likelihood, threshold)
'''
