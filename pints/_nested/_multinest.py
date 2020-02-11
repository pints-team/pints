#
# Nested ellipsoidal sampler implementation.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy.special
import scipy.cluster.vq
import random


class MultinestSampler(pints.NestedSampler):
    r"""
    Creates a MultiNest nested sampler that estimates the marginal likelihood
    and generates samples from the posterior.

    This is the form of nested sampler described in [1]_, where multiple
    ellipsoids are drawn around surviving particles (typically with an
    enlargement factor to avoid missing prior mass), and then random samples
    are drawn from within the bounds of the ellipsoids (accounting for any
    overlap between them). By sampling in the space of surviving particles,
    the efficiency of this algorithm aims to improve upon simple rejection
    sampling. This algorithm has the following steps:

    Initialise::

        Z = 0

    Draw samples from prior::

        for i in 1:n_active_points:
            theta_i ~ p(theta), i.e. sample from the prior
            L_i = p(theta_i|X)
        endfor
        L_min = min(L)
        indexmin = min_index(L)

    Run rejection sampling for ``n_rejection_samples`` to generate
    an initial sample, along with updated values of ``L_min`` and
    ``indexmin``.

    Transform all active points into the unit cube via the cumulative
    distribution function of the priors:

    .. math::
        u_i = \int_{-\infty}^{\theta_i} \pi(\theta') d\theta'

    Fit transformed active points using minimum volume bounding ellipsoids
    (that potentially overlap) as described by Algorithm 1 in [1]_.
    Explicitly, this involves the following steps (which we term
    ``f_s_minimisation`` in what follows)::

        f_s_minimisation(t, u):
            calculate bounding ellipsoid E and its volume V(E)
            V(S) = exp(-t/n_active_points); t is iteration and
                S is prior volume remaining
            enlarge E so that V(E) = max(V(E), V(S))
            using k-means algorithm partition S into S_1 and S_2 containing n_1
                and n_2 points
            (A) find E_1 and E_2 (bounding ellipsoids) and their volumes V(E_1)
                and V(E_2)
            enlarge E_k (k=1,2) so that V(E_k) = max(V(E_k), V(S_k)),
                where V(S_k) = n_k V(S) / n_active_points
            for all active points:
                assign u_i to S_k such that h_k(u_i) = min(h_1(u_i), h_2(u_i))
            endfor
            where h_k(u_i) = (V(E_k) / V(S_k)) * d(u_i, S_k) and
                d(u_i, S_k) = (u_i-mu_k)' (f_k C_k)^-1 (u_i-mu_k) is the
                Mahalanobis distance from u_i to the centroid mu_k; f_k is a
                factor that ensures it is a bounding ellipsoid; and C_k is the
                empirical covariance matrix of the subset S_k
            if no point is reassigned, go to step (B) below;
                else go back to (A)
            (B) if V(E_1) + V(E_2) < V(E) or V(E) > 2 V(S):
                parition S into S_1 and S_2 and repeat algorithm for
                    each subset
            else:
                return E as the optimal ellipsoid of set S
            endif

    To find the minimum bounding ellipsoid, we use the following procedure
    that returns the positive definite matrix C with centre mu that define the
    ellipsoid by :math:`(x - mu)^t C (x - mu) = 1`::

        cov = covariance(transpose(active_points))
        cov_inv = inv(cov)
        mu = mean(points)
        for i in n_active_points:
            dist[i] = (points[i] - mu) * cov_inv * (points[i] - mu)
        endfor
        enlargement_factor = max(dist)
        C = (1.0 / enlargement_factor) * cov_inv
        return C, mu

    From then on, in each iteration (t), the following occurs::

        V(E_k) = max(V(E_k),
            exp(-(t + 1) / n_active_points) * n_k / n_active_points)
        V(S_k) = (n_k / n_active_points) * exp(-(t + 1) / n_active_points)
        F(S) = (1 / V(S)) sum_{k=1}^{K} V(E_k)
        if F(S) > f_s_threshold:
            (E_1,..E_K), (S_1,...,S_K) = f_s_minimisation(t, u)
        endif
        L_min = min(L)
        indexmin = min_index(L)
        theta* = ellipsoids_sample((E_1,..E_K), (S_1,...,S_K), L_min)
        X_t = exp(-t / n_active_points)
        w_t = X_t - X_t-1
        Z = Z + L_min * w_t
        theta_indexmin = theta*
        L_indexmin = p(theta*|X)

    In the above, ``F(S)>=1`` is the ratio of the total volume overlapping
    ellipsoids to the volume of prior space remaining -- it is this
    functional that is minimised by ``f_s_minimisation``.

    To sample from the (potentially) overlapping ellipsoids, we use the
    following steps::

        ellipsoids_sample((E_1,..E_K), (S_1,...,S_K), L_min):
            choose ellipsoid k with probability:
                p_k = V(E_k) / sum_{k=1}^{K} V(E_k)
            theta* ~ ellipsoid_sample(E_k)
            while p(theta*|X) < L_min:
                theta* ~ ellipsoid_sample(E_k)
            endwhile
            n_e = count_ellipsoids(theta*)
            v ~ uniform(0, 1)
            if (1 / n_e) < v:
                theta* = ellipsoids_sample((E_1,..E_K), (S_1,...,S_K), L_min)
            endif
            return theta*

    The function ``ellipsoid_sample`` uniformly samples from within an
    ellipsoid. The function ``count_ellipsoids`` finds the number of
    ellipsoids a point is contained within.

    At the end of iterations, there is a final ``Z`` increment::

        Z = Z + (1 / n_active_points) * (L_1 + L_2 + ..., + L_n_active_points)

    The posterior samples are generated as described in [2]_ on page 849 by
    weighting each dropped sample in proportion to the volume of the
    posterior region it was sampled from. That is, the probability
    for drawing a given sample j is given by::

        p_j = L_j * w_j / Z

    where j = 1, ..., n_iterations.

    Extends :class:`NestedSampler`.

    References
    ----------
    .. [1] "MultiNest: an efficient and robust Bayesian inference tool for
            cosmology and particle physics."
            Feroz, F., M. P. Hobson, and M. Bridges.
            Monthly Notices of the Royal Astronomical Society 398.4 (2009):
            1601-1614.
    .. [2] "Nested Sampling for General Bayesian Computation", John Skilling,
           Bayesian Analysis 1:4 (2006).
           https://doi.org/10.1214/06-BA127
    """

    def __init__(self, log_prior):
        super(MultinestSampler, self).__init__(log_prior)

        # Enlargement factor for ellipsoid
        self.set_enlargement_factor()

        # Initial phase of rejection sampling
        # Number of nested rejection samples before starting ellipsoidal
        # sampling
        self.set_n_rejection_samples()
        self.set_initial_phase(True)

        self.set_f_s_threshold()

        self._needs_sensitivities = False

        self._alpha = 0.2
        self._A = None
        self._centroid = None
        self._A_l = []
        self._c_l = []
        self._V_S_l = []
        self._V_E_l = []
        self._F_S = 1.0
        self._f_s_minimisation_called = False

        # self._prior_cdf = log_prior.cdf()

    def ask(self, n_points):
        """
        If in initial phase, then uses rejection sampling. Afterwards,
        points are drawn from within an ellipsoid (needs to be in uniform
        sampling regime).
        """
        i = self._accept_count
        if (i + 1) % self._n_rejection_samples == 0:
            self._rejection_phase = False
            # determine bounding ellipsoids
            samples = self._m_active[:, :self._n_parameters]
            self._m_active_transformed = ([self._transform_to_unit_cube(x)
                                           for x in samples])
            (self._A_l, self._c_l, self._F_S, self._assignments, self._V_E_l,
             self._V_S_l) = (
                self._f_s_minimisation(i, self._m_active_transformed)
            )

        if self._rejection_phase:
            if n_points > 1:
                self._proposed = self._log_prior.sample(n_points)
            else:
                self._proposed = self._log_prior.sample(n_points)[0]
        else:
            self._A_l, self._F_S = self._update_ellipsoid_volumes(i)
            if self._F_S > self._f_s_threshold:
                samples = self._m_active[:, :self._n_parameters]
                self._m_active_transformed = ([self._transform_to_unit_cube(x)
                                               for x in samples])
                (self._A_l, self._c_l, self._F_S, self._assignments,
                 self._V_E_l, self._V_S_l) = (
                    self._f_s_minimisation(i, self._m_active_transformed)
                )
            u = self._sample_overlapping_ellipsoids(n_points, self._A_l,
                                                    self._c_l, self._V_E_l)
            if n_points > 1:
                self._proposed = [self._transform_from_unit_cube(x) for x in u]
            else:
                self._proposed = self._transform_from_unit_cube(u[0])

        return self._proposed

    def _comparison_enlargement(self, V_S, V_E, A):
        """
        Compares volume of prior space to that of ellispoid. If ``V_E`` exceeds
        ``V_S``, returns ellipsoid covariance matrix; otherwise, enlarges
        ellipsoid and returns new covariance matrix.
        """
        if V_E > V_S:
            return A
        else:
            enlargement_factor = V_S / V_E
            return self._enlarge_ellipsoid_A(enlargement_factor, A)

    def _count_ellipsoids(self, x, A_l, c_l):
        """
        Count number of ellipsoids point ``x`` is found within.
        """
        n_e = 0
        for i in range(len(A_l)):
            if self._mahalanobis_distance(x, c_l[i], A_l[i]) <= 1:
                n_e += 1
        return n_e

    def _draw_from_ellipsoid(self, covmat, cent, npts):
        """
        Draw ``npts`` random uniform points from within an ellipsoid with a
        covariance matrix covmat and a centroid cent, as per:
        http://www.astro.gla.ac.uk/~matthew/blog/?p=368
        """
        try:
            ndims = covmat.shape[0]
        except IndexError:  # pragma: no cover
            ndims = 1

        # calculate eigen_values (e) and eigen_vectors (v)
        eigen_values, eigen_vectors = np.linalg.eig(covmat)
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

    def ellipsoid_update_gap(self):
        """
        Returns the ellipsoid update gap used in the algorithm (see
        :meth:`set_ellipsoid_update_gap()`).
        """
        return self._ellipsoid_update_gap

    def _ellipsoid_sample(self, A, centroid, n_points):
        """
        Draws uniformly from the bounding ellipsoid.
        """
        if n_points > 1:
            return self._draw_from_ellipsoid(
                np.linalg.inv(A), centroid, n_points)
        else:
            return self._draw_from_ellipsoid(
                np.linalg.inv(A), centroid, 1)[0]

    def _ellipsoid_find_volume_calculator(self, a_index, u, assignments):
        """ Finds volume of a particular ellipsoid. """
        points = np.array(u)[np.where(assignments == a_index)]
        A, c = self._minimum_volume_ellipsoid(points)
        return A, c, self._ellipsoid_volume_calculator(A)

    def _ellipsoid_volume_calculator(self, A):
        """ Find volume of ellipsoid given its covariance matrix. """
        d = A.shape[0]
        r = np.sqrt(1 / np.linalg.eigvals(A))
        return (
            (np.pi**(d / 2.0) / scipy.special.gamma((d / 2.0) + 1.0))
            * np.prod(r))

    def _enlarge_ellipsoid_A(self, enlargement_factor, A):
        """ Enlarges an ellipsoid via its covariance matrix."""
        return (1 / enlargement_factor) * A

    def enlargement_factor(self):
        """
        Returns the enlargement factor used in the algorithm (see
        :meth:`set_enlargement_factor()`).
        """
        return self._enlargement_factor

    def _f_s_minimisation(self, iteration, u):
        """
        Runs ``F(S)`` minimisation and returns minimum bounding ellipsoid
        covariance matrices, then centroids and value of ``F(S)`` attained.
        """
        if not self._f_s_minimisation_called:
            self._f_s_minimisation_called = True
        assignments, A, N, V_E, V_S, c = (
            self._f_s_minimisation_steps_1_to_3(iteration, u))
        assignments_new, A_new_l, V_S_k_l, c_k_l, V_E_k_l = (
            self._f_s_minimisation_lines_4_to_13(assignments, u, V_S, 1))
        # lines 14 onwards
        A_l_running = []
        c_l_running = []
        V_E_k_tot = np.sum(V_E_k_l)
        if V_E_k_tot < V_E or V_E > 2 * V_S:
            for i in range(0, 2):
                u_new = u[np.where(assignments_new == i)]
                A_l_running, c_l_running = (
                    self._f_s_minimisation_lines_2_onwards(
                        u_new, V_E_k_l[i], V_S_k_l[i], A_new_l[i], c_k_l[i],
                        A_l_running, c_l_running))
            V_E_k_l1 = []
            for j in range(0, len(A_l_running)):
                V_E_k_l1.append(
                    self._ellipsoid_volume_calculator(A_l_running[j]))
            return (A_l_running, c_l_running, np.sum(V_E_k_l1) / V_S,
                    assignments_new, V_E_k_l1, V_S_k_l)
        else:
            return [A], [c], V_E / V_S, assignments_new, [V_E], [V_S]

    def _f_s_minimisation_steps_1_to_3(self, i, u):
        """ Performs steps 1-3 in Algorithm 1 in [1]_."""
        A, c, V_E = self._step_1(u)
        N = len(u)
        A, V_S = self._step_2(i, N, V_E, A)
        V_E = self._ellipsoid_volume_calculator(A)
        centers, assignments = self._step_3(u)
        return assignments, A, N, V_E, V_S, c

    def _f_s_minimisation_lines_2_onwards(self, u, V_E, V_S, A, c, A_l_running,
                                          c_l_running):
        A = self._comparison_enlargement(V_S, V_E, A)
        V_E = self._ellipsoid_volume_calculator(A)
        centers, assignments = self._step_3(u)
        assignments_new, A_new_l, V_S_k_l, c_k_l, V_E_k_l = (
            self._f_s_minimisation_lines_4_to_13(assignments, u, V_S, 1))
        # lines 14 onwards
        V_E_k_tot = np.sum(V_E_k_l)
        if V_E_k_tot < V_E or V_E > 2 * V_S:
            for i in range(0, 2):
                u_new = u[np.where(assignments_new == i)]
                # added this line to prevent too small clusters
                if len(u_new) < 50:
                    A_l_running.append(A)
                    c_l_running.append(c)
                    return A_l_running, c_l_running
                A_l_running, c_l_running = (
                    self._f_s_minimisation_lines_2_onwards(
                        u_new, V_E_k_l[i], V_S_k_l[i], A_new_l[i], c_k_l[i],
                        A_l_running, c_l_running))
            return A_l_running, c_l_running
        else:
            A_l_running.append(A)
            c_l_running.append(c)
            return A_l_running, c_l_running

    def _f_s_minimisation_lines_4_to_13(self, assignments, u, V_S,
                                        max_recursion):
        """ Performs steps 4-13 in Algorithm 1 in [1]_."""
        A_k_l, c_k_l, V_E_l = self._step_4(assignments, u)
        A_new_l, V_S_k_l, V_E_k_l = self._step_5(assignments, V_E_l, A_k_l,
                                                 V_S)
        assignments_new = self._step_6(u, c_k_l, A_k_l, V_E_k_l, V_S_k_l)
        assignments_new = assignments_new.astype(int)
        # stops algorithmic oscillation (not in original algorithm)
        if sum(assignments_new == 0) < 3 or sum(assignments_new == 1) < 3:
            return assignments, A_k_l, V_S_k_l, c_k_l, V_E_k_l
        if max_recursion > 10:
            return assignments_new, A_new_l, V_S_k_l, c_k_l, V_E_k_l
        if np.array_equal(assignments, assignments_new):
            return assignments_new, A_new_l, V_S_k_l, c_k_l, V_E_k_l
        else:
            return self._f_s_minimisation_lines_4_to_13(assignments_new, u,
                                                        V_S,
                                                        max_recursion + 1)

    def f_s_threshold(self):
        """ Returns threshold for ``F_S``."""
        return self._f_s_threshold

    def _h_k_calculator(self, point, mean_k, A_k, V_E_k, V_S_k):
        """ Calculates h_k as in eq. (23) in [1]_."""
        d = self._mahalanobis_distance(point, mean_k, A_k)
        return V_E_k * d / V_S_k

    def in_initial_phase(self):
        """ See :meth:`pints.NestedSampler.in_initial_phase()`. """
        return self._rejection_phase

    def _mahalanobis_distance(self, point, mean, A):
        """
        Finds Mahalanobis distance between a point and the centroid of
        of an ellipsoid.
        """
        return np.matmul(np.matmul(point - mean, A), point - mean)

    def _minimum_volume_ellipsoid(self, points, tol=0.0):
        """
        Finds an approximate minimum bounding ellipsoid in "center form":
        ``(x-c).T * A * (x-c) = 1``.
        """
        cov = np.cov(np.transpose(points))
        cov_inv = np.linalg.inv(cov)
        c = np.mean(points, axis=0)
        dist = np.zeros(len(points))
        for i in range(len(points)):
            dist[i] = np.matmul(np.matmul(points[i] - c, cov_inv),
                                points[i] - c)
        enlargement_factor = np.max(dist)
        A = (1 - tol) * (1.0 / enlargement_factor) * cov_inv
        return A, c

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 6

    def n_rejection_samples(self):
        """
        Returns the number of rejection sample used in the algorithm (see
        :meth:`set_n_rejection_samples()`).
        """
        return self._n_rejection_samples

    def name(self):
        """ See :meth:`pints.NestedSampler.name()`. """
        return 'Nested ellipsoidal sampler'

    def needs_initial_phase(self):
        """ See :meth:`pints.NestedSampler.needs_initial_phase()`. """
        return True

    def _sample_overlapping_ellipsoid(self, k, A_l, c_l):
        """ Uniformly samples from a given ellipsoid accounting for overlap."""
        test_point = self._ellipsoid_sample(A_l[k], c_l[k], 1)
        n_e = self._count_ellipsoids(test_point, A_l, c_l)
        if n_e < 1:
            raise RuntimeError("Point not in any ellipse.")
        if n_e > 1:
            p_accept = 1.0 / n_e
            if p_accept > np.random.uniform():
                return test_point
            else:
                return self._sample_overlapping_ellipsoid(k, A_l)
        return test_point

    def _sample_overlapping_ellipsoids(self, n_points, A_l, c_l, V_E_l):
        """
        Uniformly sample from bounding ellipsoids accounting for overlap.
        """
        # calculate probabilities as per eq. (24)
        p = []
        V_tot = sum(V_E_l)
        for V_E in V_E_l:
            p.append(V_E / V_tot)
        points = []
        for i in range(n_points):
            k = random.choices(list(range(len(p))), weights=p)[0]
            points.append(self._sample_overlapping_ellipsoid(k, A_l, c_l))
        return points

    def set_ellipsoid_update_gap(self, ellipsoid_update_gap=100):
        """
        Sets the frequency with which the minimum volume ellipsoid is
        re-estimated as part of the nested rejection sampling algorithm.

        A higher rate of this parameter means each sample will be more
        efficiently produced, yet the cost of re-computing the ellipsoid
        may mean it is better to update this not each iteration -- instead,
        with gaps of ``ellipsoid_update_gap`` between each update. By default,
        the ellipsoid is updated every 100 iterations.
        """
        ellipsoid_update_gap = int(ellipsoid_update_gap)
        if ellipsoid_update_gap <= 1:
            raise ValueError('Ellipsoid update gap must exceed 1.')
        self._ellipsoid_update_gap = ellipsoid_update_gap

    def set_enlargement_factor(self, enlargement_factor=1.1):
        """
        Sets the factor (>1) by which to increase the minimal volume
        ellipsoidal in rejection sampling.

        A higher value means it is less likely that areas of high probability
        mass will be missed. A low value means that rejection sampling is more
        efficient.
        """
        if enlargement_factor <= 1:
            raise ValueError('Enlargement factor must exceed 1.')
        self._enlargement_factor = enlargement_factor

    def set_f_s_threshold(self, h=1.1):
        """
        Sets threshold for ``F_S`` when minimum bounding ellipsoids are refit.
        """
        if h <= 1:
            raise ValueError('F_S threshold factor must exceed 1.')
        self._f_s_threshold = h

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[# active points, # rejection samples,
        enlargement factor, ellipsoid update gap, dynamic enlargement factor,
        alpha]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_n_active_points(x[0])
        self.set_n_rejection_samples(x[1])
        self.set_enlargement_factor(x[2])

    def set_initial_phase(self, in_initial_phase):
        """ See :meth:`pints.NestedSampler.set_initial_phase()`. """
        self._rejection_phase = bool(in_initial_phase)

    def set_n_rejection_samples(self, rejection_samples=200):
        """
        Sets the number of rejection samples to take, which will be assigned
        weights and ultimately produce a set of posterior samples.
        """
        if rejection_samples < 0:
            raise ValueError('Must have non-negative rejection samples.')
        self._n_rejection_samples = rejection_samples

    def _step_1(self, u):
        """ Performs step 1 in Algorithm 1 in [1]_."""
        A, c = self._minimum_volume_ellipsoid(u)
        V_E = self._ellipsoid_volume_calculator(A)
        return A, c, V_E

    def _step_2(self, i, N, V_E, A):
        """ Performs step 2 in Algorithm 1 in [1]_."""
        V_S = self._V_S_calculator(i, N)
        return self._comparison_enlargement(V_S, V_E, A), V_S

    def _step_3(self, u):
        """ Performs step 3 in Algorithm 1 in [1]_."""
        centers, assignment = scipy.cluster.vq.kmeans2(u, 2, minit="points")
        while sum(assignment == 0) < 3 or sum(assignment == 1) < 3:
            centers, assignment = (
                scipy.cluster.vq.kmeans2(u, 2, minit="points"))
        return centers, assignment

    def _step_4(self, assignments, u):
        """ Performs step 4 in Algorithm 1 in [1]_."""
        A_l = [None] * 2
        c_l = [None] * 2
        V_E_l = [None] * 2
        for i in range(0, 2):
            A, c, V_E = self._ellipsoid_find_volume_calculator(i, u,
                                                               assignments)
            A_l[i] = A
            c_l[i] = c
            V_E_l[i] = V_E
        return A_l, c_l, V_E_l

    def _step_5(self, assignments, V_E_l, A_l, V_S):
        """ Performs step 5 in Algorithm 1 in [1]_."""
        A_new_l = [None] * 2
        V_S_k_l = [None] * 2
        V_E_k_l = [None] * 2
        N = len(assignments)
        for i in range(0, 2):
            n = np.sum(assignments == i)
            V_S_k_l[i] = self._V_S_k_calculator(n, N, V_S)
            A_new_l[i] = (
                self._comparison_enlargement(V_S_k_l[i], V_E_l[i], A_l[i]))
            V_E_k_l[i] = self._ellipsoid_volume_calculator(A_new_l[i])
        return A_new_l, V_S_k_l, V_E_k_l

    def _step_6(self, points, c_k_l, A_k_l, V_E_l, V_S_k_l):
        """ Performs step 6 in Algorithm 1 in [1]_."""
        n = len(points)
        assignments_new = np.zeros(n)
        for i in range(0, n):
            h_k_max = float('inf')
            for j in range(0, 2):
                h_k = self._h_k_calculator(points[i], c_k_l[j],
                                           A_k_l[j], V_E_l[j], V_S_k_l[j])
                if h_k < h_k_max:
                    assignments_new[i] = j
                    h_k_max = h_k
        return assignments_new

    def _transform_to_unit_cube(self, theta):
        """
        Transforms a given parameter sample to unit cube, using the prior
        cumulative distribution function.
        """
        return theta

    def _transform_from_unit_cube(self, theta):
        """
        Transforms a sample in unit cube, to parameter space using the prior
        inverse cumulative distribution function.
        """
        return theta

    def _update_ellipsoid_volumes(self, t):
        """ Updates ellipsoids as defined in text on p.1605 of [1]_. """
        if not self._f_s_minimisation_called:
            raise RuntimeError(
                '_update_ellipsoid_volumes() called before volumes have ' +
                'been calculated')
        A_l = []
        V_S = np.exp(-t / self._n_active_points)
        for i, A in enumerate(self._A_l):
            # not 100% sure about this next line as not explicitly in text
            self._V_S_l[i] = (
                np.sum(self._assignments == i) * V_S / self._n_active_points)
            enlargement_factor = self._V_S_l[i] / self._V_E_l[i]
            if enlargement_factor > 1:
                self._V_E_l[i] = self._V_S_l[i]
                A_l.append(self._enlarge_ellipsoid_A(enlargement_factor, A))
            else:
                A_l.append(A)
        F_S = sum(self._V_E_l) / V_S
        return A_l, F_S

    def _V_S_calculator(self, i, N):
        """ Calculates prior volume remaining."""
        return np.exp(-float(i) / float(N))

    def _V_S_k_calculator(self, n_k, N, V_S):
        """ Calculates prior volume remaining for set k."""
        return n_k * V_S / N
