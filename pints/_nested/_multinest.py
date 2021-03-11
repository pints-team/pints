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
from pints._nested.__init__ import Ellipsoid


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

    Run rejection sampling for ``n_rejection_samples`` to generate an initial
    sample of active points, along with updated values of ``L_min`` and
    ``indexmin``.

    Transform all active points into the unit cube via the cumulative
    distribution function of the priors:

    .. math::
        u_i = \int_{-\infty}^{\theta_i} \pi(\theta') d\theta'

    Fit transformed active points using minimum volume bounding ellipsoids
    (that potentially overlap) as described by Algorithm 1 in [1]_. Explicitly
    this algorithm seeks to minimise a quantity ``F(S)>=1`` representing the
    ratio of the total volume overlapping ellipsoids to the volume of prior
    space remaining.

    We accomplish ``F_S`` minimisation by constructing a binary tree with
    ellipsoids (E above) as its leaves. The pseudocode for this is::

        EllipsoidTree(t, u, ef):
            calculate bounding ellipsoid E and its volume V(E)
            V(S) = exp(-t/n_active_points) * ef; t is iteration, and
                S is prior volume remaining, ef is the enlargement factor
            enlarge E so that V(E) = max(V(E), V(S))
            if n_active_points > min_points:
                using k-means algorithm partition active points into subsets
                    u_1 and u_1
                num_tries = 0
                (A) find E_1 and E_2 (bounding ellipsoids of each point set)
                    and their volumes V(E_1) and V(E_2)
                enlarge E_k (k=1,2) so that V(E_k) = max(V(E_k), V(S_k)),
                    where V(S_k) = n_k V(S) / n_active_points
                for j in 1:n_active_points
                    assign u^{j} to S_k such that
                    h_k(u^{j}) = min(h_1(u^{j}), h_2(u^{j}))
                endfor
                if num_tries < max_recursion and either if insufficient points
                    reassigned or clusters are too small go back to (A)
                if V(E_1) + V(E_2) < V(E) or V(E) > 2 V(S):
                    left_branch = EllipsoidTree(t, u_1, ef)
                    right_branch = EllipsoidTree(t, u_2, ef)

            left_branch = Null
            right_branch = Null
            leaf = E

    In the above, h_k(u_i) = (V(E_k) / V(S_k)) * d(u_i, S_k) and
    d(u_i, S_k) = (u_i-mu_k)' (f_k C_k)^-1 (u_i-mu_k) is the Mahalanobis
    distance from u_i to the centroid mu_k; f_k is a factor that ensures it is
    a bounding ellipsoid; and C_k is the empirical covariance matrix of the
    subset S_k.

    From then on, in each iteration (t), the following occurs::

        V(E_k) = max(V(E_k),
            exp(-(t + 1) / n_active_points) * n_k / n_active_points)
        V(S_k) = (n_k / n_active_points) * exp(-(t + 1) / n_active_points)
        F(S) = (1 / V(S)) sum_{k=1}^{K} V(E_k)
        if F(S) > f_s_threshold:
            ellipsoid_tree = EllipsoidTree(t, u, ef)
        endif
        L_min = min(L)
        indexmin = min_index(L)
        theta* = ellipsoids_tree.sample()
        X_t = exp(-t / n_active_points)
        w_t = X_t - X_t-1
        Z = Z + L_min * w_t
        theta_indexmin = theta*
        L_indexmin = p(theta*|X)

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

        # Minimal gaps between updating ellipsoid
        self.set_ellipsoid_update_gap()

        # Initial phase of rejection sampling
        # Number of nested rejection samples before starting ellipsoidal
        # sampling
        self.set_n_rejection_samples()
        self.set_initial_phase(True)

        self.set_f_s_threshold()

        self._multiple_ellipsoids = True
        self._needs_sensitivities = False

        self._f_s_minimisation_called = False

        self._convert_to_unit_cube = log_prior.convert_to_unit_cube
        self._convert_from_unit_cube = log_prior.convert_from_unit_cube

    def ask(self, n_points):
        """
        If in initial phase, then uses rejection sampling. Afterwards,
        points are drawn from within an ellipsoid (needs to be in uniform
        sampling regime).
        """
        i = self._accept_count
        if self._rejection_phase:
            if (i + 1) > self._n_rejection_samples:
                self._rejection_phase = False
                # determine bounding ellipsoids
                samples = self._m_active[:, :self._n_parameters]
                m_active_transformed = ([self._convert_to_unit_cube(x)
                                         for x in samples])
                self._ellipsoid_tree = EllipsoidTree(
                    m_active_transformed, i, self._enlargement_factor)
                self._ellipsoid_count = (
                    self._ellipsoid_tree.n_leaf_ellipsoids())
            else:
                if n_points > 1:
                    self._proposed = self._log_prior.sample(n_points)
                else:
                    self._proposed = self._log_prior.sample(n_points)[0]
                self._ellipsoid_count = 0
        else:
            self._ellipsoid_tree.update_leaf_ellipsoids(i)
            if ((i + 1 - self._n_rejection_samples)
                    % self._ellipsoid_update_gap == 0):
                if self._ellipsoid_tree.f_s() > self._f_s_threshold:
                    print("EllipsoidTree updated!")
                    samples = self._m_active[:, :self._n_parameters]
                    m_active_transformed = ([self._convert_to_unit_cube(x)
                                             for x in samples])
                    self._ellipsoid_tree = EllipsoidTree(
                        m_active_transformed, i, self._enlargement_factor)
            u = self._ellipsoid_tree.sample_leaf_ellipsoids(n_points)
            if n_points > 1:
                self._proposed = [self._convert_from_unit_cube(x) for x in u]
            else:
                self._proposed = self._convert_from_unit_cube(u[0])
            self._ellipsoid_count = self._ellipsoid_tree.n_leaf_ellipsoids()
        return self._proposed

    def ellipsoid_tree(self):
        """ Returns final ellipsoid tree based on last iteration. """
        return self._ellipsoid_tree

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

    def f_s_threshold(self):
        """ Returns threshold for ``F_S``."""
        return self._f_s_threshold

    def in_initial_phase(self):
        """ See :meth:`pints.NestedSampler.in_initial_phase()`. """
        return self._rejection_phase

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 3

    def n_rejection_samples(self):
        """
        Returns the number of rejection sample used in the algorithm (see
        :meth:`set_n_rejection_samples()`).
        """
        return self._n_rejection_samples

    def name(self):
        """ See :meth:`pints.NestedSampler.name()`. """
        return 'MultiNest sampler'

    def needs_initial_phase(self):
        """ See :meth:`pints.NestedSampler.needs_initial_phase()`. """
        return True

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
        enlargement factor, ellipsoid update gap]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_n_active_points(x[0])
        self.set_n_rejection_samples(x[1])
        self.set_enlargement_factor(x[2])
        self.set_ellipsoid_update_gap(x[3])

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


class EllipsoidTree():
    """
    Binary tree with ellipsoids as leaf nodes which is used to minimise
    F_s as in Algorithm 1 in [1]_.
    """
    def __init__(self, points, iteration, enlargement_factor=1):
        n_points = len(points)
        if n_points < 1:
            raise ValueError(
                "More than one point is needed in a EllipsoidTree.")
        for point in points:
            if min(point) < 0 or max(point) > 1:
                raise ValueError(
                    "Points must be in unit cube.")
        self._n_points = n_points
        self._dimensions = len(points[0])
        self._max_recursion = 50
        self._min_points_to_split = 50
        self._points = points
        if iteration < 1:
            raise ValueError(
                "iteration must be >=1."
            )
        self._iteration = iteration
        self._left = None
        self._right = None

        # step 1 in Algorithm 1
        # calculate volume of space
        self._V_S = self.vs() * enlargement_factor
        # calculate bounding ellipsoid
        self._ellipsoid = Ellipsoid.minimum_volume_ellipsoid(points)

        V_E = self._ellipsoid.volume()

        # step 2 in Algorithm 1
        self.compare_enlarge(self._ellipsoid, self._V_S)

        # not in algorithm but safeguard against small ellipsoids
        if n_points > self._min_points_to_split:
            # step 3 in Algorithm 1
            _, assignments = scipy.cluster.vq.kmeans2(
                points, 2, minit="points")
            # ensures against small clusters
            while (sum(assignments == 0) < (self._dimensions + 3) or
                   sum(assignments == 1) < (self._dimensions + 3)):
                centers, assignment = (
                    scipy.cluster.vq.kmeans2(points, 2, minit="points"))

            # steps 4-13 in Algorithm 1
            ellipsoid_1, ellipsoid_2 = self.split_ellipsoids(points,
                                                             assignments,
                                                             0)
            # steps 14+ in Algorithm 1
            V_E_1 = ellipsoid_1.volume()
            V_E_2 = ellipsoid_2.volume()

            if (V_E_1 + V_E_2 < V_E) or (V_E > 2 * self._V_S):
                self._left = EllipsoidTree(ellipsoid_1.points(), iteration)
                self._right = EllipsoidTree(ellipsoid_2.points(), iteration)

    def compare_enlarge(self, ellipsoid, V_S):
        """
        Compares the volume of an ellipsoid to V_S and, if it is smaller,
        enlarges it so that it has the same volume.
        """
        r = V_S / ellipsoid.volume()
        if r > 1:
            ellipsoid.enlarge(r)

    def count_within_leaf_ellipsoids(self, point):
        """
        Determines number of ellipsoids a point is contained within.
        """
        leaves = self.leaf_ellipsoids()
        count = 0
        for leaf in leaves:
            if leaf.within_ellipsoid(point):
                count += 1
        return count

    def ellipsoid(self):
        """ Returns bounding ellipsoid of tree. """
        return self._ellipsoid

    def f_s(self):
        """
        Returns F_s representing ratio of ellipsoid volume to total volume.
        """
        return self.leaf_ellipsoids_volume() / self._V_S

    def h_k(self, point, ellipsoid, V_S_k):
        """ Calculates h_k as in eq. (23) in [1]_."""
        d = Ellipsoid.mahalanobis_distance(point,
                                           ellipsoid.weight_matrix(),
                                           ellipsoid.centroid())
        return ellipsoid.volume() * d / V_S_k

    def leaf_ellipsoids(self):
        """ Returns leaf ellipsoids of tree. """
        if self._left is None and self._right is None:
            return [self.ellipsoid()]
        else:
            return (self._left.leaf_ellipsoids() +
                    self._right.leaf_ellipsoids())

    def leaf_ellipsoids_volume(self):
        """ Returns volume of leaf ellipsoids. """
        if self._left is None and self._right is None:
            return self.ellipsoid().volume()
        else:
            return (self._left.leaf_ellipsoids_volume() +
                    self._right.leaf_ellipsoids_volume())

    def n_leaf_ellipsoids(self):
        """ Counts the leaf ellipsoids. """
        if self._left is None and self._right is None:
            return 1
        else:
            return (self._left.n_leaf_ellipsoids() +
                    self._right.n_leaf_ellipsoids())

    def sample_leaf_ellipsoids(self, ndraws):
        """
        Draws uniform samples from within leaf ellipsoids accounting for their
        overlap.
        """

        # calculate relative volumes of ellipsoids
        leaves = self.leaf_ellipsoids()
        volumes = [ell.volume() for ell in leaves]
        volume_tot = sum(volumes)
        volumes_rel = [vol / volume_tot for vol in volumes]

        # propose ellipsoid in proportion to its volume
        draws = []
        naccepted = 0
        while naccepted < ndraws:
            k = np.random.choice(len(volumes), p=volumes_rel)
            ellipsoid = leaves[k]
            test_point = ellipsoid.sample(1)
            n_e = self.count_within_leaf_ellipsoids(test_point)
            if n_e == 1:
                naccepted += 1
                draws.append(test_point)
            elif n_e > 1:
                paccept = 1.0 / n_e
                if paccept > np.random.uniform():
                    naccepted += 1
                    draws.append(test_point)
            elif n_e < 1:
                raise RuntimeError("Point not in any ellipse.")
        return draws

    def split_ellipsoids(self, points, assignments, recursion_count):
        """
        Performs steps 4-13 in Algorithm 1 in [1]_, where the points are
        partitioned into two ellipsoids to minimise a measure `h_k`.
        """
        # step 4 in Algorithm 1
        ellipsoids = []
        for i in range(2):
            points_temp = np.array(points)[np.where(assignments == i)]
            ellipsoids.append(
                Ellipsoid.minimum_volume_ellipsoid(points_temp))

        # step 5 in Algorithm 1
        V_S_ks = [self.vsk(el) for el in ellipsoids]
        for i in range(2):
            self.compare_enlarge(ellipsoids[i], V_S_ks[i])

        # step 6 in Algorithm 1
        n = self._n_points
        assignments_new = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            h_k_max = float('inf')
            for j in range(2):
                h_k = self.h_k(points[i], ellipsoids[j], V_S_ks[j])
                if h_k < h_k_max:
                    assignments_new[i] = j
                    h_k_max = h_k

        # first two conditions stop algorithmic oscillation and are not in
        # original algorithm)
        if (
            (sum(assignments_new == 0) < 3 or sum(assignments_new == 1) < 3)
            or recursion_count > self._max_recursion
            or np.array_equal(assignments, assignments_new)): # noqa
            return ellipsoids
        else:
            return self.split_ellipsoids(points,
                                         assignments_new,
                                         recursion_count + 1)

    def update_leaf_ellipsoids(self, iteration):
        """
        Updates ellipsoids according to p.1605 (bottom-right) in [1]_ according
        to iteration.
        """
        leaves = self.leaf_ellipsoids()
        [self.compare_enlarge(ell, self.vsk(ell)) for ell in leaves]

    def vs(self):
        """ Calculates volume of total space. """
        return np.exp(-self._iteration / self._n_points)

    def vsk(self, ellipsoid):
        """ Calculates subvolume of ellipsoid. """
        n_points = ellipsoid.n_points()
        if n_points > self._n_points:
            raise ValueError(
                "Number of points in ellipsoid be not exceed that in tree.")
        return ellipsoid.n_points() * self._V_S / self._n_points
