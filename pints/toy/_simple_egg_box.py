#
# Simple egg-box LogPDF
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy.stats


class SimpleEggBoxLogPDF(pints.LogPDF):
    """
    Two-dimensional multimodal Normal distribution, with four more-or-less
    independent modes, each centered in a different quadrant.

    Arguments:

    ``sigma``
        The variance of each mode.
    ``r``
        The first mode will be located at ``(d, d)``, ``(-d, d)``, (-d, -d)``,
        and ``(d, -d)``, where ``d = r * sigma``.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, sigma=2, r=4):

        # Sigma for every mode
        self._sigma = float(sigma)
        if self._sigma <= 0:
            raise ValueError('Sigma must be greater than zero.')

        # Set modes
        r = float(r)
        if r <= 0:
            raise ValueError('Argument r must be greater than zero.')
        d = r * self._sigma
        self._modes = [
            [d, d],
            [-d, d],
            [-d, -d],
            [d, -d],
        ]

        # Set covariances
        self._covs = [np.eye(2) * sigma] * 4

        # Create scipy 'random variables'
        self._vars = [
            scipy.stats.multivariate_normal(mode, self._covs[i])
            for i, mode in enumerate(self._modes)]

    def __call__(self, x):
        f = np.sum([var.pdf(x) for var in self._vars])
        return -float('inf') if f == 0 else np.log(f)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return 2

    def kl_divergence(self, samples):
        """
        Calculates a heuristic score for how well a given set of samples
        matches this LogPDF's underlying distribution, based on
        Kullback-Leibler divergence of the individual modes. This only works
        well if the modes are nicely separated, i.e. for larger values of
        ``r``.
        """
        dimension = 2

        # Check size of input
        if not len(samples.shape) == 2:
            raise ValueError('Given samples list must be 2x2.')
        if samples.shape[1] != dimension:
            raise ValueError(
                'Given samples must have length ' + str(dimension))

        # Separate samples into quadrants
        q12 = samples[samples[:, 1] >= 0]
        q34 = samples[samples[:, 1] < 0]
        q1 = q12[q12[:, 0] >= 0]
        q2 = q12[q12[:, 0] < 0]
        q3 = q34[q34[:, 0] < 0]
        q4 = q34[q34[:, 0] >= 0]
        qs = [q1, q2, q3, q4]

        # Calculate kullback-leibler for each quadrant-mode pair
        dkls = np.array([0, 0, 0, 0], dtype=float)
        for i, q in enumerate(qs):
            if len(q) == 0:
                continue
            m0 = np.mean(q, axis=0)
            s0 = np.cov(q.T)
            m1 = self._modes[i]
            s1 = self._covs[i]
            cov_inv = np.linalg.inv(s1)
            dkl1 = np.trace(cov_inv.dot(s0))
            dkl2 = np.dot((m1 - m0).T, cov_inv).dot(m1 - m0)
            dkl3 = np.log(np.linalg.det(s1) / np.linalg.det(s0))
            dkls[i] = 0.5 * (dkl1 + dkl2 + dkl3 - dimension)

        # No samples in a given quadrant? Then use 100 times max divergence
        penalty1 = 100 * np.max(dkls)
        dkls[dkls == 0] = penalty1

        # Sum divergences together
        score = np.sum(dkls)

        # Penalise unequal distribution of the points, and return
        ns = [len(q) for q in qs]
        penalty2 = np.max(ns) / max(1, np.min(ns))
        return score * penalty2

    def sample(self, n):
        """
        Returns ``n`` samples from the underlying distribution.
        """
        if n < 0:
            raise ValueError('Number of samples cannot be negative.')

        # Calculate number of samples from each distribution
        weights = [0.25] * 4
        ns = np.sum(scipy.stats.multinomial.rvs(1, weights, n), axis=0)

        # Draw samples from each distribution, then join them together
        x = [v.rvs(ns[i]) for i, v in enumerate(self._vars)]
        x = np.vstack(x)

        # Shuffle the samples and return
        np.random.shuffle(x)
        return x

    def distance(self, samples):
        """
        Calculates approximate mode-wise KL divergence (see `kl_divergence`)
        """
        return kl_divergence(samples)
