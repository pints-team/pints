#
# DRAM AC MC method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy.stats as stats


class DramACMCMC(pints.GlobalAdaptiveCovarianceMC):
    """
    DRAM (Delayed Rejection Adaptive Covariance) MCMC, as described in [1]_.
    In this method, rejections do not necessarily lead an iteration to end.
    Instead, if a rejection occurs, another point is proposed although from
    a narrower (i.e. more conservative) proposal kernel than was used for the
    first proposal.

    In this approach, in each iteration, the following steps return the next
    state of the Markov chain (assuming the current state is theta_0 and that
    there are 2 proposal kernels)::

        theta_1 ~ N(theta_0, scale_1 * sigma0)
        alpha_1(theta_0, theta_1) = min(1, p(theta_1|X) / p(theta_0|X))
        u_1 ~ uniform(0, 1)
        if alpha_1(theta_0, theta_1) > u_1:
            return theta_1
        theta_2 ~ N(theta_0, scale_2 * sigma0)
        alpha_2(theta_0, theta_1, theta_2) =
            min(1, p(theta_2|X) (1 - alpha_1(theta_2, theta_1)) /
                   (p(theta_0|X) (1 - alpha_1(theta_0, theta_1))))
        u_2 ~ uniform(0, 1)
        if alpha_2(theta_0, theta_1, theta_2) > u_2:
            return theta_2
        else:
            return theta_0

    Our implementation also allows more than 2 proposal kernels to be used.
    This means that ``k`` accept-reject steps are taken. In each step (``i``),
    the probability that a proposal ``theta_i`` is accepted is::

        alpha_i(theta_0, theta_1, ..., theta_i) = min(1,
         p(theta_i|X) / p(theta_0|X) * n_i / d_i)

    where::

        n_i = (1 - alpha_1(theta_i, theta_i-1))
              (1 - alpha_2(theta_i, theta_i-1, theta_i-2))
               ...
               ((1 - alpha_i-1(theta_i, theta_i-1, ..., theta_0))
        d_i = (1 - alpha_1(theta_0, theta_1))
              (1 - alpha_2(theta_0, theta_1, theta_2))
              ...
              (1 - alpha_i-1(theta_0, theta_1, ..., theta_i-1))

    If ``k`` proposals have been rejected, the initial point ``theta_0`` is
    returned.

    *Extends:* :class:`GlobalAdaptiveCovarianceMC`

    References
    ----------
    .. [1] "DRAM: Efficient adaptive MCMC".
           H Haario, M Laine, A Mira, E Saksman, (2006) Statistical Computing
           https://doi.org/10.1007/s11222-006-9438-0
    """
    def __init__(self, x0, sigma0=None):
        super(DramACMCMC, self).__init__(x0, sigma0)

        self._kernels = 2
        self._Y = [None] * self._kernels
        self._Y_log_pdf = np.zeros(self._kernels)
        self._proposal_count = 0
        self._adapt_kernel = np.repeat(True, self._kernels)
        self._gamma = np.repeat(1.0, self._kernels)
        self._eta = np.repeat(np.copy(self._eta), self._kernels)

        # Set kernels
        v_mu = np.copy(self._mu)
        self._mu = [v_mu for i in range(self._kernels)]
        a_min = np.log10(1)
        a_max = np.log10(1000)
        self._sigma_scale = 10**np.linspace(a_min, a_max, self._kernels)
        m_sigma = np.copy(self._sigma)
        self._sigma = [
            self._sigma_scale[i] * m_sigma for i in range(self._kernels)]

    def ask(self):
        """
        If first proposal from a position, return
        a proposal with an ambitious (i.e. large)
        proposal width; if first is rejected
        then return a proposal from a conservative
        kernel (i.e. with low width).
        """
        super(DramACMCMC, self).ask()

        # Propose new point
        if self._proposed is None:
            self._proposed = np.random.multivariate_normal(
                self._current, np.exp(self._loga) *
                self._sigma[self._proposal_count])
            self._Y[self._proposal_count] = np.copy(self._proposed)
            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def set_sigma_scale(self, minK, maxK):
        """
        Set the scale of initial covariance matrix
        multipliers for each of the kernels:
        (minK,...,maxK) where the gradations are
        uniform on the log10 scale.
        This means that the covariance matrices are:
        maxK * sigma0,..., MinK * sigma0
        where n can be modified by ``set_kernels``.
        """
        if minK > maxK:
            raise ValueError('Maximum kernel multiplier must ' +
                             'exceed minimum.')
        a_min = np.log10(minK)
        a_max = np.log10(maxK)
        self._sigma_scale = 10**np.linspace(a_min, a_max, self._kernels)
        self._sigma = [self._sigma_scale[i] * self._sigma
                       for i in range(self._kernels)]

    def tell(self, fx):
        """
        If first proposal, then accept with ordinary
        Metropolis probability; if a later proposal, use probability
        determined by [1]_.
        """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure fx is a float
        fx = float(fx)
        self._Y_log_pdf[self._proposal_count] = fx

        # First point?
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Accept
            self._current = self._proposed
            self._current_log_pdf = fx

            # Increase iteration count
            self._iterations += 1

            # Clear proposal
            self._proposed = None

            # Return first point for chain
            return self._current

        # Check if the proposed point can be accepted
        accepted = 0
        self._calculate_r_log(fx)

        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < self._r_log:
                accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Clear proposal
        self._proposed = None
        # Adapt covariance matrix
        a_count = np.copy(self._proposal_count)
        if self._adaptive:
            # Set gamma based on number of adaptive iterations
            self._gamma[a_count] = (self._adaptations**-self._eta[a_count])
            self._adaptations += 1

            # Update mu, log acceptance rate, and covariance matrix
            self._update_mu()
            self._update_sigma()

        # Return new point for chain
        if accepted == 0:
            # rejected first proposal
            if self._kernels > 1 and self._proposal_count == 0:
                self._proposal_count += 1
                return None
            else:
                self._proposal_count = 0
        # if accepted or failed on second try
        return self._current

    def _update_mu(self):
        """
        Updates the means of the various kernels being used according to
        adaptive Metropolis routine.
        """
        a_count = np.copy(self._proposal_count)
        if self._adapt_kernel[a_count]:
            self._mu[a_count] = ((1 - self._gamma[a_count]) *
                                 self._mu[a_count] +
                                 self._gamma[a_count] *
                                 self._current)

    def _update_sigma(self):
        """
        Updates the covariance matrices of the various kernels being used
        according to adaptive Metropolis routine.
        """
        a_count = np.copy(self._proposal_count)
        if self._adapt_kernel[a_count]:
            dsigm = np.reshape(self._current - self._mu[a_count],
                               (self._dimension, 1))
            self._sigma[a_count] = ((1 - self._gamma[a_count]) *
                                    self._sigma[a_count] +
                                    self._gamma[a_count] *
                                    np.dot(dsigm, dsigm.T))

    def _calculate_r_log(self, fx):
        """
        Calculates value of logged acceptance ratio (eq. 3 in [1]_).
        """
        c = self._proposal_count
        temp_Y = np.concatenate([[self._current], self._Y[0:(c + 1)]])
        temp_log_Y = np.concatenate([[self._current_log_pdf],
                                     self._Y_log_pdf[0:(c + 1)]])
        alpha_log = temp_log_Y[c + 1] - temp_log_Y[0]
        if c == 0:
            self._r_log = min(0, alpha_log)
        Y_rev = temp_Y[::-1]
        log_Y_rev = temp_log_Y[::-1]
        for i in range(c):
            alpha_log += (
                stats.multivariate_normal.logpdf(
                    x=temp_Y[c - i - 1],
                    mean=temp_Y[c + 1],
                    cov=self._sigma[c],
                    allow_singular=True) -
                stats.multivariate_normal.logpdf(
                    x=temp_Y[i],
                    mean=self._current,
                    cov=self._sigma[0],
                    allow_singular=True) +
                np.log(1 - np.exp(self._calculate_alpha_log(
                    i, Y_rev[0:(i + 2)], log_Y_rev[0:(i + 2)]))) -
                np.log(1 - np.exp(self._calculate_alpha_log(
                    i, temp_Y[0:(i + 2)], temp_log_Y[0:(i + 2)])))
            )
        self._r_log = min(0, alpha_log)
