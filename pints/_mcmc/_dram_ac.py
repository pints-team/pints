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


class DramACMC(pints.GlobalAdaptiveCovarianceMC):
    """
    DRAM (Delayed Rejection Adaptive Covariance) MCMC, as described in [1]_.
    In this method, rejections do not necessarily lead an iteration to end.
    Instead, if a rejection occurs, another point is proposed although
    typically from a narrower (i.e. more conservative) proposal kernel than was
    used for the first proposal.

    In this approach, in each iteration, the following steps return the next
    state of the Markov chain (assuming the current state is ``theta_0`` and
    that there are 2 proposal kernels)::

        theta_1 ~ N(theta_0, lambda * scale_1 * sigma)
        alpha_1(theta_0, theta_1) = min(1, p(theta_1|X) / p(theta_0|X))
        u_1 ~ uniform(0, 1)
        if alpha_1(theta_0, theta_1) > u_1:
            return theta_1
        theta_2 ~ N(theta_0, lambda * scale_2 * sigma0)
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

        alpha_i(theta_0, theta_1, ..., theta_i) = min(1, p(theta_i|X) /
                                                  p(theta_0|X) * n_i / d_i)

    where::

        n_i = (1 - alpha_1(theta_i, theta_i-1)) *
              (1 - alpha_2(theta_i, theta_i-1, theta_i-2)) *
               ...
               ((1 - alpha_i-1(theta_i, theta_i-1, ..., theta_0))
        d_i = (1 - alpha_1(theta_0, theta_1)) *
              (1 - alpha_2(theta_0, theta_1, theta_2)) *
              ...
              (1 - alpha_i-1(theta_0, theta_1, ..., theta_i-1))

    If ``k`` proposals have been rejected, the initial point ``theta_0`` is
    returned.

    If ``adaptative=1``, at the end of each iterations, a 'base' proposal
    kernel is adapted::

        mu = (1 - gamma) mu + gamma theta
        sigma = (1 - gamma) sigma + gamma (theta - mu)(theta - mu)^t
        log_lambda = log_lambda + gamma (accepted - target_acceptance_rate)

    where ``gamma = adaptations^-eta``, ``theta`` is the current state of
    the Markov chain and ``accepted`` is a binary indicator for whether any of
    the series of proposals were accepted. The kernels for the all proposals
    are then adapted as ``[scale_1, scale_2, ..., scale_k] * sigma``, where the
    scale factors are set using ``set_sigma_scale``.

    *Extends:* :class:`GlobalAdaptiveCovarianceMC`

    References
    ----------
    .. [1] "DRAM: Efficient adaptive MCMC".
           H Haario, M Laine, A Mira, E Saksman, (2006) Statistical Computing
           https://doi.org/10.1007/s11222-006-9438-0
    """
    def __init__(self, x0, sigma0=None):
        super(DramACMC, self).__init__(x0, sigma0)

        self._log_lambda = 0
        self._n_kernels = 2
        self._Y = [None] * self._n_kernels
        self._Y_log_pdf = np.zeros(self._n_kernels)
        self._proposal_count = 0
        self._adapt_kernel = True
        self._before_kernels_set = True

    def ask(self):
        """
        If first proposal from a position, return a proposal with an ambitious
        (i.e. large) proposal width; if first is rejected then return
        proposal from a conservative kernel (i.e. with low width).
        """
        super(DramACMC, self).ask()
        if self._before_kernels_set:
            self._sigma_base = np.copy(self._sigma)
            self.set_sigma_scale(1000)
            self._before_kernels_set = False
            self._Y = [None] * self._n_kernels
            self._Y_log_pdf = np.zeros(self._n_kernels)

        # Propose new point
        if self._proposed is None:
            self._proposed = np.random.multivariate_normal(
                self._current, np.exp(self._log_lambda) *
                self._sigma[self._proposal_count])
            self._Y[self._proposal_count] = np.copy(self._proposed)
            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def _calculate_alpha_log(self, n, Y, log_Y):
        """
        Calculates alpha expression necessary in eq. 3 of Haario et al. for
        determining accept/reject
        """
        alpha_log = log_Y[n + 1] - log_Y[0]
        if n == 0:
            return min(0, alpha_log)
        Y_rev = Y[::-1]
        log_Y_rev = log_Y[::-1]
        for i in range(n):
            alpha_log += (
                stats.multivariate_normal.logpdf(
                    x=Y[n - i - 1],
                    mean=Y[n + 1],
                    cov=self._sigma[n],
                    allow_singular=True) -
                stats.multivariate_normal.logpdf(
                    x=Y[i],
                    mean=self._current,
                    cov=self._sigma[0],
                    allow_singular=True) +
                np.log(1 - np.exp(self._calculate_alpha_log(
                    i, Y_rev[0:(i + 2)], log_Y_rev[0:(i + 2)]))) -
                np.log(1 - np.exp(self._calculate_alpha_log(
                    i, Y[0:(i + 2)], log_Y[0:(i + 2)])))
            )
        return min(0, alpha_log)

    def _calculate_r_log(self, fx):
        """
        Calculates value of logged acceptance ratio (eq. 3 in [1]_).
        """
        c = self._proposal_count
        temp_Y = np.concatenate([[self._current], self._Y[0:(c + 1)]])
        temp_log_Y = np.concatenate([[self._current_log_pdf],
                                     self._Y_log_pdf[0:(c + 1)]])
        self._r_log = self._calculate_alpha_log(c, temp_Y,
                                                temp_log_Y)

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Delayed Rejection Adaptive Metropolis (Dram) MCMC'

    def n_kernels(self):
        """ Returns number of proposal kernels. """
        return self._n_kernels

    def set_n_kernels(self, n_kernels):
        """ Sets number of proposal kernels. """
        if n_kernels < 1:
            raise ValueError('Number of proposal kernels must be equal to ' +
                             'or greater than 1.')
        self._n_kernels = int(n_kernels)

    def set_sigma_scale(self, upper, lower=1):
        """
        Set the scale of initial covariance matrix multipliers for each of the
        kernels: ``[lower,...,upper]`` where the gradations are uniform on the
        log10 scale meaning the proposal covariance matrices are:
        ``[10^upper,..., 10^lower] * sigma``. By default ``lower=1``.
        """
        if lower > upper:
            raise ValueError('Maximum kernel multiplier must exceed minimum.')
        a_min = np.log10(lower)
        a_max = np.log10(upper)
        self._sigma_scale = np.flip(
            10**np.linspace(a_min, a_max, self._n_kernels), 0)
        self._sigma = [self._sigma_scale[i] * self._sigma_base
                       for i in range(self._n_kernels)]

    def sigma_scale(self):
        """
        Returns scale factors used to multiply a base covariance matrix,
        resulting in proposal matrices for each accept-reject step.
        """
        return self._sigma_scale

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

        self._proposed = None

        # Return new point for chain
        if accepted == 0:
            # rejected proposal
            if self._n_kernels > 1 and (
               self._proposal_count < (self._n_kernels - 1)):
                self._proposal_count += 1
                return None
            else:
                if self._adaptive:
                    self._gamma = (self._adaptations**-self._eta)
                    self._adaptations += 1

                    # Update mu, covariance matrix and log lambda
                    self._update_mu()
                    self._update_sigma()
                    self._log_lambda += (self._gamma *
                                         (accepted - self._target_acceptance))
                self._proposal_count = 0
        # if accepted or failed on second try
        return self._current

    def _update_sigma(self):
        """
        Updates the covariance matrices of the various kernels being used
        according to adaptive Metropolis routine.
        """
        dsigm = np.reshape(self._current - self._mu, (self._n_parameters, 1))
        self._sigma_base = ((1 - self._gamma) * self._sigma_base +
                            self._gamma * np.dot(dsigm, dsigm.T))
        self._sigma = [self._sigma_scale[i] * self._sigma_base
                       for i in range(self._n_kernels)]
