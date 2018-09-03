#
# Adaptive covariance MCMC method
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
from scipy.misc import logsumexp
import scipy

class DramMCMC(pints.AdaptiveCovarianceMCMC):
    """
    DRAM (Delayed Rejection Adaptive Covariance) MCMC, as described in [1].

    Using a covariance matrix, that is tuned so that the acceptance rate of the
    MCMC steps converges to a user specified value.

    [1] Uncertainty and variability in models of the cardiac action potential:
    Can we build trustworthy models?
    Johnstone, Chang, Bardenet, de Boer, Gavaghan, Pathmanathan, Clayton,
    Mirams (2015) Journal of Molecular and Cellular Cardiology

    [2] An adaptive Metropolis algorithm
    Heikki Haario, Eero Saksman, and Johanna Tamminen (2001) Bernoulli

    *Extends:* :class:`AdaptiveCovarianceMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(DramMCMC, self).__init__(x0, sigma0)

    def ask(self):
        """
        If first proposal from a position, return
        a proposal with an ambitious (i.e. large)
        proposal width; if first is rejected
        then return a proposal from a conservative
        kernel (i.e. with low width)
        """
        super(DramMCMC, self).ask()

        # Propose new point
        if self._proposed is None:
            self._proposed = np.random.multivariate_normal(
                                self._current, np.exp(self._loga) * self._sigma[self._proposal_count])
            self._Y[self._proposal_count] = np.copy(self._proposed)
            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def _initialise(self):
        """
        See :meth: `AdaptiveCovarianceMCMC._initialise()`.
        """
        super(DramMCMC, self)._initialise()

        self._kernels = 2
        self._Y = [None] * self._kernels
        self._Y_log_pdf = np.repeat(float('-Inf'), self._kernels)
        self._proposal_count = 0
        self._adapt_kernel = np.repeat(True, self._kernels)

        # Set kernels
        v_mu = np.copy(self._mu)
        self._mu = [v_mu for i in range(self._kernels)]
        self._sigma_scale = np.array([100, 1])
        m_sigma = np.copy(self._sigma)
        self._sigma = [self._sigma_scale[i] * m_sigma for i in range(self._kernels)]

    def set_sigma_scale(self, minK, maxK):
        """
        Set the scale of initial covariance matrix
        multipliers for each of the kernels:
        (minK,...,maxK) where the gradations are
        uniform on the log10 scale.
        This means that the covariance matrices are:
        maxK * sigma0,..., MinK * sigma0
        where n can be modified by `set_kernels`.
        """
        if a_min > a_max:
            raise ValueError('Maximum kernel multiplier must ' +
                             'exceed minimum.')
        a_min=np.log10(minK)
        a_max=np.log10(maxK)
        self._sigma_scale = 10**np.linspace(a_min, a_max, self._kernels)
        self._sigma = [self._sigma_scale[i] * m_sigma for i in range(self._kernels)]

    def tell(self, fx):
        """
        If first proposal, then accept with ordinary
        Metropolis probability: _alpha_x_y = min(1, )
        """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure fx is a float
        fx = float(fx)

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
        r_log = fx - self._current_log_pdf

         # First proposal?
        if self._proposal_count == 0:
            self._alpha_x_y_log = min(0, r_log)
            self._Y1_log_pdf = fx
        else:
            # modify according to eqn. (2)
            r_log += (np.log(1 - np.exp((self._Y1_log_pdf - fx))) -
                      np.log(1 -np.exp(self._alpha_x_y_log)) +
                      scipy.stats.multivariate_normal.logpdf(x=self._Y[0],mean=self._Y[1],cov=self._sigma[1],allow_singular=True) -
                      scipy.stats.multivariate_normal.logpdf(x=self._Y[1],mean=self._Y[0],cov=self._sigma[1],allow_singular=True))

        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < r_log:
                accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Clear proposal
        self._proposed = None
        # Adapt covariance matrix
        if self._adaptive:
            # Set gamma based on number of adaptive iterations
            self._gamma = self._adaptations ** -self._eta
            self._adaptations += 1

            # Update mu, log acceptance rate, and covariance matrix
            self._update_mu()
            self._update_sigma()

        # Update acceptance rate (only used for output!)
        self._acceptance = ((self._iterations * self._acceptance + accepted) /
                            (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

        # Return new point for chain
        if accepted == 0:
            # rejected first proposal
            if self._proposal_count == 0:
                self._proposal_count += 1
                return None
            else:
                self._proposal_count = 0
        # if accepted or failed on second try
        return self._current

    def _update_mu(self):
        """
        Updates the means of the various kernels being used,
        according to adaptive Metropolis routine (later this
        will be able to be swapped with another routine), if
        adaptation is turned on
        """
        if self._adapt_kernel[self._proposal_count]:
            self._mu[self._proposal_count] = ((1 - self._gamma) *
                                               self._mu[self._proposal_count] +
                                               self._gamma * self._current)

    def _update_sigma(self):
        """
        Updates the covariance matrices of the various kernels being used,
        according to adaptive Metropolis routine (later this
        will be able to be swapped with another routine), if
        adaptation is turned on
        """
        if self._adapt_kernel:
              dsigm = np.reshape(self._current - self._mu[self._proposal_count], (self._dimension, 1))
              self._sigma[self._proposal_count] = ((1 - self._gamma) * self._sigma[self._proposal_count] +
                                                  self._gamma * np.dot(dsigm, dsigm.T))
