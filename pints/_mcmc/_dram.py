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
            # high (risky) proposal width
            if self._first_proposal:
                self._proposed = np.random.multivariate_normal(
                      self._current, np.exp(self._loga) * self._sigma)
                self._first_proposal = True
                self._Y1 = np.copy(self._proposed)
            # low (risk) proposal width
            else:
                self._proposed = np.random.multivariate_normal(
                      self._current, np.exp(self._loga) * self._sigma)
                self._Y2 = np.copy(self._proposed)
            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def _initialise(self):
        """
        See :meth: `AdaptiveCovarianceMCMC._initialise()`.
        """
        super(DramMCMC, self)._initialise()
        # First proposal
        self._first_proposal = True
        self._Y1 = 0
        self._Y2 = 0
        self._Y1_log_pdf = float('-Inf')

    def tell(self, fx):
        """
        
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
        r = fx - self._current_log_pdf
        
         # First or second proposal
        if self._first_proposal:
            self._alpha_x_y = r
            self._Y1_log_pdf = fx
        else:
            # modify according to eqn. (2)
            r += (1 - (self._Y1_log_pdf - fx)) - (1 - self._alpha_x_y) 

        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < r:
                accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Clear proposal
        self._proposed = None
        # Adapt covariance matrix
        if self._adaptive:
            # Set gamma based on number of adaptive iterations
            gamma = self._adaptations ** -0.6
            self._adaptations += 1

            # Update mu, log acceptance rate, and covariance matrix
            self._mu = (1 - gamma) * self._mu + gamma * self._current
            self._loga += gamma * (accepted - self._target_acceptance)
            dsigm = np.reshape(self._current - self._mu, (self._dimension, 1))
            self._sigma = (
                (1 - gamma) * self._sigma + gamma * np.dot(dsigm, dsigm.T))

        # Update acceptance rate (only used for output!)
        self._acceptance = ((self._iterations * self._acceptance + accepted) /
                            (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

        # Return new point for chain
        if accepted == 0:
            # rejected first proposal
            if not self._first_proposal:
                self._first_proposal = False
                return None
        # if accepted or failed on second try
        return self._current

    def replace(self, x, fx):
        """ See :meth:`pints.SingleChainMCMC.replace()`. """
        # Must already be running
        if not self._running:
            raise RuntimeError(
                'Replace can only be used when already running.')

        # Must be after tell, before ask
        if self._proposed is not None:
            raise RuntimeError(
                'Replace can only be called after tell / before ask.')

        # Check values
        x = pints.vector(x)
        if not len(x) == len(self._current):
            raise ValueError('Dimension mismatch in `x`.')
        fx = float(fx)

        # Store
        self._current = x
        self._current_log_pdf = fx

    def set_target_acceptance_rate(self, rate=0.234):
        """
        Sets the target acceptance rate.
        """
        rate = float(rate)
        if rate <= 0:
            raise ValueError('Target acceptance rate must be greater than 0.')
        elif rate > 1:
            raise ValueError('Target acceptance rate cannot exceed 1.')
        self._target_acceptance = rate

    def target_acceptance_rate(self):
        """
        Returns the target acceptance rate.
        """
        return self._target_acceptance

