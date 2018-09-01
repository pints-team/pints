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


class AdaptiveCovarianceRemiMCMC(pints.AdaptiveCovarianceMCMC):
    """
    Adaptive covariance MCMC, as described in [1, 2].

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
        super(AdaptiveCovarianceRemiMCMC, self).__init__(x0, sigma0)

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(AdaptiveCovarianceRemiMCMC, self).ask()
        # Propose new point
        if self._proposed is None:

            # Note: Normal distribution is symmetric
            #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
            #  distribution term from the acceptance criterion
            self._proposed = np.random.multivariate_normal(self._current, np.exp(self._loga) * self._sigma)

            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def _initialise(self):
        """
        See :meth: `AdaptiveCovarianceMCMC._initialise()`.
        """
        super(AdaptiveCovarianceRemiMCMC, self)._initialise()
        
        # log adaptation
        self._loga = 0

    def tell(self, fx):
        """ See :meth:`pints.AdaptiveCovarianceMCMC.tell()`. """
        super(AdaptiveCovarianceRemiMCMC, self).tell(fx)
        # Update log acceptance
        if self._adaptive:
            self._loga += self._gamma * (self._accepted - self._target_acceptance)
        # Return new point for chain
        return self._current

