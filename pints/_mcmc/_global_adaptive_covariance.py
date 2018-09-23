#
# Base class for Adaptive covariance MCMC methods
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


class GlobalAdaptiveCovarianceMCMC(pints.AdaptiveCovarianceMCMC):
    """
    Base class for single chain MCMC methods that adapt a covariance matrix
    when running, in order to control the acceptance rate. This base class
    is for those methods which globally adapt the covariance matrix.

    In all cases ``self._adaptations ^ -eta`` is used to control decay of
    adaptation

    *Extends:* :class:`SingleChainMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(GlobalAdaptiveCovarianceMCMC, self).__init__(x0, sigma0)

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(GlobalAdaptiveCovarianceMCMC, self).ask()

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        super(GlobalAdaptiveCovarianceMCMC, self).tell(fx)

        self._alpha = np.minimum(1, np.exp(self._r))

        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < self._r:
                self._accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Clear proposal
        self._proposed = None

        # Adapt covariance matrix
        if self._adaptive:
            # Set gamma based on number of adaptive iterations
            self._gamma = self._adaptations ** -self._eta
            self._adaptations += 1
            self._update_mu()
            self._update_sigma()
        return self._current

    def _update_mu(self):
        """
        Updates the current running mean used to calculate the sample
        covariance matrix of proposals. Note that this default is overidden in
        some of the methods
        """
        self._mu = (1 - self._gamma) * self._mu + self._gamma * self._current

    def _update_sigma(self):
        """
        Updates the covariance matrix used to generate proposals.
        Note that this default is overidden in some of the methods
        """
        dsigm = np.reshape(self._current - self._mu, (self._dimension, 1))
        self._sigma = ((1 - self._gamma) * self._sigma +
                       self._gamma * np.dot(dsigm, dsigm.T))
