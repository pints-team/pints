#
# Base class for global adaptive covariance MCMC methods
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class GlobalAdaptiveCovarianceMC(pints.AdaptiveCovarianceMC):
    """
    Base class for single chain MCMC methods that globally adapt a proposal
    covariance matrix when running, in order to control the acceptance rate.

    Extends :class:`AdaptiveCovarianceMC`.
    """
    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        super(GlobalAdaptiveCovarianceMC, self).tell(fx)

        self._accepted = 0
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < self._log_acceptance_ratio:
                self._accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx
                self._accepted_count += 1

        self._acceptance = (
            float(self._accepted_count) / float(self._iterations))

        # Clear proposal
        self._proposed = None

        # Adapt covariance matrix
        if self._adaptive:
            # Set gamma based on number of adaptive iterations
            self._gamma = (self._adaptations + 1) ** -self._eta
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
        dsigm = np.reshape(self._current - self._mu, (self._n_parameters, 1))
        self._sigma = ((1 - self._gamma) * self._sigma +
                       self._gamma * np.dot(dsigm, dsigm.T))
