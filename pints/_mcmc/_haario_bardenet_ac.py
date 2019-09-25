#
# Haario-Bardenet adaptive covariance MCMC method
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


class HaarioBardenetACMC(pints.GlobalAdaptiveCovarianceMCMC):
    """
    Adaptive Metropolis MCMC, which is algorithm in the supplementary materials
    of [1]_, which in turn is based on [2]_.

    Initialise::

        mu
        Sigma
        adaptation_count = 0
        log lambda = 0

    In each adaptive iteration (t)::

        adaptation_count = adaptation_count + 1
        gamma = (adaptation_count)^-eta
        theta* ~ N(theta_t, lambda * Sigma)
        alpha = min(1, p(theta*|data) / p(theta_t|data))
        u ~ uniform(0, 1)
        if alpha > u:
            theta_(t+1) = theta*
            accepted = 1
        else:
            theta_(t+1) = theta_t
            accepted = 0

        alpha = accepted

        mu = (1 - gamma) mu + gamma theta_(t+1)
        Sigma = (1 - gamma) Sigma + gamma (theta_(t+1) - mu)(theta_(t+1) - mu)
        log lambda = log lambda + gamma (alpha - self._target_acceptance)
        gamma = adaptation_count^-eta

    Extends :class:`GlobalAdaptiveCovarianceMCMC`.

    References
    ----------
    .. [1] Johnstone, Chang, Bardenet, de Boer, Gavaghan, Pathmanathan,
           Clayton, Mirams (2015) "Uncertainty and variability in models of the
           cardiac action potential: Can we build trustworthy models?"
           Journal of Molecular and Cellular Cardiology.
           https://10.1016/j.yjmcc.2015.11.018

    .. [2] Haario, Saksman, Tamminen (2001) "An adaptive Metropolis algorithm"
           Bernoulli.
           https://doi.org/10.2307/3318737
    """
    def __init__(self, x0, sigma0=None):
        super(HaarioBardenetACMC, self).__init__(x0, sigma0)
        self._log_lambda = 0
        self._binary_accept = True
        self._accepted = True

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(HaarioBardenetACMC, self).ask()

        # Propose new point
        if self._proposed is None:
            self._proposed = (
                np.random.multivariate_normal(self._current,
                                              ((np.exp(self._log_lambda) *
                                               self._sigma)))
            )

            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Haario-Bardenet adaptive covariance MCMC'

    def tell(self, fx):
        """ See :meth:`pints.AdaptiveCovarianceMCMC.tell()`. """
        super(HaarioBardenetACMC, self).tell(fx)

        _acceptance_prob = self._accepted
        if self._adaptive:
            self._log_lambda += (self._gamma *
                                 (_acceptance_prob -
                                  self._target_acceptance))

        # Return new point for chain
        return self._current
