#
# Haario-Bardenet adaptive covariance MCMC method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np


class HaarioBardenetACMC(pints.AdaptiveCovarianceMC):
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

    Extends :class:`AdaptiveCovarianceMC`.

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

        # Initial log lambda is zero
        self._log_lambda = 0

    def _adapt_internal(self, accepted, log_ratio):
        """ See :meth:`pints.AdaptiveCovarianceMC.tell()`. """
        p = 1 if accepted else 0
        self._log_lambda += self._gamma * (p - self._target_acceptance)

    def _generate_proposal(self):
        """ See :meth:`AdaptiveCovarianceMC._generate_proposal()`. """
        return np.random.multivariate_normal(
            self._current, self._sigma * np.exp(self._log_lambda))

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Haario-Bardenet adaptive covariance MCMC'


class AdaptiveCovarianceMCMC(HaarioBardenetACMC):
    """
    Deprecated alias of :class:`pints.HaarioBardenetACMC`.
    """

    def __init__(self, x0, sigma0=None):

        # Deprecated on 2019-09-26
        import warnings
        warnings.warn(
            'The class `pints.AdaptiveCovarianceMCMC` is deprecated.'
            ' Please use `pints.HaarioBardenetACMC` instead.')
        super(AdaptiveCovarianceMCMC, self).__init__(x0, sigma0)

