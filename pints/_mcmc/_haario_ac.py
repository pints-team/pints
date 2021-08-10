#
# Adaptive covariance MCMC method by Haario
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np


class HaarioACMC(pints.AdaptiveCovarianceMC):
    """
    Adaptive Metropolis MCMC, which is algorithm 4 in [1]_ and is described in
    the text in [2]_.

    This algorithm differs from :class:`HaarioBardenetACMC` only through its
    use of ``alpha`` in the updating of ``log_lambda`` (rather than a binary
    accept/reject).

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

        mu = (1 - gamma) mu + gamma theta_(t+1)
        Sigma = (1 - gamma) Sigma + gamma (theta_(t+1) - mu)(theta_(t+1) - mu)
        log lambda = log lambda + gamma (alpha - self._target_acceptance)
        gamma = adaptation_count^-eta

    Extends :class:`AdaptiveCovarianceMC`.

    References
    ----------
    .. [1] A tutorial on adaptive MCMC
           Christophe Andrieu and Johannes Thoms, Statistical Computing, 2008,
           18: 343-373.
           https://doi.org/10.1007/s11222-008-9110-y

    .. [2] An adaptive Metropolis algorithm
           Heikki Haario, Eero Saksman, and Johanna Tamminen (2001) Bernoulli.
    """
    def __init__(self, x0, sigma0=None):
        super(HaarioACMC, self).__init__(x0, sigma0)
        self._log_lambda = 0

    def _adapt_internal(self, accepted, log_ratio):
        """ See :meth:`pints.AdaptiveCovarianceMC._adapt()`. """
        p = np.exp(log_ratio) if log_ratio < 0 else 1
        self._log_lambda += self._gamma * (p - self._target_acceptance)

    def _generate_proposal(self):
        """ See :meth:`AdaptiveCovarianceMC._generate_proposal()`. """
        return np.random.multivariate_normal(
            self._current, self._sigma * np.exp(self._log_lambda))

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Haario adaptive covariance MCMC'

