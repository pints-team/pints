#
# Functions to calculate various MCMC diagnostics
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

import numpy as np


def autocorrelation(chains):
    r"""
    Calculates the autocorrelation of samples for different lags.

    The autocorrelation of MCMC samples at lag :math:`n` is defined
    as the mean correlation between the samples in the chain and the
    samples shifted by :math:`n` iterations

    .. math::
        \rho _n = \frac{1}{\sigma ^2}\int \text{d}\theta \,
        \text{d}\theta _n \,(\theta - \mu ) (\theta _n - \mu )
        \, p(\theta ,\theta _n),

    where :math:`p(\theta , \theta _n )` is the joint distribution of the
    samples and the samples at lag :math:`n`. :math:`\mu` and
    :math:`\sigma ^2` are the mean and variance of the samples which is
    invariant under the lag :math:`n`. If there is no correlation
    between :math:`\theta` and :math:`\theta _n` for :math:`n\neq 0`, i.e.
    if :math:`\theta` and :math:`\theta _n` are i.i.d. distributed,
    :math:`\rho _n` is zero for all lags :math:`n\neq 0`. For
    :math:`n=0` the correlation is always one.

    In practice we will approximate the above expression by the finite
    sample size analogon

    .. math::
        \hat{\rho} _n = \frac{1}{(N-1)\hat{\sigma} ^2}\sum ^N_{i=1}
        (\theta _i - \hat{\mu }) (\theta _{i+n} - \hat{\mu } ),

    where :math:`N` is the total number of samples in the chain and
    :math:`\theta _i` is the sample at iteration :math:`i`. Here,
    :math:`\hat{\mu }` and :math:`\hat{\sigma } ^2` are the estimators
    of the chain mean and variance

    .. math::
        \hat{\mu } = \frac{1}{N}\sum ^N_{i=1}\theta _i \quad \text{and}
        \quad
        \hat{\sigma } ^2 = \frac{1}{N-1}\sum ^N_{i=1}
        \left( \theta _i - \hat{\mu} \right) ^2.

    If samples for multiple parameters and/or chains are provided, the
    autocorrelation for different lags is computed for all marginal
    chains independently.

    .. note::
        The estimation of the autocorrelation is only justified if the
        samples are ordered according to their sample iteration.

    :param chains: A numpy array with :math:`n` samples and :math:`p`
        parameters. Optionally the autocorrelation may be computed
        simultaneously for :math:`m` chains.
    :type chains: np.ndarray of shape (n, p) or (m, n, p)

    Returns
    -------
    autocorr : np.ndarray of shape (m, n, p)
        The autocorrelation of :math:`m` chains for each parameter
        :math:`p` at lag :math:`n`.
    """
    # Make sure samples have the correct dimensions
    chains = np.asarray(chains)
    if (chains.ndim < 2) or (chains.ndim > 3):
        raise ValueError(
            'The chains array must have 2 or 3 dimensions.')

    # Reshape chains for later convenience
    if chains.ndim == 2:
        # Add chain dimension
        chains = chains[np.newaxis, :, :]

    # Create container for autocorrelations
    n_chains, n_samples, n_parameters = chains.shape
    autocorrelations = np.empty(shape=(n_chains, n_samples, n_parameters))

    # Standardise samples
    std = np.std(chains, axis=1, ddof=1)[:, np.newaxis, :]
    mean = np.mean(chains, axis=1)[:, np.newaxis, :]
    chains = (chains - mean) / std

    # Compute autocorrelations for each chain and each parameter
    chains = chains.swapaxes(1, 2)
    for chain_id, chain_samples in enumerate(chains):
        for param_id, parameter_samples in enumerate(chain_samples):
            # Compute mean correlation
            # (np.correlate returns cumulative correlation)
            autocorr = np.correlate(
                parameter_samples, parameter_samples, mode='full')
            autocorr = autocorr / (n_samples - 1)

            # At the center the correlation is 1 because the sliding samples
            # exactly overlap with the other samples, i.e. we compute compute
            # the correlation of each sample with itself.
            # (autocorr has length 2*n_samples - 1)
            autocorr = autocorr[n_samples - 1:]

            # Add to container
            autocorrelations[chain_id, :, param_id] = autocorr

    # For zero std replace nan with 1
    mask = np.broadcast_to(std, autocorrelations.shape) == 0
    autocorrelations[mask] = 1

    return autocorrelations
