#
# Functions to calculate various MCMC diagnostics
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

import numpy as np


def autocorrelation(samples):
    r"""
    Calculates the autocorrelation of samples for different lags using
    a spectrum density calculation.

    The autocorrelation of MCMC samples at lag :math:`n` is defined
    as the mean correlation between the samples in the chain and the
    samples shifted by :math:`n` iterations

    .. math::
        \rho _n = \frac{1}{\sigma ^2}\int \text{d}\, \theta ,
        (\theta _0 - \mu ) (\theta _n - \mu ) p(\theta ),

    where :math:`p(\theta )` is the converged distribution of samples
    at any iteration, and :math:`\mu ` and :math:`\sigma ^2` are the
    mean and variance of the samples according to that distribution.
    Here teh subscript indicates whether the sample is drawn from the
    original or shifted chain.

    In practice we will approximate the above expression by the finite
    sample size analogon

    .. math::
        \hat{\rho} _n = \frac{1}{N\hat{\sigma} ^2}\int sum _i
        (\theta _i - \hat{\mu }) (\theta _{i+n} - \hat{\mu } ),

    where :math:`N` is the total number of samples in the chain and
    :math:`\theta _i` is the sample at iteration :math:`i`.

    If samples for multiple parameters and/or chains are provided, the
    autocorrelation for different lags is computed for all marginal
    chains independently.

    .. note::
        The computation of the autocorrelation is only accurate if the
        samples are ordered according to theit sample iteration.

    Parameters
    ----------
    samples np.ndarray of shape (n,), (m, n) or (m, n, p)
        A numpy array with :math:`n` samples. Optionally the autocorrelation
        may be computed simultaneously for :math:`m` chains and :math:`p`
        parameters.
    """
    # Make sure samples are one-dimensional
    samples = np.asarray(samples)
    if samples.dims > 3:
        raise ValueError(
            'The samples can have at most the-dimensions.')

    # Reshape samples for later convenience
    if samples.dims == 1:
        # Add chain and parameter dimension
        samples = samples[np.newaxis, :, np.newaxis]

    elif samples.dims == 2:
        # Add parameter dimension
        samples = samples[:, :, np.newaxis]

    # Standardise sample (center and normalise by std.)
    samples = (samples - np.mean(samples, axis=1)) / np.std(samples, axis=1)

    # Create container for autocorrelations
    n_chains, n_samples, n_parameters = samples.shape
    autocorrelations = np.empty(shape=(n_chains, n_samples, n_parameters))

    # Compute autocorrelations for each chain and each parameter
    samples = samples.swapaxes(samples, axis1=1, axis2=2)
    for chain_id, chain_samples in enumerate(samples):
        for param_id, parameter_samples in enumerate(chain_samples):
            # Compute mean correlation
            # (np.correlate returns cumulative correlation)
            autocorr = np.correlate(
                parameter_samples, parameter_samples, mode='full')
            autocorr = autocorr / n_samples

            # At the center the correlation is 1 because the sliding samples
            # exactly overlap with the other samples, i.e. we compute compute
            # the correlation of each sample with itself.
            # (autocorr has length 2*n_samples - 1)
            autocorr = autocorr[n_samples:]

            # Add to container
            autocorrelations[chain_id, :, param_id]

    # Remove padded dimensions
    if n_parameters == 1:
        autocorrelations = autocorrelations[:, :, 0]
    if n_chains == 1:
        autocorrelations = autocorrelations[0]

    return autocorr


def _autocorrelate_negative(autocorrelation):
    """
    Returns the index of the first negative entry in ``autocorrelation``, or
    ``len(autocorrelation)`` if no negative entry is found.
    """
    try:
        return np.where(np.asarray(autocorrelation) < 0)[0][0]
    except IndexError:
        return len(autocorrelation)


def effective_sample_size_single_parameter(x):
    """
    Calculates effective sample size (ESS) for samples of a single parameter.

    Parameters
    ----------
    x
        A sequence (e.g. a list or a 1-dimensional array) of parameter values.
    """
    rho = autocorrelation(x)
    T = _autocorrelate_negative(rho)
    n = len(x)
    ess = n / (1 + 2 * np.sum(rho[0:T]))
    return ess


def effective_sample_size(samples):
    """
    Calculates effective sample size (ESS) for a list of n-dimensional samples.

    Parameters
    ----------
    samples
        A 2d array of shape ``(n_samples, n_parameters)``.
    """
    try:
        n_samples, n_params = samples.shape
    except (ValueError, IndexError):
        raise ValueError('Samples must be given as a 2d array.')
    if n_samples < 2:
        raise ValueError('At least two samples must be given.')

    return [effective_sample_size_single_parameter(samples[:, i])
            for i in range(0, n_params)]
