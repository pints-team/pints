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
    :math:`\rho _n` is zero for all lags greater :math:`n\neq 0`. For
    :math:`n=0` the correlation is one.

    In practice we will approximate the above expression by the finite
    sample size analogon

    .. math::
        \hat{\rho} _n = \frac{1}{N\hat{\sigma} ^2}\sum ^N_{i=1}
        (\theta _i - \hat{\mu }) (\theta _{i+n} - \hat{\mu } ),

    where :math:`N` is the total number of samples in the chain and
    :math:`\theta _i` is the sample at iteration :math:`i`. Here,
    :math:`\hat{\mu }` and :math:`\hat{\sigma } ^2` are the estimates
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

    :param samples: A numpy array with :math:`n` samples and :math:`p`
        parameters. Optionally the autocorrelation may be computed
        simultaneously for :math:`m` chains.
    :type samples: np.ndarray of shape (n, p) or (m, n, p)
    """
    # Make sure samples are one-dimensional
    samples = np.asarray(samples)
    if (samples.dims < 2) or (samples.dims > 3):
        raise ValueError(
            'The samples array must have 2 or 3 dimensions.')

    # Reshape samples for later convenience
    if samples.dims == 2:
        # Add chain dimension
        samples = samples[np.newaxis, :, :]

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
    if n_chains == 1:
        autocorrelations = autocorrelations[0]

    return autocorrelations


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


def effective_sample_size(samples, combine_chains=True):
    r"""
    Estimates the effective samples size (ESS) of MCMC chains.

    The effective sample size approximates the effective number of i.i.d.
    samples generated by a MCMC routine. For a single chain the effective
    samples size is defined as

    .. math::
        N_{\text{eff}} = \frac{N}{1 + 2\sum ^{\infty}_{n=1}\rho _n},

    where :math:`N` is the number of samples and :math:`\rho _n` is the
    mean correlation between the samples and the samples at lag :math:`n`, see
    :func:`autocorrelation`. The autocorrelation at lag
    :math:`n` and at :math:`-n` yield the same value, so we focus on
    positive lags and multiply by 2.

    Intuitively, the denominator reduces to 1 (:math:`N_{\text{eff}}=N`)
    when there is no correlation between the samples and will be greater
    than 1 (:math:`N_{\text{eff}}<N`) when there is positive correlation
    between the samples. Some samplers may also lead to negative
    correlation between samples (:math:`N_{\text{eff}}>N`) which is
    referred to as superefficient sampling.

    The error in the autocorrelation estimate :math:`\hat{\rho }_n`
    increases with the lag which motivates to truncate the autocorrelation
    sum in practice

    .. math::
        N_{\text{eff}} = \frac{N}{1 + 2\sum ^{t}_{n=1}\hat{\rho} _n},

    where :math:`t` is the truncation lag. We follow a widely accepted
    truncation criterion introduced by Geyer.

    For :math:`M` chains with :math:`N` samples the total effective
    sample size is similarly estimated by

    .. math::
        N_{\text{eff}} = \frac{NM}{1 + 2\sum ^{t}_{n=1}\tilde{\rho} _n},

    where :math:`\tilde{\rho} _n` is a autocorrelation measure that
    incorporates the autocorrelations within the individual chains, the
    explored parameter range of the individual chains and the overall
    convergence

    .. math::
        \tilde{\rho} _n =
        1 - \frac{1 - \bar{\rho}_n}{\hat{R}^2}.

    Here :math:`\bar{\rho}_n` is the mean of the autocorrelations weighted
    by the normalised within chain variances

    .. math::
        \bar{\rho}_n = \frac{1}{M}\sum ^M_{m=1}w_m \hat{\rho}_{nm}\quad
        \text{and} \quad w_m = \frac{\sigma ^2_m}{W},

    where :math:`\hat{\rho}_{nm}` is the autocorrelation of chain
    :math:`m` at lag :math:`n`. :math:`W` is the mean within chain variance
    estimator and :math:`\hat{R}` is the MCMC chain convergence metric,
    see :func:`rhat`. Note that :math:`\bar{\rho} _0=1`.

    Intuitivley, :math:`\tilde{\rho}` tries to incorporate the mixing /
    convergence of the chains by comparing the normalised mean
    autocorrelation :math:`\bar{\rho}_n` to the convergence
    of the chains :math:`\hat{R}`. If the convergence is terrible
    :math:`\hat{R} \gg 1` the correlation of the samples is high, even if the
    estimated within chain autocorrelation is low
    :math:`\bar{\rho}_n \approx 0`. On the other hand if the mean within chain
    autocorrelation is large :math:`\bar{\rho}_n \approx 1` the overall
    autocorrelation is large irrespective of the convergence.

    :param samples: A numpy array with :math:`n` samples and :math:`p`
        parameters. Optionally the autocorrelation may be computed
        simultaneously for :math:`m` chains.
    :type samples: np.ndarray of shape (n, p) or (m, n, p)
    :param combine_chains: A boolean that determines whether the effective
        samples sizes of multiple chains are kept separate or combined a
        ccording to the above equation. If ``False`` the effective sample
        sizes for each chain are returned separately as a
        :class:`numpy.ndarray` of shape (m, p).
    :type combine_chains: bool, optional
    """
    try:
        n_samples, n_params = samples.shape
    except (ValueError, IndexError):
        raise ValueError('Samples must be given as a 2d array.')
    if n_samples < 2:
        raise ValueError('At least two samples must be given.')

    return [effective_sample_size_single_parameter(samples[:, i])
            for i in range(0, n_params)]
