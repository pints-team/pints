#
# Functions to calculate various MCMC diagnostics
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np


def autocorrelation(x):
    """
    Calculates autocorrelation for a vector ``x`` using a spectrum density
    calculation.
    """
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(len(x)))
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]


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


def _within(chains):
    r"""
    Calculates mean within-chain variance.

    The mean within chain variance :math:`W` of :math:`m` chains of length
    :math:`n` is defined as

    .. math::
        W = \frac{1}{m}\sum _{j=1}^{m}s_j^2\quad \text{where}\quad
        s_j^2=\frac{1}{n-1}\sum _{i=1}^n(\psi _{ij} - \bar{\psi} _j)^2.

    Here, :math:`\psi _{ij}` is the :math:`j`th sample of the :math:`i`th
    chain and :math:`\bar{\psi _j}=\sum _{i=1}^{n}\psi _{ij}/n` is the within
    chain mean of the parameter :math:`\psi`.

    Parameters
    ----------
    chains : np.ndarray of shape (m, n) or (m, n, p)
        A numpy array with the :math:`n` samples for `:math:`m` chains.
    """
    # Compute unbiased within-chain variance estimate
    within_chain_var = np.var(chains, axis=1, ddof=1)

    # Compute mean-within chain variance
    w = np.mean(within_chain_var, axis=0)

    return w


def _between(chains):
    r"""
    Calculates mean between-chain variance.

    The mean between-chain variance :math:`W` of :math:`m` chains of length
    :math:`n` is defined as

    .. math::
        B = \frac{n'}{m'-1}\sum _{j=1}^{m'}(\bar{\psi} _j - \bar{\psi})^2,

    where :math:`\psi _{ij}` is the :math:`j`th sample of the :math:`i`th
    chain, :math:`\bar{\psi _j}=\sum _{i=1}^{n'}\psi _{ij}/n'` is the within
    chain mean of the parameter :math:`\psi`, and
    :math:`\bar{\psi } = \sum _{j=1}^{m}\bar{\psi} _{j}/m` is the between
    chain mean of the within chain means.

    Parameters
    ----------
    chains : np.ndarray of shape (m, n) or (m, n, p)
        A numpy array with the :math:`n` samples for `:math:`m` chains.
    """
    # Get number of samples
    n = chains.shape[1]

    # Compute within-chain mean
    within_chain_means = np.mean(chains, axis=1)

    # Compute variance across chains of within-chain means
    between_chain_var = np.var(within_chain_means, axis=0, ddof=1)

    # Weight variance with number of samples per chain
    b = n * between_chain_var

    return b


def rhat(chains, warm_up=0.0):
    r"""
    Returns the convergence measure :math:`\hat{R}` for the approximate
    posterior according to [1]_.

    :math:`\hat{R}` diagnoses convergence by checking mixing and stationarity
    of :math:`m` chains (at least two, :math:`m\geq 2`). To diminish the
    influence of starting values, the first portion of each chain can be
    excluded from the computation. Subsequently, the truncated
    chains are split in half, resulting in a total number of :math:`m'=2m`
    chains of length :math:`n'=(1-\text{warm_up})n/2`. The mean of the
    variances within and between the resulting chains are computed, :math:`W`
    and :math:`B` respectively. Based on those variances an estimator of the
    marginal posterior variance is constructed

    .. math::
        \widehat{\text{var}}^+ = \frac{n'-1}{n'}W + \frac{1}{n'}B,

    The estimator overestimates the variance of the marginal posterior
    if the chains are not well mixed and stationary, but is unbiased if the
    original chains equal the target distribution. At the same time, the mean
    within variance :math:`W` underestimates the marginal posterior variance
    for finite :math:`n`, but converges to the true variance for
    :math:`n\rightarrow \infty`. By comparing :math:`\widehat{\text{var}}^+`
    and :math:`W` the mixing and stationarity of the chains can be quantified

    .. math::
        \hat{R} = \sqrt{\frac{\widehat{\text{var}}^+}{W}}.

    For well mixed and stationary chains :math:`\hat{R}` will be close to one.

    The mean within :math:`W` and mean between :math:`B` variance of the
    :math:`m'=2m` chains of length :math:`n'=(1-\text{warm_up})n/2` are defined
    as

    .. math::
        W = \frac{1}{m'}\sum _{j=1}^{m'}s_j^2\quad \text{where}\quad
        s_j^2=\frac{1}{n'-1}\sum _{i=1}^{n'}(\psi _{ij} - \bar{\psi} _j)^2,

    .. math::
        B = \frac{n'}{m'-1}\sum _{j=1}^{m'}(\bar{\psi} _j - \bar{\psi})^2.

    Here, :math:`\psi _{ij}` is the jth sample of the ith
    chain, :math:`\bar{\psi _j}=\sum _{i=1}^{n'}\psi _{ij}/n'` is the within
    chain mean of the parameter :math:`\psi` and
    :math:`\bar{\psi } = \sum _{j=1}^{m'}\bar{\psi} _{j}/m'` is the between
    chain mean of the within chain means.

    References
    ----------
    ..  [1] "Bayesian data analysis", ch. 11.4 'Inference and assessing
        convergence', 3rd edition, Gelman et al., 2014.

    Parameters
    ----------
    chains : np.ndarray of shape (m, n) or (m, n, p)
        A numpy array with :math:`n` samples for each of :math:`m` chains.
        Optionally the :math:`\hat{R}` for :math:`p` parameters can be computed
        by passing a numpy array with :math:`m` chains of length :math:`n`
        for :math:`p` parameters.
    warm_up : float
        First portion of each chain that will not be used for the
        computation of :math:`\hat{R}`.

    Returns
    -------
    rhat : float or np.ndarray of shape (p,)
        :math:`\hat{R}` of the posteriors for each parameter.
    """
    if not (chains.ndim == 2 or chains.ndim == 3):
        raise ValueError(
            'Dimension of chains is %d. ' % chains.ndim
            + 'Method computes Rhat for one '
            'or multiple parameters and therefore only accepts 2 or 3 '
            'dimensional arrays.')
    if warm_up > 1 or warm_up < 0:
        raise ValueError(
            '`warm_up` is set to %f. `warm_up` only takes values in [0,1].' %
            warm_up)

    # Get number of samples
    n = chains.shape[1]

    # Exclude warm-up
    chains = chains[:, int(n * warm_up):]
    n = chains.shape[1]

    # Split chains in half
    n = n // 2  # new length of chains
    if n < 1:
        raise ValueError(
            'Number of samples per chain after warm-up and chain splitting is '
            '%d. Method needs at least 1 sample per chain.' % n)
    chains = np.vstack([chains[:, :n], chains[:, -n:]])

    # Compute mean within-chain variance
    w = _within(chains)

    # Compute mean between-chain variance
    b = _between(chains)

    # Compute Rhat
    rhat = np.sqrt((n - 1.0) / n + b / (w * n))

    return rhat


def rhat_all_params(chains):
    """ Deprecated alias of :func:`rhat`. """
    # Deprecated on 2020-07-01
    import warnings
    warnings.warn(
        'The function `pints.rhat_all_params` is deprecated.'
        ' Please use `pints.rhat` instead.')

    return rhat(chains)
