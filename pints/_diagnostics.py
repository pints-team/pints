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
    Calculate autocorrelation for a vector x using a spectrum density
    calculation.
    """
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(len(x)))
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]


def autocorrelate_negative(autocorrelation):
    """
    Finds last positive autocorrelation, T.
    """
    T = 1
    for a in autocorrelation:
        if a < 0:
            return T - 1
        T += 1
    return T


def ess_single_param(x):
    """
    Calculates ESS for a single parameter.
    """
    rho = autocorrelation(x)
    T = autocorrelate_negative(rho)
    n = len(x)
    ess = n / (1 + 2 * np.sum(rho[0:T]))
    return ess


def effective_sample_size(samples):
    """
    Calculates ESS for a matrix of samples.
    """
    try:
        n_samples, n_params = samples.shape
    except (ValueError, IndexError):
        raise ValueError('Samples must be given as a 2d array.')
    if n_samples < 2:
        raise ValueError('At least two samples must be given.')

    return [ess_single_param(samples[:, i]) for i in range(0, n_params)]


def within(samples):
    """
    Calculates within-chain variance.
    """
    mu = list(map(lambda x: np.var(x, ddof=1), samples))
    W = np.mean(mu)
    return W


def between(samples):
    """
    Calculates between-chain variance.
    """
    mu = list(map(lambda x: np.mean(x), samples))
    mu_overall = np.mean(mu)
    m = len(samples)
    t = len(samples[0])
    return (t / (m - 1.0)) * np.sum((mu - mu_overall) ** 2)


def reorder(param_number, chains):
    """
    Reorders chains for a given parameter into a more useful format for
    calculating rhat.
    """
    # split chains in two
    a_len = int(chains.shape[1] / 2)
    chains_first = [chain[:a_len, :] for chain in chains]
    chains_second = [chain[a_len:, :] for chain in chains]
    chains = chains_first + chains_second
    num_chains = len(chains)
    samples = [chains[i][:, param_number] for i in range(0, num_chains)]
    return samples


def reorder_all_params(chains):
    """
    Reorders chains for all parameters into a more useful format for
    calculating rhat.
    """
    num_params = chains[0].shape[1]
    samples_all = [reorder(i, chains) for i in range(0, num_params)]
    return samples_all


def rhat(chains):
    r"""
    Returns the convergence measure :math:`\hat{R}` according to [1]_.

    :math:`\hat{R}` diagnoses convergence by checking mixing and stationarity
    of :math:`M` chains (at least two, :math:`M\geq 2`). To diminish the
    influence of starting values, the first half of each chain is considered
    as warm up and does not enter the calculation. Subsequently, the truncated
    chains are split in half yet again, and the mean of the variances within
    and between the resulting chains are computed. Based on the mean within
    variance :math:`W` and the mean between variance :math:`B` (definition
    below) is an estimator of the marginal posterior variance constructed

    .. math::
        \widehat{\text{var}}^+ = \frac{n-1}{n}W + \frac{1}{n}B,

    where :math:`n` is the length of the individual chains (i.e. :math:`4n` is
    the length of the original chains). The estimator overestimates the
    variance of the marginal posterior if the chains are not well mixed and
    stationary, but is unbiased if the original chains equal the target
    distribution. At the same time, the mean within variance :math:`W`
    underestimates the marginal posterior variance for finite :math:`n`, but
    converges to the true variance for :math:`n\rightarrow \infty`. By
    comparing :math:`\widehat{\text{var}}^+` and :math:`W` the mixing and
    stationarity of the chains can be quantified

    .. math::
        \hat{R} = \sqrt{\frac{\widehat{\text{var}}^+}{W}}.

    For well mixed and stationary chains :math:`\hat{R}` will be close to one.

    The mean within :math:`W` and mean between :math:`B` variance of the
    :math:`m=2M` chains of length :math:`n` (original length of the chains
    :math:`4n`) is defined as

    .. math::
        W = \frac{1}{m}\sum _{j=1}^{m}s_j^2\quad \text{where}\quad
        s_j^2=\frac{1}{n-1}\sum _{i=1}^n(\psi _{ij} - \bar{\psi} _j)^2,

    .. math::
        B = \frac{n}{m-1}\sum _{j=1}^m(\bar{\psi} _j - \bar{\psi})^2.

    Here :math:`\bar{\psi _j}=\sum _{i=1}^n\psi _{ij}/n` is the within chain
    mean of the parameter :math:`\psi` and
    :math:`\bar{\psi _j} = \sum _{j=1}^m\bar{\psi} _{j}/m` is the between
    chain mean of the within chain means.
    """
    W = within(chains)
    B = between(chains)
    t = len(chains[0])
    return np.sqrt((W + (1.0 / t) * (B - W)) / W)


def rhat_all_params(chains):
    r"""
    Calculates :math:`\hat{R} = \sqrt{(W(n - 1) / n + (1 / n) B) / W}` as per
    [1]_ for all parameters. It does this after splitting each chain into two.

    References
    ----------
    ..  [1] "Bayesian data analysis", 3rd edition, Gelman et al., 2014.
    """
    samples_all = reorder_all_params(chains)
    rhat_all = list(map(lambda x: rhat(x), samples_all))
    return rhat_all
