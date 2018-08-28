#
# Functions to calculate various MCMC diagnostics
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
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
    Finds last positive autocorrelation, T
    """
    T = 1
    for a in autocorrelation:
        if a < 0:
            return T - 1
        T += 1
    return -1


def ess_single_param(x):
    """
    Calculates ESS for a single parameter
    """
    rho = autocorrelation(x)
    T = autocorrelate_negative(rho)
    n = len(x)
    ess = n / (1 + 2 * np.sum(rho[0:(T - 1)]))
    return ess


def effective_sample_size(sample):
    """
    Calculates ESS for a matrix of samples
    """
    try:
        n_sample, n_params = sample.shape
    except IndexError:
        IndexError('There must be at least one parameter')
    assert n_sample > 1

    ess = np.zeros(n_params)
    for i in range(0, n_params):
        ess[i] = ess_single_param(sample[:, i])
    return ess


def within(samples):
    """
    Calculates within chain variance
    """
    mu = list(map(lambda x: np.var(x), samples))
    W = np.mean(mu)
    return W


def between(samples):
    """
    Calculates between chain variance
    """
    mu = list(map(lambda x: np.mean(x), samples))
    mu_overall = np.mean(mu)
    m = len(samples)
    t = len(samples[0])
    return (t / (m - 1.0)) * np.sum((mu - mu_overall) ** 2)


def reorder(param_number, chains):
    """
    Reorders chains for a given parameter into a more useful
    format for calculating rhat
    """
    num_chains = len(chains)
    samples = [chains[i][:, param_number] for i in range(0, num_chains)]
    return samples


def reorder_all_params(chains):
    """
    Reorders chains for all parameters into a more useful
    format for calculating rhat
    """
    num_params = chains[0].shape[1]
    samples_all = [reorder(i, chains) for i in range(0, num_params)]
    return samples_all


def rhat(samples):
    """
    Calculates r-hat = sqrt(((n - 1)/n * W + (1/n) * B)/W) as per
    "Bayesian data analysis", 3rd edition, Gelman et al., 2014
    """
    W = within(samples)
    B = between(samples)
    t = len(samples[0])
    return np.sqrt((W + (1.0 / t) * (B - W)) / W)


def rhat_all_params(chains):
    """
    Calculates r-hat for all parameters in chains
    """
    samples_all = reorder_all_params(chains)
    rhat_all = list(map(lambda x: rhat(x), samples_all))
    return rhat_all

