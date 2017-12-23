#
# Functions to calculate various MCMC diagnostics
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
import numpy as np


def autocorrelation(x):
    """
    Calculate autocorrelation for a vector x using a spectrum density
    calculation.
    """
    xp = x - np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size / 2] / np.sum(xp**2)


def autocorrelate_negative(autocorrelation):
    """
    ## finds last positive autocorrelation, T
    """
    T = 1
    for a in autocorrelation:
        if a < 0:
            return T - 1
        T += 1
    return -1


def ess_single_param(x):
    """
    ## calculates ESS for a single parameter
    """
    rho = autocorrelation(x)
    T = autocorrelate_negative(rho)
    n = len(x)
    ess = n / (1 + 2 * np.sum(rho[0:(T - 1)]))
    return ess


def effective_sample_size(sample):
    """
    ## calculates ESS for a matrix of samples
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

