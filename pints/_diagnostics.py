#
# Functions to calculate various MCMC diagnostics
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
import numpy as np


def autocorrelation(lx):
    """
    Calculate autocorrelation for a vector x using a spectrum density
    calculation.
    """
    xp = lx - np.mean(lx)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:lx.size / 2] / np.sum(xp**2)


def autocorrelate_negative(l_autocorrelation):
    """
    ## finds last positive autocorrelation, T
    """
    T = 1
    for a in l_autocorrelation:
        if a < 0:
            return T - 1
        T += 1
    return -1


def ess_single_param(lx):
    """
    ## calculates ESS for a single parameter
    """
    l_rho = autocorrelation(lx)
    T = autocorrelate_negative(l_rho)
    n = len(lx)
    ess = n / (1 + 2 * np.sum(l_rho[0:(T - 1)]))
    return ess


def effective_sample_size(m_sample):
    """
    ## calculates ESS for a matrix of samples
    """
    try:
        n_sample, n_params = m_sample.shape
    except IndexError:
        IndexError('There must be at least one parameter')
    assert n_sample > 1

    ess = np.zeros(n_params)
    for i in range(0, n_params):
        ess[i] = ess_single_param(m_sample[:, i])
    return ess

