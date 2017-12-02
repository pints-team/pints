#
# Functions to calculate various MCMC diagnostics
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
import pints
import numpy as np

## calculates autocorrelation function for a vector x by spectra density calculation
def autocorrelation(lx) :
    xp = lx-np.mean(lx)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:lx.size/2]/np.sum(xp**2)

## finds last positive autocorrelation, T
def autocorrelateNegative(lAutocorrelation):
    T = 1
    for a in lAutocorrelation:
        if a < 0:
            return T - 1
        T += 1
    return -1

## calculates ESS for a single parameter
def ESS_singleParam(lx):
    lRho = autocorrelation(lx)
    T = autocorrelateNegative(lRho)
    n = len(lx)
    ESS = n / (1 + 2 * np.sum(lRho[0:(T-1)]))
    return ESS

## calculates ESS for a matrix of samples
def effectiveSampleSize(mSample):
    try: 
        nSample, nParams = mSample.shape
    except IndexError:
        IndexError('There must be at least one parameter')
    assert nSample > 1
    
    ESS = np.zeros(nParams)
    for i in range(0,nParams):
        ESS[i] = ESS_singleParam(mSample[:,i])
    return ESS
    
    
    