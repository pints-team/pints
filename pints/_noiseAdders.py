#
# A number of functions which can be used to add various types of noise to exact simulations
# to create fake data
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
import numpy as np
import pints


def addIndependentNoise(lValues,sigma):
    """ adds independent Gaussian noise - iid N(0,sigma) - to a list of simulated values."""
    return lValues + np.random.normal(0, sigma, lValues.shape)

def AR1(rho, sigma, T):
    """ creates an autoregressive order 1 series vX[t+t] ~ rho * vX[t-1] + e(t)
    where e(t) ~ N(0,sqrt(sigma^2 / (1 - rho^2))) with vX[0] = 0, of length T. This choice
    of parameterisation ensures that the AR1 process has a mean of 0 and a std
    deviation of sigma."""
    vX = np.zeros(T)
    for t in range(1,T):
        vX[t] = rho * vX[t-1] + np.random.normal(0,sigma * np.sqrt(1 - rho**2))
    return vX

def addAR1Noise(lValues,rho,sigma):
    """ adds autoregressive order 1 noise to data. i.e. the errors follow a,
    e(t) ~ rho * e(t-1) + v(t), where v(t) ~ iid N(0,sigma)."""    
    return lValues + AR1(rho,sigma,lValues.shape)

def AR1_unity(rho, sigma, T):
    """ creates an autoregressive order 1 series vX[t+t] ~ (1 - rho) + rho * vT[t-1] + e(t)
    where e(t) ~ N(0,sqrt(sigma^2 / (1 - rho^2))) with vX[0] = 0, of length T. This choice
    of parameterisation ensures that the AR1 process has mean 1 and a std deviation of sigma."""
    vX = np.zeros(T)
    vX[0] = 1
    for t in range(1,T):
        vX[t] = (1 - rho) + rho * vX[t-1] + np.random.normal(0,sigma * np.sqrt(1 - rho**2))
    return vX

def multiplyAR1Noise(lValues,rho,sigma):
    """ multiplies signal by a noise process that follows an autoregressive order 1 process
    of mean 1."""
    return lValues * AR1_unity(rho,sigma,lValues.shape)