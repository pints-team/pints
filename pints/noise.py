#
# A number of functions which can be used to generate different types of noise,
# which can then be added to model output to simulate experimental data.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import numpy as np


def independent(sigma, shape):
    """
    Generates independent Gaussian noise (``iid N(0,sigma)``).

    Arguments:

    ``sigma``
        The standard deviation of the noise. Must be zero or greater.
    ``shape``
        A tuple (or sequence) defining the shape of the generated noise array.

    Returns an array of shape ``shape`` containing the generated noise.

    Example::

        values = model.simulate(parameters, times)
        noisy_values = values + noise.independent(5, values.shape)

    """

    # Don't test sigma/shape: handled by numpy for higher-dimensions etc.!
    return np.random.normal(0, sigma, shape)


def ar1(rho, sigma, n):
    """
    Generates first-order autoregressive (AR1) noise that can be added to a
    vector of simulated data.

    The generated noise follows the distribution
    ``e(t) ~ rho * e(t - 1) + v(t)``, where ``v(t) ~ iid N(0, sigma)``.

    Arguments:

    ``rho``
        Determines the magnitude of the noise (see above). Must be less than or
        equal to 1.
    ``sigma``
        The standard deviation of ``v(t)`` (see above). Must be greater than
        zero.
    ``n``
        The length of the signal. (Only single time-series are supported.)

    Returns an array of length ``n`` containing the generated noise.

    Example::

        values = model.simulate(parameters, times)
        noisy_values = values + noise.ar1(5, len(values))

    """
    if np.absolute(rho) > 1:
        raise ValueError(
            'Magnitude of rho cannot be greater than 1 (otherwise the process'
            ' is explosive).')
    if sigma < 0:
        raise ValueError('Standard deviation cannot be negative.')

    n = int(n)
    if n < 0:
        raise ValueError('Length of signal cannot be negative.')
    elif n == 0:
        return np.array([])

    # Generate noise
    s = sigma * np.sqrt(1 - rho**2)
    if s == 0:
        v = np.zeros(n)
    else:
        v = np.random.normal(0, s, n)
    v[0] = np.random.rand()
    for t in range(1, n):
        v[t] += rho * v[t - 1]
    return v


def ar1_unity(rho, sigma, n):
    """
    Generates noise following an autoregressive order 1 process of mean 1, that
    a vector of simulated data can be multiplied with.

    ``rho``
        Determines the magnitude of the noise (see :meth:`ar1`). Must be less
        than or equal to 1.
    ``sigma``
        The standard deviation of ``v(t)`` (see :meth:`ar`). Must be zero or
        greater.
    ``n``
        The length of the signal. (Only single time-series are supported.)

    Returns an array of length ``n`` containing the generated noise.

    Example::

        values = model.simulate(parameters, times)
        noisy_values = values * noise.ar1_unity(0.5, 1, len(values))

    """
    if np.absolute(rho) > 1:
        raise ValueError(
            'Rho must be less than 1 in magnitude (otherwise the process is'
            ' explosive).')
    if sigma < 0:
        raise ValueError('Standard deviation cannot be negative.')

    n = int(n)
    if n < 1:
        raise ValueError('Must supply at least one value.')

    # Generate noise
    v = np.random.normal(0, sigma * np.sqrt(1 - rho**2), n + 1)
    v[0] = 1
    for t in range(1, n + 1):
        v[t] += (1 - rho) + rho * v[t - 1]
    return v[1:]

