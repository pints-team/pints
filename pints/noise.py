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
    ``e(t) = rho * e(t - 1) + v(t)``,

    where ``v(t) ~ iid N(0, sigma * sqrt(1 - rho^2))``.

    Arguments:

    ``rho``
        Determines the magnitude of the noise (see above). Must be less than 1.
    ``sigma``
        The marginal standard deviation of ``e(t)`` (see above).
        Must be greater than zero.
    ``n``
        The length of the signal. (Only single time-series are supported.)

    Returns an array of length ``n`` containing the generated noise.

    Example::

        values = model.simulate(parameters, times)
        noisy_values = values + noise.ar1(0.9, 5, len(values))

    """
    if np.absolute(rho) >= 1:
        raise ValueError(
            'Magnitude of rho must be less than 1 (otherwise the process'
            ' is non-stationary).')
    if sigma <= 0:
        raise ValueError('Standard deviation must be positive.')

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


def arma11(rho, theta, sigma, n):
    """
    Generates an ARMA(1,1) error process of the form:

    ``e(t) = (1 - rho) + rho * e(t - 1) + v(t) + theta * v[t-1]``,

    where ``v(t) ~ iid N(0, sigma')``,

    and ``sigma' = sigma * sqrt((1 - rho^2) / (1 + theta * (1 + rho)))``.
    """
    if np.absolute(rho) >= 1:
        raise ValueError(
            'Magnitude of rho must be less than 1 (otherwise the process'
            ' is non-stationary).')
    if sigma <= 0:
        raise ValueError('Standard deviation must be positive.')
    if np.abs(theta) >= 1:
        raise ValueError('theta must be less than 1 so the process is ' +
                         'invertible.')
    n = int(n)
    if n < 0:
        raise ValueError('Length of signal cannot be negative.')
    elif n == 0:
        return np.array([])

    # Generate noise
    s = sigma * np.sqrt((1 - rho**2) / (1 + theta * (1 + rho)))
    v = np.random.normal(0, s, n)
    e = np.zeros(n)
    e[0] = v[0]
    for i in range(1, n):
        e[i] = rho * e[i - 1] + v[i] + theta * v[i - 1]
    return e


def ar1_unity(rho, sigma, n):
    """
    Generates noise following an autoregressive order 1 process of mean 1, that
    a vector of simulated data can be multiplied with.

    ``rho``
        Determines the magnitude of the noise (see :meth:`ar1`). Must be less
        than or equal to 1.
    ``sigma``
        The marginal standard deviation of ``e(t)`` (see :meth:`ar`).
        Must be greater than 0.
    ``n``
        The length of the signal. (Only single time-series are supported.)

    Returns an array of length ``n`` containing the generated noise.

    Example::

        values = model.simulate(parameters, times)
        noisy_values = values * noise.ar1_unity(0.5, 0.8, len(values))

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


def arma11_unity(rho, theta, sigma, n):
    """
    Generates an ARMA(1,1) error process of the form:

    ``e(t) = (1 - rho) + rho * e(t - 1) + v(t) + theta * v[t-1]``,

    where ``v(t) ~ iid N(0, sigma')``,

    and ``sigma' = sigma * sqrt((1 - rho^2) / (1 + theta * (1 + rho)))``

    ``rho``
        Determines the long-run persistence of the noise (see :meth:`ar1`).
        Must be less than 1.
    ``theta``
        Contributes to first order autocorrelation of noise. Must be less
        than 1.
    ``sigma``
        The marginal standard deviation of ``e(t)`` (see :meth:`ar`).
        Must be greater than 0.
    ``n``
        The length of the signal. (Only single time-series are supported.)

    Returns an array of length ``n`` containing the generated noise.

    Example::

        values = model.simulate(parameters, times)
        noisy_values = values * noise.ar1_unity(0.5, 0.8, len(values))

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
