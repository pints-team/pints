#
# A number of functions which can be used to generate different types of noise,
# which can then be added to model output to simulate experimental data.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#


def independent(sigma, shape):
    r"""
    Generates independent Gaussian noise iid :math:`\mathcal{N}(0,\sigma)`.

    Returns an array of shape ``shape`` containing the generated noise.

    Parameters
    ----------
    sigma
        The standard deviation of the noise. Must be zero or greater.
    shape
        A tuple (or sequence) defining the shape of the generated noise array.

    Example
    -------
    ::

        values = model.simulate(parameters, times)
        noisy_values = values + noise.independent(5, values.shape)

    """
    import numpy as np

    # Don't test sigma/shape: handled by numpy for higher-dimensions etc.!
    return np.random.normal(0, sigma, shape)


def ar1(rho, sigma, n):
    r"""
    Generates first-order autoregressive (AR1) noise that can be added to a
    vector of simulated data.

    The generated noise follows the distribution

    .. math::
        e(t) = \rho  e(t - 1) + v(t),

    where :math:`v(t) \stackrel{\text{iid}}{\sim }\mathcal{N}(0, \sigma
    \sqrt{1 - \rho ^2})`.

    Returns an array of length ``n`` containing the generated noise.

    Parameters
    ----------
    rho
        Determines the magnitude of the noise :math:`\rho` (see above). Must
        be less than 1.
    sigma
        The marginal standard deviation :math:`\sigma` of ``e(t)`` (see above).
        Must be greater than zero.
    n
        The length of the signal. (Only single time-series are supported.)

    Example
    -------
    ::

        values = model.simulate(parameters, times)
        noisy_values = values + noise.ar1(0.9, 5, len(values))

    """
    import numpy as np

    if abs(rho) >= 1:
        raise ValueError(
            'Magnitude of rho must be less than 1 (otherwise the process'
            ' is non-stationary).')
    if sigma <= 0:
        raise ValueError('Standard deviation must be positive.')

    n = int(n)
    if n < 1:
        raise ValueError('Number of values to generate must be at least one.')

    # Generate noise
    s = sigma * np.sqrt(1 - rho**2)
    v = np.random.normal(0, s, n)
    v[0] = np.random.rand()
    for t in range(1, n):
        v[t] += rho * v[t - 1]
    return v


def arma11(rho, theta, sigma, n):
    r"""
    Generates an ARMA(1,1) error process of the form:

    .. math::
        e(t) = (1 - \rho) + \rho * e(t - 1) + v(t) + \theta * v(t-1),

    where :math:`v(t) \stackrel{\text{iid}}{\sim }\mathcal{N}(0, \sigma ')`,
    and

    .. math::
        \sigma ' = \sigma \sqrt{\frac{1 - \rho ^ 2}{1 + 2 \theta  \rho +
        \theta ^ 2}}.
    """
    import numpy as np

    if abs(rho) >= 1:
        raise ValueError(
            'Magnitude of rho must be less than 1 (otherwise the process'
            ' is non-stationary).')
    if abs(theta) >= 1.0:
        raise ValueError('Absolute value of theta must be less than 1 ' +
                         'so that the process is invertible.')
    if sigma <= 0:
        raise ValueError('Standard deviation must be positive.')

    n = int(n)
    if n < 1:
        raise ValueError('Number of values to generate must be at least one.')

    # Generate noise
    s = sigma * np.sqrt((1 - rho**2) / (1 + 2 * theta * rho + theta**2))
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

    Returns an array of length ``n`` containing the generated noise.

    Parameters
    ----------
    rho
        Determines the magnitude of the noise (see :meth:`ar1`). Must be less
        than or equal to 1.
    sigma
        The marginal standard deviation of ``e(t)`` (see :meth:`ar`).
        Must be greater than 0.
    n : int
        The length of the signal. (Only single time-series are supported.)

    Example
    -------
    ::

        values = model.simulate(parameters, times)
        noisy_values = values * noise.ar1_unity(0.5, 0.8, len(values))

    """
    import numpy as np

    if abs(rho) >= 1:
        raise ValueError(
            'Magnitude of rho must be less than 1 (otherwise the process is'
            ' non-stationary).')
    if sigma <= 0:
        raise ValueError('Standard deviation must be positive.')

    n = int(n)
    if n < 1:
        raise ValueError('Number of values to generate must be at least one.')

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

    and
    ``sigma' = sigma * sqrt((1 - rho^2) / (1 + 2 * theta * rho + theta^2))``.

    Returns an array of length ``n`` containing the generated noise.

    Parameters
    ----------
    rho
        Determines the long-run persistence of the noise (see :meth:`ar1`).
        Must be less than 1.
    theta
        Contributes to first order autocorrelation of noise. Must be less
        than 1.
    sigma
        The marginal standard deviation of ``e(t)`` (see :meth:`ar`).
        Must be greater than 0.
    n : int
        The length of the signal. (Only single time-series are supported.)

    Example
    -------
    ::

        values = model.simulate(parameters, times)
        noisy_values = values * noise.ar1_unity(0.5, 0.8, len(values))

    """
    import numpy as np

    if abs(rho) >= 1:
        raise ValueError(
            'Magnitude of rho must be less than 1 (otherwise the process is'
            ' explosive).')
    if abs(theta) >= 1.0:
        raise ValueError('Absolute value of theta must be less than 1 ' +
                         'so that the process is invertible.')
    if sigma <= 0:
        raise ValueError('Standard deviation must be positive.')
    n = int(n)
    if n < 1:
        raise ValueError('Number of values to generate must be at least one.')

    # Generate noise
    s = sigma * np.sqrt((1 - rho**2) / (1 + 2 * theta * rho + theta**2))
    v = np.random.normal(0, s, n + 1)
    e = np.zeros(n)
    e[0] = v[1]
    for i in range(1, n):
        e[i] = (1 - rho) + rho * e[i - 1] + v[i] + theta * v[i - 1]
    return e


def multiplicative_gaussian(eta, sigma, f):
    r"""
    Generates multiplicative Gaussian noise for a single output.

    With multiplicative noise, the measurement error scales with the magnitude
    of the output. Given a model taking the form,

    .. math::
        X(t) = f(t; \theta) + \epsilon(t)

    multiplicative Gaussian noise models the noise term as:

    .. math::
        \epsilon(t) = f(t; \theta)^\eta v(t)

    where v(t) is iid Gaussian:

    .. math::
        v(t) \stackrel{\text{ iid }}{\sim} \mathcal{N}(0, \sigma)

    The output magnitudes ``f`` are required as an input to this function. The
    noise terms are returned in an array of the same shape as ``f``.

    Parameters
    ----------
    ``eta``
        The exponential power controlling the rate at which the noise scales
        with the output. The argument must be either a float (for single-output
        or multi-output noise) or an array_like of floats (for multi-output
        noise only, with one value for each output).
    ``sigma``
        The baseline standard deviation of the noise (must be greater than
        zero). The argument must be either a float (for single-output
        or multi-output noise) or an array_like of floats (for multi-output
        noise only, with one value for each output).
    ``f``
        A NumPy array giving the time-series for the output over time. For
        multiple outputs, the array should have shape ``(n_outputs, n_times)``.
    """
    import numpy as np

    f = np.array(f)

    # Check the dimensions of the inputs
    if f.ndim > 2:
        raise ValueError('f must have be of shape (n_outputs, n_times).')

    if f.ndim == 2:
        n_outputs = f.shape[0]
    else:
        n_outputs = 1

    if not np.isscalar(eta):
        eta = np.array(eta)
        if eta.ndim > 1 or (eta.shape[0] != 1 and eta.shape[0] != n_outputs):
            raise ValueError('eta must be a scalar or of shape (n_outputs,).')

        # Reshape eta so that it broadcasts to f correctly
        eta = eta[:, np.newaxis]

    if not np.isscalar(sigma):
        sigma = np.array(sigma)
        if sigma.ndim > 1 or (sigma.shape[0] != 1 and
                              sigma.shape[0] != n_outputs):
            raise ValueError('sigma must be a scalar or of shape '
                             '(n_outputs,).')

        # Reshape sigma so that it broadcasts to f correctly
        sigma = sigma[:, np.newaxis]

    # Check the values of the inputs
    if np.isscalar(sigma):
        if sigma <= 0:
            raise ValueError('Standard deviation must be greater than zero.')
    else:
        if np.any(sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')

    e = np.random.normal(0, sigma, f.shape)
    return f ** eta * e
