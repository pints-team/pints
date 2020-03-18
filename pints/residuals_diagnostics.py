#
# Functions for analysing the residuals and evaluating noise models
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
import math
import numpy as np
import scipy.special


def plot_residuals_autocorrelation(parameters,
                                   problem,
                                   max_lag=10,
                                   thinning=None,
                                   significance_level=0.05,
                                   posterior_interval=0.95):
    r"""
    Generate an autocorrelation plot of the residuals.

    This function can be used to analyse the results of either optimisation or
    MCMC Bayesian inference. When multiple samples of the residuals are present
    (corresponding to multiple MCMC samples), the plot illustrates the
    distribution of autocorrelations across the MCMC samples. At each lag,
    a point is drawn at the median autocorrelation, and a line is drawn giving
    the percentile range of the posterior interval specified as an argument (by
    default, the 2.5th to the 97.5th percentile).

    When multiple outputs are present, one residuals plot will be generated for
    each output.

    When a significance level is provided, confidence bounds for the sample
    autocorrelations under the assumption of IID residuals are drawn on the
    plot. Many of the observed residuals autocorrelations falling outside these
    bounds could imply evidence against the residuals being IID.

    Under the assumption that the residuals of length :math:`n` are IID with
    mean 0 and variance :math:`\sigma^2`, for large :math:`n` the residuals
    sample autocorrelations are approximately IID Normal(mean=0, variance=1/n).
    This result is proved in [1]_ (see Theorem 7.2.2 and Example 7.2.1).
    Therefore, the confidence bounds can be calculated by :math:`\pm z^*
    n^{-1/2}` for the appropriate critical value :math:`z^*`.

    This function returns a ``matplotlib`` figure.

    Parameters
    ----------
    parameters
        The parameter values with shape ``(n_samples, n_parameters)``. When
        passing a single best fit parameter vector, ``n_samples`` will be 1.
    problem
        The problem given by a :class:`pints.SingleOutputProblem` or
        :class:`pints.MultiOutputProblem`, with ``n_parameters`` greater than
        or equal to the ``n_parameters`` of the ``parameters``. Extra
        parameters not found in the problem are ignored.
    max_lag
        Optional int value (default 10). The highest lag to plot.
    thinning
        Optional int value (greater than zero). If thinning is set to ``n``,
        only every nth sample in parameters will be used. If set to ``None``
        (default), some thinning will be applied so that about 200 samples will
        be used.
    significance_level
        ``None`` or float value (default 0.05). When a significance level is
        provided, dashed lines for the confidence interval corresponding to
        that significance level are drawn on the plot. When ``None``, no lines
        are drawn.
    posterior_interval
        Float value (default 0.95). When multiple samples of the parameter
        values are provided, this gives the size of the credible region of the
        posterior to plot.

    References
    ----------
    .. [1] Brockwell, P. J., & Davis, R. A. (1991). Time series: Theory and
           methods (2nd ed.). New York: Springer.
    """
    import matplotlib.pyplot as plt

    # Get the number of problem outputs
    n_outputs = problem.n_outputs()

    # Get the matrix of residuals values
    residuals = calculate_residuals(parameters, problem, thinning=thinning)

    # Get the number of samples
    n_samples = residuals.shape[0]

    # If there are multiple samples of the parameters, calculate the
    # percentiles of the posterior to plot
    if n_samples > 1:
        if posterior_interval > 1 or posterior_interval < 0:
            raise ValueError('posterior interval must fall between 0 and 1')
        upper_pctle = (0.5 + posterior_interval / 2) * 100
        lower_pctle = (0.5 - posterior_interval / 2) * 100

    # Find the location of the confidence interval dotted lines
    if significance_level is not None:
        if significance_level > 1 or significance_level < 0:
            raise ValueError('significance level must fall between 0 and '
                             '1')
        threshold = scipy.special.ndtri(1 - significance_level / 2)
        threshold /= math.sqrt(residuals.shape[2])

    # Set up one axes for each output
    fig, axes = plt.subplots(n_outputs,
                             1,
                             sharex=True,
                             figsize=(6, 4 * n_outputs))

    # If there is only a single axes, place it in a list anyway.
    if n_outputs == 1:
        axes = [axes]

    # Iterate over each problem output. Each output gets its own residuals
    # autocorrelation plot
    for output_idx in range(n_outputs):

        if n_samples == 1:
            c = acorr(residuals[0, output_idx, :], max_lag)
            median_acorr = c[max_lag:]
            yerr = None

        else:
            # In this case, there are multiple samples, and a point plot of the
            # distribution of autocorrelations is desired. Start by
            # instantiating an empty array to hold autocorrelations for each
            # sample.
            cs = np.zeros((n_samples, max_lag + 1))

            # For each residual vector, get the autocorrelations
            for sample_idx in range(n_samples):
                c = acorr(residuals[sample_idx, output_idx, :], max_lag)
                cs[sample_idx, :] = c[max_lag:]

            # Calculate the necessary percentiles of the sample of
            # autocorrelations
            median_acorr = np.median(cs, axis=0)
            lower_acorr = np.percentile(cs, lower_pctle, axis=0)
            upper_acorr = np.percentile(cs, upper_pctle, axis=0)

            # Calculate the length of each bar in the point plot
            yerr = np.vstack((median_acorr - lower_acorr,
                              upper_acorr - median_acorr))

        # Plot the autocorrelation points and distributions. matplotlib
        # errorbar is used to handle the distribution lines
        axes[output_idx].errorbar(np.arange(0, max_lag + 1),
                                  median_acorr,
                                  yerr=yerr,
                                  color='red',
                                  fmt='o-')

        # Draw the dashed lines showing the confidence interval
        if significance_level is not None:
            axes[output_idx].axhline(threshold, ls='--', c='k', zorder=-10)
            axes[output_idx].axhline(-threshold, ls='--', c='k', zorder=-10)

        # Draw a horizontal line at 0 autocorrelation
        axes[output_idx].axhline(0, color='C0', zorder=-10)

        # Add y-label and adjust limits
        axes[output_idx].set_ylabel('Output %d\nresiduals autocorrelation'
                                    % (output_idx + 1))
        axes[output_idx].set_xlim(-0.5, max_lag + 0.5)

    # Add x-label (common to all outputs)
    axes[-1].set_xlabel('Lag')

    return fig


def plot_residuals_vs_output(parameters, problem, thinning=None):
    """Draw a plot of the magnitude of residuals versus the solution output.

    This plot is useful to detect any dependence between the error model and
    the magnitude of the solution. For example, it may help to detect
    multiplicative Gaussian noise, in which the standard deviation of the error
    scales with the output.

    When multiple samples of the parameters are provided (from an MCMC chain),
    the residuals are calculated and plotted relative to the posterior median
    of the solution outputs.

    This function returns a ``matplotlib`` figure.

    Parameters
    ----------
    ``parameters``
        The parameter values with shape ``(n_samples, n_parameters)``. When
        passing a single best fit parameter vector, ``n_samples`` will be 1.
    ``problem``
        The problem given by a :class:`pints.SingleOutputProblem` or
        :class:`pints.MultiOutputProblem`, with ``n_parameters`` greater than
        or equal to the ``n_parameters`` of the ``parameters``. Extra
        parameters not found in the problem are ignored.
    ``thinning``
        Optional, integer value (greater than zero). If thinning is set to
        ``n``, only every nth sample in parameters will be used. If set to
        ``None`` (default), some thinning will be applied so that about 200
        samples will be used.
    """
    import matplotlib.pyplot as plt

    # Make sure that the parameters argument has the correct shape
    try:
        n_samples, n_params = parameters.shape
    except ValueError:
        raise ValueError('`parameters` must be of shape (n_samples,'
                         + ' n_parameters).')

    n_outputs = problem.n_outputs()

    # Get the thinning level
    if thinning is None:
        thinning = max(1, int(n_samples / 200))
    else:
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError(
                'Thinning rate must be `None` or an integer greater than'
                ' zero.')

    # Solve the model for each parameter
    predicted_values = []
    for params in parameters[::thinning, :n_params]:
        predicted_values.append(problem.evaluate(params))

    # Get the posterior median solution
    posterior_median_values = np.median(predicted_values, axis=0)

    # Calculate the residuals relative to this posterior median
    residuals = problem.values() - posterior_median_values

    # If there is only one output, add the output dimension manually
    if n_outputs == 1:
        residuals = residuals[np.newaxis, :]
        posterior_median_values = posterior_median_values[np.newaxis, :]

    # Set up one axes for each output
    fig, axes = plt.subplots(n_outputs,
                             1,
                             figsize=(6, 4 * n_outputs))

    # If there is only a single axes, place it in a list anyway
    if n_outputs == 1:
        axes = [axes]

    # Plot the calculated residuals
    for output_idx in range(n_outputs):
        ax = axes[output_idx]
        ax.scatter(posterior_median_values[output_idx],
                   np.abs(residuals[output_idx]),
                   alpha=0.4)
        ax.set_xlabel('solution magnitude')
        ax.set_ylabel('residuals (absolute value)')

    return fig


def acorr(x, max_lag):
    """
    Calculate the normalised autocorrelation for a given data series.

    This function uses the same procedure as ``matplotlib.pyplot.acorr``, but
    it just calculates the autocorrelation without plotting anything.

    Returns the autocorrelation as a numpy array.

    Parameters
    ----------
    x
        A 1d numpy array containing the time series for which to calculate
        autocorrelation.
    max_lag
        An int specifying the highest lag to consider.
    """
    c = np.correlate(x, x, mode='full')

    # Normalise
    c /= np.dot(x, x)

    # Truncate at max_lag in each direction
    T = len(x)
    c = c[T - 1 - max_lag:T + max_lag]

    return c


def calculate_residuals(parameters, problem, thinning=None):
    """
    Calculate the residuals (difference between actual data and the fit).

    Either a single set of parameters or a chain of MCMC samples can be
    provided.

    The residuals are returned as a 3-dimensional numpy array with shape
    ``(n_samples, n_outputs, n_times)``.

    Parameters
    ----------
    parameters
        The parameter values with shape ``(n_samples, n_parameters)``. When
        passing a single best fit parameter vector, ``n_samples`` will be 1.
    problem
        The problem given by a :class:`pints.SingleOutputProblem` or
        :class:`pints.MultiOutputProblem`, with ``n_parameters`` greater than
        or equal to the ``n_parameters`` of the ``parameters``. Extra
        parameters not found in the problem are ignored.
    thinning
        Optional, integer value (greater than zero). If thinning is set to
        ``n``, only every nth sample in parameters will be used. If set to
        ``None`` (default), some thinning will be applied so that about 200
        samples will be used.
    """
    # Make sure that the parameters argument has the correct shape
    try:
        n_samples, n_params = parameters.shape
    except ValueError:
        raise ValueError('`parameters` must be of shape (n_samples,'
                         + ' n_parameters).')

    # Get the number of problem parameters
    n_params = problem.n_parameters()

    # Get the thinning level
    if thinning is None:
        thinning = max(1, int(n_samples / 200))
    else:
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError(
                'Thinning rate must be `None` or an integer greater than'
                ' zero.')

    # Solve the model for each parameter setting
    i = 0
    predicted_values = []
    for params in parameters[::thinning, :n_params]:
        predicted_values.append(problem.evaluate(params))
        i += 1
    predicted_values = np.array(predicted_values)

    # Calculate the residuals
    residuals = problem.values() - predicted_values

    # Arrange the residuals into the correct format
    # (n_samples, n_outputs, n_times)
    if residuals.ndim == 2:
        residuals = residuals[:, np.newaxis, :]
    else:
        residuals = np.swapaxes(residuals, 2, 1)

    return residuals
