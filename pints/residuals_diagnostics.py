#
# Functions for analyzing the residuals and evaluating noise models
#
# This file is part of PINTS
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing informating, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
import math


def plot_residuals_autocorrelation(parameters,
                                   problem,
                                   max_lag=10,
                                   thinning=None,
                                   show_confidence=False,
                                   significance_level=0.05,
                                   posterior_interval=0.95):
    """
    Generate an autocorrelation plot of the residuals.

    This function can be used to analyze the results of either optimisation or
    MCMC Bayesian inference. When multiple samples of the residuals are present
    (corresponding to multiple MCMC samples), the plot can illustrate the
    distribution of autocorrelations across the MCMC samples. At each lag,
    a dot is drawn at the median autocorrelation, and a line is drawn giving
    the extent of the posterior interval specified as an argument (by default,
    the 2.5th to the 97.5th percentile).

    When multiple outputs are present, one residuals plot will be generated for
    each output.

    Returns a ``matplotlib`` figure.

    Parameters
    ----------
    parameters
        The parameter values with shape ``(n_samples, n_parameters)``. When
        passing a single best fit parameter vector, `n_samples` will be 1.
    problem
        The problem given by a :class:``pints.SingleOutputProblem`` or
        :class:``pints.MultiOutputProblem``, with `n_parameters` greater than
        or equal to the ``n_parameters`` of the `parameters`. Extra parameters
        not found in the problem are ignored.
    max_lag
        Optional int value (default 10). The highest lag to plot.
    thinning
        Optional, integer value (greater than zero). If thinning is set to `n`,
        only every nth sample in parameters will be used. If set to None
        (default), some thinning will be applied so that about 200 samples will
        be used.
    significance_level
        None or float value (default 0.05). When a significance level is
        provided, dashed lines for the confidence interval corresponding to
        that significance level are drawn on the plot. When None, no lines are
        drawn.
    posterior_interval
        Float value (default 0.95). When multiple samples of the parameter
        values are provided, this gives the size of the credible region of the
        posterior to plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.special

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


def acorr(x, max_lag):
    """
    Calculate the normalized autocorrelation for a given data series.

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
    import numpy as np
    c = np.correlate(x, x, mode='full')

    # Normalize
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
        passing a single best fit parameter vector, `n_samples` will be 1.
    problem
        The problem given by a :class:``pints.SingleOutputProblem`` or
        :class:``pints.MultiOutputProblem``, with `n_parameters` greater than
        or equal to the ``n_parameters`` of the `parameters`. Extra parameters
        not found in the problem are ignored.
    thinning
        Optional, integer value (greater than zero). If thinning is set to `n`,
        only every nth sample in parameters will be used. If set to None
        (default), some thinning will be applied so that about 200 samples will
        be used.
    """
    import numpy as np

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
