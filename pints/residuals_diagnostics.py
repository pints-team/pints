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
                                   show_confidence=False):
    """
    Plot the autocorrelation plot of the residuals.

    This function can be used to analyze the results of either optimization or
    MCMC Bayesian inference. When multiple samples of the residuals are present
    (corresponding to multiple MCMC samples), the plot will show box-whisker
    diagrams illustrating the distribution of autocorrelations across the MCMC
    samples.

    When multiple outputs are present, one residuals plot will be generated for
    each output.

    Returns a ``matplotlib`` figure.

    Parameters
    ----------
    parameters
        The parameter values with shape ``(n_samples, n_parameters)``.
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
    show_confidence
        Optional bool value (default False). If True, the 95% confidence
        interval under the hypothesis of uncorrelated Gaussian noise is drawn
        on the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.special

    # Get the number of problem outputs parameters
    n_outputs = problem.n_outputs()

    # Get the matrix of residuals values
    residuals = calculate_residuals(parameters, problem, thinning=thinning)

    # Get the number of samples
    n_samples = residuals.shape[0]

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
            # In this case, we are plotting a single autocorrelation value at
            # each lag. The default matplotlib function can be used.
            axes[output_idx].acorr(residuals[0, output_idx, :],
                                   maxlags=max_lag)

        else:
            # In this case, there are multiple samples, and a boxplot of the
            # distribution of autocorrelations is desired. Start by
            # instantiating an empty array to hold autocorrelations for each
            # sample.
            cs = np.zeros((n_samples, max_lag + 1))

            # For each residual vector, get the autocorrelations using the
            # same matplotlib function -- but dump the plot it makes.
            null_figure = plt.Figure()
            null_ax = null_figure.gca()
            for sample_idx in range(n_samples):
                _, c, _, _ = null_ax.acorr(residuals[sample_idx,
                                                     output_idx, :],
                                           maxlags=max_lag)
                cs[sample_idx, :] = c[max_lag:]

            # Draw the boxplots as well as a line connecting the medians.
            result = axes[output_idx].boxplot(cs,
                                              positions=np.arange(max_lag + 1))
            medians = [med_line.get_ydata()[0] for
                       med_line in result['medians']]
            axes[output_idx].plot(np.arange(0, max_lag + 1),
                                  medians,
                                  color='red')
            axes[output_idx].axhline(0)

        axes[output_idx].set_ylabel('Output %d\nresiduals autocorrelation'
                                    % (output_idx + 1))
        axes[output_idx].set_xlim(-0.5, max_lag + 0.5)

        if show_confidence:
            significance_level = 0.05
            threshold = scipy.special.ndtri(1 - significance_level / 2)
            threshold /= math.sqrt(residuals.shape[2])
            axes[output_idx].axhline(threshold, ls='--', c='k')
            axes[output_idx].axhline(-threshold, ls='--', c='k')

    axes[-1].set_xlabel('Lag')

    return fig


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
