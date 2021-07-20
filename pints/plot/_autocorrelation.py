#
# Plots autocorrelation in a chain
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np


def autocorrelation(samples, max_lags=100, parameter_names=None):
    """
    Creates and returns an autocorrelation plot for a given markov chain or
    list of `samples`.

    Returns a ``matplotlib`` figure object and axes handle.

    Parameters
    ----------
    samples
        A list of samples, with shape ``(n_samples, n_parameters)``, where
        ``n_samples`` is the number of samples in the list and ``n_parameters``
        is the number of parameters.
    max_lags
        The maximum autocorrelation lag to plot.
    parameter_names
        A list of parameter names, which will be displayed in the legend of the
        autocorrelation subplots. If no names are provided, the parameters are
        enumerated.
    """
    import matplotlib.pyplot as plt

    # Check samples size
    try:
        n_sample, n_param = samples.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample,'
                         + ' n_parameters).')

    # Check parameter names
    if parameter_names is None:
        parameter_names = ['Parameter' + str(i + 1) for i in range(n_param)]
    elif len(parameter_names) != n_param:
        raise ValueError(
            'Length of `parameter_names` must be same as number of'
            ' parameters.')

    fig, axes = plt.subplots(n_param, 1, sharex=True, figsize=(6, 2 * n_param))
    if n_param == 1:
        axes = np.asarray([axes], dtype=object)
    for i in range(n_param):
        axes[i].acorr(samples[:, i] - np.mean(samples[:, i]), maxlags=max_lags)
        axes[i].set_xlim(-0.5, max_lags + 0.5)
        axes[i].legend([parameter_names[i]], loc='upper right')

    # Add x-label to final plot only
    axes[i].set_xlabel('Lag')

    # Add vertical y-label to middle plot
    # fig.text(0.04, 0.5, 'Autocorrelation', va='center', rotation='vertical')
    axes[int(i / 2)].set_ylabel('Autocorrelation')

    plt.tight_layout()
    return fig, axes

