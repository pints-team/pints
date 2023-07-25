#
# Plots pairwise scatterplots for all parameter pairs
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import warnings

from distutils.version import LooseVersion

import numpy as np
import scipy.stats as stats


def pairwise(samples,
             kde=False,
             heatmap=False,
             opacity=None,
             n_percentiles=None,
             parameter_names=None,
             ref_parameters=None):
    """
    Takes a markov chain or list of ``samples`` and creates a set of pairwise
    scatterplots for all parameters (p1 versus p2, p1 versus p3, p2 versus p3,
    etc.).

    The returned plot is in a 'matrix' form, with histograms of each individual
    parameter on the diagonal, and scatter plots of parameters ``i`` and ``j``
    on each entry ``(i, j)`` below the diagonal.

    Returns a ``matplotlib`` figure object and axes handle.

    Parameters
    ----------
    samples
        A list of samples, with shape ``(n_samples, n_parameters)``, where
        ``n_samples`` is the number of samples in the list and ``n_parameters``
        is the number of parameters.
    kde
        Set to ``True`` to use kernel-density estimation for the
        histograms and scatter plots. Cannot use together with ``heatmap``.
    heatmap
        Set to ``True`` to plot heatmap for the pairwise plots.
        Cannot be used together with ``kde``.
    opacity
        This value can be used to manually set the opacity of the
        points in the scatter plots (when ``kde=False`` and ``heatmap=False``
        only).
    n_percentiles
        Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.
    parameter_names
        A list of parameter names, which will be displayed on the axes of the
        subplots. If no names are provided, the parameters are enumerated.
    ref_parameters
        A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # Check matplotlib version
    use_old_matplotlib = LooseVersion(matplotlib.__version__) \
        < LooseVersion("2.2")

    # Check options kde and heatmap
    if kde and heatmap:
        raise ValueError('Cannot use `kde` and `heatmap` together.')

    # Check samples size
    try:
        n_sample, n_param = samples.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample,'
                         + ' n_parameters).')

    # Check number of parameters
    if n_param < 2:
        raise ValueError('Number of parameters must be larger than 2.')

    # Check parameter names
    if parameter_names is None:
        parameter_names = ['Parameter' + str(i + 1) for i in range(n_param)]
    elif len(parameter_names) != n_param:
        raise ValueError(
            'Length of `parameter_names` must be same as number of'
            ' parameters.')

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters.')

    # Create figure
    fig_size = (3 * n_param, 3 * n_param)
    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

    bins = 25
    for i in range(n_param):
        for j in range(n_param):
            if i == j:

                # Diagonal: Plot a histogram
                if n_percentiles is None:
                    xmin, xmax = np.min(samples[:, i]), np.max(samples[:, i])
                else:
                    xmin = np.percentile(samples[:, i],
                                         50 - n_percentiles / 2.)
                    xmax = np.percentile(samples[:, i],
                                         50 + n_percentiles / 2.)
                xbins = np.linspace(xmin, xmax, bins)
                axes[i, j].set_xlim(xmin, xmax)
                if use_old_matplotlib:  # pragma: no cover
                    axes[i, j].hist(samples[:, i], bins=xbins, normed=True)
                else:
                    axes[i, j].hist(samples[:, i], bins=xbins, density=True)

                # Add kde plot
                if kde:
                    x = np.linspace(xmin, xmax, 100)
                    axes[i, j].plot(x, stats.gaussian_kde(samples[:, i])(x))

                # Add reference parameters if given
                if ref_parameters is not None:
                    ymin_tv, ymax_tv = axes[i, j].get_ylim()
                    axes[i, j].plot(
                        [ref_parameters[i], ref_parameters[i]],
                        [0.0, ymax_tv],
                        '--', c='k')

            elif i < j:
                # Top-right: no plot
                axes[i, j].axis('off')

            else:
                # Lower-left: Plot the samples as density map
                if n_percentiles is None:
                    xmin, xmax = np.min(samples[:, j]), np.max(samples[:, j])
                    ymin, ymax = np.min(samples[:, i]), np.max(samples[:, i])
                else:
                    xmin = np.percentile(samples[:, j],
                                         50 - n_percentiles / 2.)
                    xmax = np.percentile(samples[:, j],
                                         50 + n_percentiles / 2.)
                    ymin = np.percentile(samples[:, i],
                                         50 - n_percentiles / 2.)
                    ymax = np.percentile(samples[:, i],
                                         50 + n_percentiles / 2.)
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].set_ylim(ymin, ymax)

                if not kde and not heatmap:
                    # Create scatter plot

                    # Determine point opacity
                    num_points = len(samples[:, i])
                    if opacity is None:
                        if num_points < 10:
                            opacity = 1.0
                        else:
                            opacity = 1.0 / np.log10(num_points)

                    # Scatter points
                    axes[i, j].scatter(
                        samples[:, j], samples[:, i], alpha=opacity, s=0.1)
                elif kde:
                    # Create a KDE-based plot

                    # Plot values
                    values = np.vstack([samples[:, j], samples[:, i]])
                    axes[i, j].imshow(
                        np.rot90(values), cmap=plt.cm.Blues,
                        extent=[xmin, xmax, ymin, ymax])

                    # Create grid
                    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])

                    # Get kernel density estimate and plot contours
                    kernel = stats.gaussian_kde(values)
                    f = np.reshape(kernel(positions).T, xx.shape)
                    axes[i, j].contourf(xx, yy, f, cmap='Blues')
                    axes[i, j].contour(xx, yy, f, colors='k')

                    # Force equal aspect ratio
                    # Matplotlib raises a warning here (on 2.7 at least)
                    # We can't do anything about it, so no other option than
                    # to suppress it at this stage...
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UnicodeWarning)
                        axes[i, j].set_aspect((xmax - xmin) / (ymax - ymin))
                elif heatmap:
                    # Create a heatmap-based plot

                    # Create bins
                    xbins = np.linspace(xmin, xmax, bins)
                    ybins = np.linspace(ymin, ymax, bins)

                    # Plot heatmap
                    axes[i, j].hist2d(samples[:, j], samples[:, i],
                                      bins=(xbins, ybins), cmap=plt.cm.Blues)

                    # Force equal aspect ratio
                    # Matplotlib raises a warning here (on 2.7 at least)
                    # We can't do anything about it, so no other option than
                    # to suppress it at this stage...
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UnicodeWarning)
                        axes[i, j].set_aspect((xmax - xmin) / (ymax - ymin))

                # Add reference parameters if given
                if ref_parameters is not None:
                    axes[i, j].plot(
                        [ref_parameters[j], ref_parameters[j]],
                        [ymin, ymax],
                        '--', c='k')
                    axes[i, j].plot(
                        [xmin, xmax],
                        [ref_parameters[i], ref_parameters[i]],
                        '--', c='k')

            # Set tick labels
            if i < n_param - 1:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            else:
                # Rotate the x tick labels to fit in the plot
                for tl in axes[i, j].get_xticklabels():
                    tl.set_rotation(45)

            if j > 0:
                # Only show y tick labels for the first column
                axes[i, j].set_yticklabels([])

        # Set axis labels
        axes[-1, i].set_xlabel(parameter_names[i])
        if i == 0:
            # The first one is not a parameter
            axes[i, 0].set_ylabel('Frequency')
        else:
            axes[i, 0].set_ylabel(parameter_names[i])

    return fig, axes

