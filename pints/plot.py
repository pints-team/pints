#
# Quick diagnostic plots.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import warnings
import numpy as np
import scipy.stats as stats


def function(f, x, lower=None, upper=None, evaluations=20):
    """
    Creates 1d plots of a :class:`LogPDF` or a :class:`ErrorMeasure` around a
    point `x` (i.e. a 1-dimensional plot in each direction).

    Arguments:

    ``f``
        A :class:`pints.LogPDF` or :class:`pints.ErrorMeasure` to plot.
    ``x``
        A point in the function's input space.
    ``lower``
        (Optional) Lower bounds for each parameter, used to specify the lower
        bounds of the plot.
    ``upper``
        (Optional) Upper bounds for each parameter, used to specify the upper
        bounds of the plot.
    ``evaluations``
        (Optional) The number of evaluations to use in each plot.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # Check function and get n_parameters
    if not (isinstance(f, pints.LogPDF) or isinstance(f, pints.ErrorMeasure)):
        raise ValueError(
            'Given function must be pints.LogPDF or pints.ErrorMeasure.')
    n_param = f.n_parameters()

    # Check point
    x = pints.vector(x)
    if len(x) != n_param:
        raise ValueError(
            'Given point `x` must have same number of parameters as function.')

    # Check boundaries
    if lower is None:
        # Guess boundaries based on point x
        lower = x * 0.95
        lower[lower == 0] = -1
    else:
        lower = pints.vector(lower)
        if len(lower) != n_param:
            raise ValueError('Lower bounds must have same number of'
                             + ' parameters as function.')
    if upper is None:
        # Guess boundaries based on point x
        upper = x * 1.05
        upper[upper == 0] = 1
    else:
        upper = pints.vector(upper)
        if len(upper) != n_param:
            raise ValueError('Upper bounds must have same number of'
                             + ' parameters as function.')

    # Check number of evaluations
    evaluations = int(evaluations)
    if evaluations < 1:
        raise ValueError('Number of evaluations must be greater than zero.')

    # Create points to plot
    xs = np.tile(x, (n_param * evaluations, 1))
    for j in range(n_param):
        i1 = j * evaluations
        i2 = i1 + evaluations
        xs[i1:i2, j] = np.linspace(lower[j], upper[j], evaluations)

    # Evaluate points
    fs = pints.evaluate(f, xs, parallel=False)

    # Create figure
    fig, axes = plt.subplots(n_param, 1, figsize=(6, 2 * n_param))
    if n_param == 1:
        axes = np.asarray([axes], dtype=object)
    for j, p in enumerate(x):
        i1 = j * evaluations
        i2 = i1 + evaluations
        axes[j].plot(xs[i1:i2, j], fs[i1:i2], c='green', label='Function')
        axes[j].axvline(p, c='blue', label='Value')
        axes[j].set_xlabel('Parameter ' + str(1 + j))
        axes[j].legend()

    plt.tight_layout()
    return fig, axes


def function_between_points(f, point_1, point_2, padding=0.25, evaluations=20):
    """
    Creates and returns a plot of a function between two points in parameter
    space.

    Arguments:

    ``f``
        A :class:`pints.LogPDF` or :class:`pints.ErrorMeasure` to plot.
    ``point_1``, ``point_2``
        Two points in parameter space. The method will find a line from
        ``point_1`` to ``point_2`` and plot ``f`` at several points along it.
    ``padding``
        (Optional) Specifies the amount of padding around the line segment
        ``[point_1, point_2]`` that will be shown in the plot.
    ``evaluations``
        (Optional) The number of evaluation along the line in parameter space.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # Check function and get n_parameters
    if not (isinstance(f, pints.LogPDF) or isinstance(f, pints.ErrorMeasure)):
        raise ValueError(
            'Given function must be pints.LogPDF or pints.ErrorMeasure.')
    n_param = f.n_parameters()

    # Check points
    point_1 = pints.vector(point_1)
    point_2 = pints.vector(point_2)
    if not (len(point_1) == len(point_2) == n_param):
        raise ValueError('Both points must have the same number of parameters'
                         + ' as the given function.')

    # Check padding
    padding = float(padding)
    if padding < 0:
        raise ValueError('Padding cannot be negative.')

    # Check evaluation
    evaluations = int(evaluations)
    if evaluations < 3:
        raise ValueError('The number of evaluations must be 3 or greater.')

    # Figure setting
    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    axes.set_xlabel('Point 1 to point 2')
    axes.set_ylabel('Function')

    # Generate some x-values near the given parameters
    s = np.linspace(-padding, 1 + padding, evaluations)

    # Direction
    r = point_2 - point_1

    # Calculate function with other parameters fixed
    x = [point_1 + sj * r for sj in s]
    y = pints.evaluate(f, x, parallel=False)

    # Plot
    axes.plot(s, y, color='green')
    axes.axvline(0, color='#1f77b4', label='Point 1')
    axes.axvline(1, color='#7f7f7f', label='Point 2')
    axes.legend()

    return fig, axes


def histogram(samples, ref_parameters=None, n_percentiles=None):
    """
    Takes one or more markov chains or lists of samples as input and creates
    and returns a plot showing histograms for each chain or list of samples.

    Arguments:

    ``samples``
        A list of lists of samples, with shape
        ``(n_lists, n_samples, n_parameters)``, where ``n_lists`` is the
        number of lists of samples, ``n_samples`` is the number of samples in
        one list and ``n_parameters`` is the number of parameters.
    ``ref_parameters``
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    ``n_percentiles``
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # If we switch to Python3 exclusively, bins and alpha can be keyword-only
    # arguments
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    _, n_param = samples[0].shape

    # Check number of parameters
    for samples_j in samples:
        if n_param != samples_j.shape[1]:
            raise ValueError(
                'All samples must have the same number of parameters.'
            )

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters.')

    # Set up figure
    fig, axes = plt.subplots(
        n_param, 1, figsize=(6, 2 * n_param),
        squeeze=False,    # Tell matlab to always return a 2d axes object
    )

    # Plot first samples
    for i in range(n_param):
        for j_list, samples_j in enumerate(samples):
            # Add histogram subplot
            axes[i, 0].set_xlabel('Parameter ' + str(i + 1))
            axes[i, 0].set_ylabel('Frequency')
            if n_percentiles is None:
                xmin = np.min(samples_j[:, i])
                xmax = np.max(samples_j[:, i])
            else:
                xmin = np.percentile(samples_j[:, i],
                                     50 - n_percentiles / 2.)
                xmax = np.percentile(samples_j[:, i],
                                     50 + n_percentiles / 2.)
            xbins = np.linspace(xmin, xmax, bins)
            axes[i, 0].hist(
                samples_j[:, i], bins=xbins, alpha=alpha,
                label='Samples ' + str(1 + j_list))

        # Add reference parameters if given
        if ref_parameters is not None:
            # For histogram subplot
            ymin_tv, ymax_tv = axes[i, 0].get_ylim()
            axes[i, 0].plot(
                [ref_parameters[i], ref_parameters[i]],
                [0.0, ymax_tv],
                '--', c='k')
    if n_list > 1:
        axes[0, 0].legend()

    plt.tight_layout()
    return fig, axes[:, 0]


def trace(samples, ref_parameters=None, n_percentiles=None):
    """
    Takes one or more markov chains or lists of samples as input and creates
    and returns a plot showing histograms and traces for each chain or list of
    samples.

    Arguments:

    ``samples``
        A list of lists of samples, with shape
        ``(n_lists, n_samples, n_parameters)``, where ``n_lists`` is the
        number of lists of samples, ``n_samples`` is the number of samples in
        one list and ``n_parameters`` is the number of parameters.
    ``ref_parameters``
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    ``n_percentiles``
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # If we switch to Python3 exclusively, bins and alpha can be keyword-only
    # arguments
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    _, n_param = samples[0].shape

    # Check number of parameters
    for samples_j in samples:
        if n_param != samples_j.shape[1]:
            raise ValueError(
                'All samples must have the same number of parameters.'
            )

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters.')

    # Set up figure
    fig, axes = plt.subplots(
        n_param, 2, figsize=(12, 2 * n_param),

        # Tell matplotlib to return 2d, even if n_param is 1
        squeeze=False,
    )

    # Plot first samples
    for i in range(n_param):
        ymin_all, ymax_all = np.inf, -np.inf
        for j_list, samples_j in enumerate(samples):
            # Add histogram subplot
            axes[i, 0].set_xlabel('Parameter ' + str(i + 1))
            axes[i, 0].set_ylabel('Frequency')
            if n_percentiles is None:
                xmin = np.min(samples_j[:, i])
                xmax = np.max(samples_j[:, i])
            else:
                xmin = np.percentile(samples_j[:, i],
                                     50 - n_percentiles / 2.)
                xmax = np.percentile(samples_j[:, i],
                                     50 + n_percentiles / 2.)
            xbins = np.linspace(xmin, xmax, bins)
            axes[i, 0].hist(samples_j[:, i], bins=xbins, alpha=alpha,
                            label='Samples ' + str(1 + j_list))

            # Add trace subplot
            axes[i, 1].set_xlabel('Iteration')
            axes[i, 1].set_ylabel('Parameter ' + str(i + 1))
            axes[i, 1].plot(samples_j[:, i], alpha=alpha)

            # Set ylim
            ymin_all = ymin_all if ymin_all < xmin else xmin
            ymax_all = ymax_all if ymax_all > xmax else xmax
        axes[i, 1].set_ylim([ymin_all, ymax_all])

        # Add reference parameters if given
        if ref_parameters is not None:
            # For histogram subplot
            ymin_tv, ymax_tv = axes[i, 0].get_ylim()
            axes[i, 0].plot(
                [ref_parameters[i], ref_parameters[i]],
                [0.0, ymax_tv],
                '--', c='k')

            # For trace subplot
            xmin_tv, xmax_tv = axes[i, 1].get_xlim()
            axes[i, 1].plot(
                [0.0, xmax_tv],
                [ref_parameters[i], ref_parameters[i]],
                '--', c='k')
    if n_list > 1:
        axes[0, 0].legend()

    plt.tight_layout()
    return fig, axes


def autocorrelation(samples, max_lags=100):
    """
    Creates and returns an autocorrelation plot for a given markov chain or
    list of `samples`.

    Arguments:

    ``samples``
        A list of samples, with shape ``(n_samples, n_parameters)``, where
        ``n_samples`` is the number of samples in the list and ``n_parameters``
        is the number of parameters.
    ``max_lags``
        (Optional) The maximum autocorrelation lag to plot.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # Check samples size
    try:
        n_sample, n_param = samples.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample,'
                         + ' n_parameters).')

    fig, axes = plt.subplots(n_param, 1, sharex=True, figsize=(6, 2 * n_param))
    if n_param == 1:
        axes = np.asarray([axes], dtype=object)
    for i in range(n_param):
        axes[i].acorr(samples[:, i] - np.mean(samples[:, i]), maxlags=max_lags)
        axes[i].set_xlim(-0.5, max_lags + 0.5)
        axes[i].legend(['Parameter ' + str(1 + i)], loc='upper right')

    # Add x-label to final plot only
    axes[i].set_xlabel('Lag')

    # Add vertical y-label to middle plot
    # fig.text(0.04, 0.5, 'Autocorrelation', va='center', rotation='vertical')
    axes[int(i / 2)].set_ylabel('Autocorrelation')

    plt.tight_layout()
    return fig, axes


def series(samples, problem, ref_parameters=None, thinning=None):
    """
    Creates and returns a plot of predicted time series for a given list of
    ``samples`` and a single-output or multi-output ``problem``.

    Because this method runs simulations, it can take a considerable time to
    run.

    Arguments:

    ``samples``
        A list of samples, with shape ``(n_samples, n_parameters)``, where
        `n_samples` is the number of samples in the list and ``n_parameters``
        is the number of parameters.
    ``problem``
        A :class:``pints.SingleOutputProblem`` or
        :class:``pints.MultiOutputProblem`` of a n_parameters equal to or
        greater than the ``n_parameters`` of the `samples`. Any extra
        parameters present in the chain but not accepted by the
        ``SingleOutputProblem`` or ``MultiOutputProblem`` (for example
        parameters added by a noise model) will be ignored.
    ``ref_parameters``
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    ``thinning``
        (Optional) An integer greater than zero. If specified, only every
        n-th sample (with ``n = thinning``) in the samples will be used. If
        left at the default value ``None``, a value will be chosen so that
        200 to 400 predictions are shown.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # Check samples size
    try:
        n_sample, n_param = samples.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample,'
                         + ' n_parameters).')

    # Get problem n_parameters
    n_parameters = problem.n_parameters()

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param and \
                len(ref_parameters) != n_parameters:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters.')
        ref_series = problem.evaluate(ref_parameters[:n_parameters])

    # Get number of problem output
    n_outputs = problem.n_outputs()

    # Get thinning rate
    if thinning is None:
        thinning = max(1, int(n_sample / 200))
    else:
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError(
                'Thinning rate must be `None` or an integer greater than'
                ' zero.')

    # Get times
    times = problem.times()

    # Evaluate the model for all parameter sets in the samples
    i = 0
    predicted_values = []
    for params in samples[::thinning, :n_parameters]:
        predicted_values.append(problem.evaluate(params))
        i += 1
    predicted_values = np.array(predicted_values)
    mean_values = np.mean(predicted_values, axis=0)

    # Guess appropriate alpha (0.05 worked for 1000 plots)
    alpha = max(0.05 * (1000 / (n_sample / thinning)), 0.5)

    # Plot prediction
    fig, axes = plt.subplots(n_outputs, 1, figsize=(8, np.sqrt(n_outputs) * 3),
                             sharex=True)

    if n_outputs == 1:
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.plot(
            times, problem.values(), 'x', color='#7f7f7f', ms=6.5, alpha=0.5,
            label='Original data')
        plt.plot(
            times, predicted_values[0], color='#1f77b4',
            label='Inferred series')
        for v in predicted_values[1:]:
            plt.plot(times, v, color='#1f77b4', alpha=alpha)
        plt.plot(times, mean_values, 'k:', lw=2,
                 label='Mean of inferred series')

        # Add reference series if given
        if ref_parameters is not None:
            plt.plot(times, ref_series, color='#d62728', ls='--',
                     label='Reference series')

        plt.legend()

    elif n_outputs > 1:
        # Remove horizontal space between axes and set common xlabel
        fig.subplots_adjust(hspace=0)
        axes[-1].set_xlabel('Time')

        # Go through each output
        for i_output in range(n_outputs):
            axes[i_output].set_ylabel('Output %d' % (i_output + 1))
            axes[i_output].plot(
                times, problem.values()[:, i_output], 'x', color='#7f7f7f',
                ms=6.5, alpha=0.5, label='Original data')
            axes[i_output].plot(
                times, predicted_values[0][:, i_output], color='#1f77b4',
                label='Inferred series')
            for v in predicted_values[1:]:
                axes[i_output].plot(times, v[:, i_output], color='#1f77b4',
                                    alpha=alpha)
            axes[i_output].plot(times, mean_values[:, i_output], 'k:', lw=2,
                                label='Mean of inferred series')

            # Add reference series if given
            if ref_parameters is not None:
                axes[i_output].plot(times, ref_series[:, i_output],
                                    color='#d62728', ls='--',
                                    label='Reference series')

        axes[0].legend()

    plt.tight_layout()
    return fig, axes


def pairwise(samples,
             kde=False,
             opacity=None,
             ref_parameters=None,
             n_percentiles=None):
    """
    Takes a markov chain or list of `samples` and creates a set of pairwise
    scatterplots for all parameters (p1 versus p2, p1 versus p3, p2 versus p3,
    etc.).

    The returned plot is in a 'matrix' form, with histograms of each individual
    parameter on the diagonal, and scatter plots of parameters ``i`` and ``j``
    on each entry ``(i, j)`` below the diagonal.

    Arguments:

    ``samples``
        A list of samples, with shape ``(n_samples, n_parameters)``, where
        ``n_samples`` is the number of samples in the list and ``n_parameters``
        is the number of parameters.
    ``kde``
        (Optional) Set to ``True`` to use kernel-density estimation for the
        histograms and scatter plots.
    ``opacity``
        (Optional) When ``kde=False``, this value can be used to manually set
        the opacity of the points in the scatter plots.
    ``ref_parameters``
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    ``n_percentiles``
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from distutils.version import LooseVersion

    # Check matplotlib version
    use_old_matplotlib = LooseVersion(matplotlib.__version__) \
        < LooseVersion("2.2")

    # Check samples size
    try:
        n_sample, n_param = samples.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample,'
                         + ' n_parameters).')

    # Check number of parameters
    if n_param < 2:
        raise ValueError('Number of parameters must be larger than 2.')

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

                if not kde:
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
                else:
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

                    # Force equal aspect ratio
                    # Matplotlib raises a warning here (on 2.7 at least)
                    # We can't do anything about it, so no other option than
                    # to suppress it at this stage...
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UnicodeWarning)
                        axes[i, j].set_aspect((xmax - xmin) / (ymax - ymin))

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
        axes[-1, i].set_xlabel('Parameter %d' % (i + 1))
        if i == 0:
            # The first one is not a parameter
            axes[i, 0].set_ylabel('Frequency')
        else:
            axes[i, 0].set_ylabel('Parameter %d' % (i + 1))

    return fig, axes
