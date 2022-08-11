#
# Evaluate a function around a point
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np

import pints


def function(f, x, lower=None, upper=None, evaluations=20):
    """
    Creates 1d plots of a :class:`LogPDF` or a :class:`ErrorMeasure` around a
    point `x` (i.e. a 1-dimensional plot in each direction).

    Returns a ``matplotlib`` figure object and axes handle.

    Parameters
    ----------
    f
        A :class:`pints.LogPDF` or :class:`pints.ErrorMeasure` to plot.
    x
        A point in the function's input space.
    lower
        Optional lower bounds for each parameter, used to specify the lower
        bounds of the plot.
    upper
        Optional upper bounds for each parameter, used to specify the upper
        bounds of the plot.
    evaluations
        The number of evaluations to use in each plot.
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

