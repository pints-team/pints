#
# Evaluate function between two points
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np

import pints


def function_between_points(f, point_1, point_2, padding=0.25, evaluations=20):
    """
    Creates and returns a plot of a function between two points in parameter
    space.

    Returns a ``matplotlib`` figure object and axes handle.

    Parameters
    ----------
    f
        A :class:`pints.LogPDF` or :class:`pints.ErrorMeasure` to plot.
    point_1
        The first point in parameter space. The method will find a line from
        ``point_1`` to ``point_2`` and plot ``f`` at several points along it.
    point_2
        The second point.
    padding
        Specifies the amount of padding around the line segment
        ``[point_1, point_2]`` that will be shown in the plot.
    evaluations
        The number of evaluation along the line in parameter space.
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

