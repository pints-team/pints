#
# Plotting functions for emulator related problems
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

def surface( 
        x_grid, y_grid, z_grid, 
        cmap="Blues", angle=(25, 300), alpha=1.,
        fontsize = 14, labelpad = 10, 
        title="", x_label="", y_label="", z_label ="log_likelihood"):
    """
    Creates 3d contour plot given a grid for each axis. 

    Arguments:

    ``x_grid``
        An NxN grid of values.
    ``y_grid``
        An NxN grid of values.
    ``z_grid``
        An NxN grid of values. z_grid determines colour.
    ``cmap``
        (Optional) Colour map used in the plot
    ``angle``
        (Optional) tuple specifying the viewing angle of the graph
    ``alpha``
        (Optional) alpha parameter of the surface
    ``fill``
        (Optional) Used to specify whether or not contour plot should be filled. Deafault False.
    ``fontsize``
        (Optional) the fontsize used for labels
    ``labelpad``
        (Optional) distance of axis labels from the labels
    ``x_label``
        (Optional) The label of the x-axis
    ``y_label``
        (Optional) The label of the y-axis
    ``z_label``
        (Optional) The label of the z-axis 

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    ax.plot_surface(x_grid, y_grid, z_grid, cmap = cmap, alpha = alpha)
    ax.view_init(*angle)

    fontsize = fontsize
    labelpad = labelpad
    
    if title:
        plt.title(title, fontsize = fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize = fontsize, labelpad = labelpad)
    if y_label:
        ax.set_ylabel(y_label, fontsize = fontsize, labelpad = labelpad)
    if z_label:
        ax.set_zlabel(z_label, fontsize = fontsize, labelpad = labelpad)
    
    return ax

def contour( 
        x_grid, y_grid, z_grid, 
        cmap="Blues", fill = False,
        fontsize = 14, labelpad = 10, 
        title="", 
        x_label="", y_label=""):
    """
    Creates 3d contour plot given a grid for each axis. 

    Arguments:

    ``x_grid``
        An NxN grid of values.
    ``y_grid``
        An NxN grid of values.
    ``z_grid``
        An NxN grid of values. z_grid determines colour.
    ``cmap``
        (Optional) Colour map used in the plot
    ``fill``
        (Optional) Used to specify whether or not contour plot should be filled. Deafault False.
    ``fontsize``
        (Optional) the fontsize used for labels
    ``labelpad``
        (Optional) distance of axis labels from the labels
    ``x_label``
        (Optional) The label of x-axis
    ``y_label``
        (Optional) The label of y-axis

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    
    if fill:
        axes.contourf(x_grid, y_grid, z_grid, cmap = cmap)
    else:
        axes.contour(x_grid, y_grid, z_grid, cmap = cmap)

    fontsize = fontsize
    labelpad = labelpad
    
    if title:
        plt.title(title, fontsize = fontsize)
    if x_label:
        axes.set_xlabel(x_label, fontsize = fontsize, labelpad = labelpad)
    if y_label:
        axes.set_ylabel(y_label, fontsize = fontsize, labelpad = labelpad)

    plt.tight_layout()
    return fig, axes

def confidence_interval(param_range, mean, conf, show_points = True):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    lower = mean - conf
    upper = mean + conf

    axes.plot(param_range, mean, color = "black")
    axes.plot(param_range, upper, color = "grey")
    axes.plot(param_range, lower, color = "grey")
    axes.fill_between(param_range, lower, upper, color = "lightgrey")

    if show_points:
        axes.scatter(param_range, mean)

    plt.tight_layout()
    return fig, axes

