#
# Plots a function defined on a two-dimensional space.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
# This module was was adapted from Myokit (see http://myokit.org) by the
# original author.
#
# The code to plot voronoi regions was based on an example shown here:
# http://stackoverflow.com/a/20678647/423420
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np

import pints


def surface(points, values, boundaries=None, markers='+', figsize=None):
    """
    Takes irregularly spaced points and function evaluations in a
    two-dimensional parameter space and creates a coloured surface plot using a
    voronoi diagram.

    Returns a ``matplotlib`` figure object and axes handle.

    Parameters
    ----------
    points
        A list of (two-dimensional) points in parameter space.
    values
        The values corresponding to these points.
    boundaries
        An optional :class:`pints.RectangularBoundaries` object to restrict the
        area shown.
    markers
        The markers to use to plot the sampled points. Set to ``None`` to
        disable.

    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    # Check points and values
    points = pints.matrix2d(points)
    n, d = points.shape
    if d != 2:
        raise ValueError('Only two-dimensional parameters are supported.')
    values = pints.vector(values)
    if len(values) != n:
        raise ValueError(
            'The number of values must match the number of points.')

    # Extract x and y points
    x = points[:, 0]
    y = points[:, 1]
    del(points)

    # Check boundaries
    if boundaries is None:
        xmin = 0.80 * np.min(x)
        xmax = 1.02 * np.max(x)
        ymin = 0.80 * np.min(y)
        ymax = 1.02 * np.max(y)
    else:
        if boundaries.n_parameters() != 2:
            raise ValueError(
                'If given, the boundaries must be two-dimensional.')

        xmin, ymin = boundaries.lower()
        xmax, ymax = boundaries.upper()

    # Get voronoi regions (and filter points and evaluations)
    xlim = xmin, xmax
    ylim = ymin, ymax
    x, y, values, regions = _voronoi_regions(x, y, values, xlim, ylim)

    # Create figure and axes
    figure, axes = plt.subplots(figsize=figsize)
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)

    # Add coloured voronoi regions
    c = PolyCollection(
        regions, array=values, edgecolors='none', cmap='viridis_r')
    axes.add_collection(c)

    # Add markers
    if markers:
        axes.plot(x, y, markers)

    # Add colorbar
    figure.colorbar(c, ax=axes)

    return figure, axes


def _voronoi_regions(x, y, f, xlim, ylim):
    """
    Takes a set of ``(x, y, f)`` points and returns the edgepoints of the
    voronoi region around each point within the boundaries specified by
    ``xlim`` and ``ylim``.

    Points and voronoi regions entirely outside the specified bounds will be
    dropped. Voronoi regions partially outside the bounds will be truncated.
    The third array ``f`` will be filtered the same way as ``x`` and ``y`` but
    is otherwise not used.

    Parameters
    ----------
    x
        A list of x-coorindates
    y
        A list of y-coordinates
    f
        The score function at the given x and y coordinates
    xlim
        Lower and upper bound for the x coordinates
    ylim
        Lower and upper bound for the y coordinates

    Returns
    -------
    A tuple ``(x, y, f, regions)`` where ``x``, ``y`` and ``f`` are the
    coordinates of the accepted points and each ``regions[i]`` is a list of the
    vertices making up the voronoi region for point ``(x[i], y[i])``.
    """
    from scipy.spatial import Voronoi
    try:
        from itertools import izip  # Python 2's izip acts like Python 3's zip
    except ImportError:
        izip = zip

    # Don't check x, y, f: handled by calling method
    n = len(x)

    # Check limits
    xmin, xmax = [float(a) for a in sorted(xlim)]
    ymin, ymax = [float(a) for a in sorted(ylim)]

    # Drop any points outside the bounds
    # within_bounds = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    # x = x[within_bounds]
    # y = y[within_bounds]
    # f = f[within_bounds]

    # Create voronoi diagram
    vor = Voronoi(np.array([x, y]).transpose())

    # The voronoi diagram consists of a number of ridges drawn between a set
    # of points.
    #
    #   points          Are the points the diagram is based on.
    #   vertices        The coordinates of the vertices connecting the ridges
    #   ridge_points    Is a list of tuples (p1, p2) defining the points each
    #                   ridge belongs to. Points are given as their index in
    #                   the list of points.
    #   ridge_vertices  Is a list of vertices (v1, v2) defining the vertices
    #                   between which each ridge is drawn. Vertices are given
    #                   as their index in the list of vertice coordinates. For
    #                   ridges extending to infinity, one of the vertices will
    #                   be given as -1.
    #
    # Get the center of the voronoi diagram's points and define a radius that
    # will bring us outside the visible area for any starting point / direction
    #
    center = vor.points.mean(axis=0)
    radius2 = 2 * np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)

    # Create a list containing the set of vertices defining each region
    regions = [set() for i in range(n)]
    for (p1, p2), (v1, v2) in izip(vor.ridge_points, vor.ridge_vertices):
        # Ensure only v1 can every be -1
        if v1 > v2:
            v1, v2 = v2, v1

        # Get vertice coordinates
        x2 = vor.vertices[v2]  # Only v1 can be -1
        if v1 >= 0:
            # Finite vertex
            x1 = vor.vertices[v1]
        else:
            # Replacement vertex needed
            # Tangent line to points involved
            y1, y2 = vor.points[p1], vor.points[p2]
            t = y2 - y1
            t /= np.linalg.norm(t)

            # Normal line
            q = np.array([-t[1], t[0]])

            # Midpoint between involved points
            midpoint = np.mean([y1, y2], axis=0)

            # Point beyond the outer boundary
            x1 = x2 + np.sign(np.dot(midpoint - center, q)) * q * radius2

        # Add vertice coordinates to both region coordinate lists
        x1, x2 = tuple(x1), tuple(x2)  # arrays and lists aren't hashable
        regions[p1].update((x1, x2))
        regions[p2].update((x1, x2))

    # Order vertices in regions counter clockwise, truncate the regions at the
    # border, and remove regions outside of the bounds.
    selection = []
    for k, region in enumerate(regions):
        if len(region) == 0:
            continue

        # Convert set of tuples to 2d array
        region = np.asarray([np.asarray(v) for v in region])

        # Filter out any regions lying entirely outside the bounds
        if not (np.any((region[:, 0] > xmin) | (region[:, 0] < xmax)) or
                np.any((region[:, 1] > ymin) | (region[:, 1] < ymax))):
            continue

        # Sort vertices counter clockwise
        p = vor.points[k]
        angles = np.arctan2(region[:, 1] - p[1], region[:, 0] - p[0])
        regions[k] = region[np.argsort(angles)]

        # Region fully contained? Then keep in selection and continue
        if not np.any((region[:, 0] < xmin) | (region[:, 0] > xmax) |
                      (region[:, 1] < ymin) | (region[:, 1] > ymax)):
            selection.append(k)
            continue

        # Region needs truncating

        # Drop points outside of boundary and replace by 0, 1 or 2 new points
        # on the actual boundaries.
        # Run twice: once for x violations, once for y violations (two
        # successive corrections may be needed, to solve corner issues).
        new_region = []
        for j, p in enumerate(region):
            if p[0] < xmin:
                q = region[j - 1] if j > 0 else region[-1]
                r = region[j + 1] if j < len(region) - 1 else region[0]
                if q[0] < xmin and r[0] < xmin:
                    # Point connecting two outsiders: drop
                    continue
                if q[0] >= xmin:
                    # Add point on line p-q
                    s = p[1] + (xmin - p[0]) * (q[1] - p[1]) / (q[0] - p[0])
                    new_region.append(np.array([xmin, s]))
                if r[0] >= xmin:
                    # Add point on line p-r
                    s = p[1] + (xmin - p[0]) * (r[1] - p[1]) / (r[0] - p[0])
                    new_region.append(np.array([xmin, s]))
            elif p[0] > xmax:
                q = region[j - 1] if j > 0 else region[-1]
                r = region[j + 1] if j < len(region) - 1 else region[0]
                if q[0] > xmax and r[0] > xmax:
                    # Point connecting two outsiders: drop
                    continue
                if q[0] <= xmax:
                    # Add point on line p-q
                    s = p[1] + (xmax - p[0]) * (q[1] - p[1]) / (q[0] - p[0])
                    new_region.append(np.array([xmax, s]))
                if r[0] <= xmax:
                    # Add point on line p-r
                    s = p[1] + (xmax - p[0]) * (r[1] - p[1]) / (r[0] - p[0])
                    new_region.append(np.array([xmax, s]))
            else:
                # Point is fine, just add
                new_region.append(p)
        region = new_region

        # Run again for y-violations
        new_region = []
        for j, p in enumerate(region):
            if p[1] < ymin:
                q = region[j - 1] if j > 0 else region[-1]
                r = region[j + 1] if j < len(region) - 1 else region[0]
                if q[1] < ymin and r[1] < ymin:
                    # Point connecting two outsiders: drop
                    continue
                if q[1] >= ymin:
                    # Add point on line p-q
                    s = p[0] + (ymin - p[1]) * (q[0] - p[0]) / (q[1] - p[1])
                    new_region.append(np.array([s, ymin]))
                if r[1] >= ymin:
                    # Add point on line p-r
                    s = p[0] + (ymin - p[1]) * (r[0] - p[0]) / (r[1] - p[1])
                    new_region.append(np.array([s, ymin]))
            elif p[1] > ymax:
                q = region[j - 1] if j > 0 else region[-1]
                r = region[j + 1] if j < len(region) - 1 else region[0]
                if q[1] > ymax and r[1] > ymax:
                    # Point connecting two outsiders: drop
                    continue
                if q[1] <= ymax:
                    # Add point on line p-q
                    s = p[0] + (ymax - p[1]) * (q[0] - p[0]) / (q[1] - p[1])
                    new_region.append(np.array([s, ymax]))
                if r[1] <= ymax:
                    # Add point on line p-r
                    s = p[0] + (ymax - p[1]) * (r[0] - p[0]) / (r[1] - p[1])
                    new_region.append(np.array([s, ymax]))
            else:
                # Point is fine, just add
                new_region.append(p)
        region = new_region

        # Store regions that are still OK
        if len(region) > 2:
            selection.append(k)

    # Filter out bad regions
    regions = np.array(regions)
    regions = regions[selection]
    x = x[selection]
    y = y[selection]
    f = f[selection]

    # Return output
    return x, y, f, regions
