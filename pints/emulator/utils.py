#
# A collection of useful functions when dealing with emulators
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np

def generate_grid(lower, upper, splits, fixed = []):
    """
    Generates a grid of evenly spaced out points for testing
    returns grid of the first paramater, grid of second parameter, 
    and their values stacked with any fixed values provided
    Generated values are convenient for plotting surface
    or contour plots.
    fixed -- contains position and argument to keep fixed
    """
    # TODO: add appropriate length checks


    #
    if not fixed:
        p1_low, p2_low = lower
        p1_high, p2_high = upper
    else:
        # find out which parameters are not fixed and get their bounds
        n_params = len(lower)
        params = list(range(n_params))
        for (i, _) in fixed:
            params.pop(i)
        p1_idx, p2_idx = params
        p1_low, p2_low = lower[p1_idx], lower[p2_idx]
        p1_high, p2_high = upper[p1_idx], upper[p2_idx] 
    
    p1_range = np.linspace(p1_low, p1_high, splits)
    p2_range = np.linspace(p2_low, p2_high, splits)
    p1_grid, p2_grid = np.meshgrid(p1_range, p2_range)

    if fixed:
        # create a grid for every parameter and insert in
        # corresponding position in the grids array
        grids = [None] * n_params
        grids[p1_idx] = p1_grid
        grids[p2_idx] = p2_grid
        for (i, val) in fixed:
            fixed_grid = np.zeros((splits, splits)) + val
            grids[i] = fixed_grid
    else:
        grids = [p1_grid, p2_grid]
    
    grid = np.dstack( tuple(grids) )

    return p1_grid, p2_grid, grid

def predict_grid(model, grid, dims = None):
    """
    Given a PDF and a grid of inputs calculates probability for
    each index in the grid
    """
    rows, cols, n_params = grid.shape
    flatten_grid = grid.reshape((rows * cols, n_params))
    pred = np.apply_along_axis(model, 1, flatten_grid)
    return pred.reshape(rows, cols)