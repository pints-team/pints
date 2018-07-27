#
# Base class for all emulators
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np
import pints
import copy

class Emulator(pints.LogLikelihood):
    """
    *Extends:* :class:`LogLikelihood`

    Abstract class from which all emulators should inherit.
    An instance of the Emulator models given LogLikelihoods with given 

    Arguments:

    ``log_likelihood``
        A :class:`LogLikelihood`, the likelihood distribution being emulated.
    ``X``
        N by n_paremeters matrix containing inputs for training data
    ``y``
        N by 1, target values for each input vector
    ``normalize_input``
        If true then inputs will be normalized to have mean 0
    """

    def __init__(self, log_likelihood, X, y, normalize_input=False):
        # Perform sanity checks for given data
        if not isinstance(log_likelihood, pints.LogLikelihood):
            raise ValueError("Given pdf must extand LogLikelihood")

        self._n_parameters = log_likelihood.n_parameters()

        # check if dimensions are valid
        if X.ndim != 2:
            raise ValueError("Input should be 2 dimensional")

        X_r, X_c = X.shape

        if (X_c != self._n_parameters):
            raise ValueError("Input data should have n_parameters features")

        # if given target array is 1d convert automatically
        if y.ndim == 1:
            y = y.reshape(len(y), 1)

        if y.ndim != 2:
            raise ValueError("Target array should be 2 dimensional (N, 1)")

        y_r, y_c = y.shape

        if y_c != 1:
            raise ValueError("Target array should only have 1 feature")

        if (X_r != y_r):
            raise ValueError("Input and target dimensions don't match")

        self._normalize_input = normalize_input
        if normalize_input:
            self._means = np.mean(X, axis=0)
            self._stds = np.std(X, axis=0)
            X = (X - self._means) / self._stds

        # copy input data
        self._X = copy.deepcopy(X)
        self._y = copy.deepcopy(y)

    def n_parameters(self):
        return self._n_parameters
