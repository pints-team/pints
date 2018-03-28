#
# Core modules and methods
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class ForwardModel(object):
    """
    Defines an interface for user-supplied forward models.

    Classes extending ``ForwardModel`` can implement the required methods
    directly in Python or interface with other languages (for example via
    Python wrappers around C code).
    """

    def __init__(self):
        super(ForwardModel, self).__init__()

    def n_parameters(self):
        """
        Returns the dimension of the parameter space.
        """
        raise NotImplementedError

    def simulate(self, parameters, times):
        """
        Runs a forward simulation with the given ``parameters`` and returns a
        time-series with data points corresponding to the given ``times``.

        Arguments:

        ``parameters``
            An ordered list of parameter values.
        ``times``
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.

        Note: For efficiency, both ``parameters`` and ``times`` will be passed
        in as read-only numpy arrays.
        """
        raise NotImplementedError

    def n_outputs(self):
        """
        Returns the number of outputs this model has. The default is 1.
        """
        return 1


class SingleOutputProblem(object):
    """
    Represents an inference problem where a model is fit to a single time
    series, such as measured from a system with a single output.

    Arguments:

    ``model``
        A model or model wrapper extending :class:`ForwardModel`.
    ``times``
        A sequence of points in time. Must be non-negative and increasing.
    ``values``
        A sequence of scalar output values, measured at the times in ``times``.

    """
    def __init__(self, model, times, values):

        # Check model
        self._model = model
        self._dimension = int(model.n_parameters())
        if model.n_outputs() != 1:
            raise ValueError(
                'Only single-output models can be used for a'
                ' SingleOutputProblem.')

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times can not be negative.')
        if np.any(self._times[:-1] >= self._times[1:]):
            raise ValueError('Times must be increasing.')

        # Check values, copy so that they can no longer be changed
        self._values = pints.vector(values)

        # Check times and values array have write shape
        if len(self._times) != len(self._values):
            raise ValueError(
                'Times and values arrays must have same length.')

    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._dimension

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        """
        return self._model.simulate(parameters, self._times)

    def n_outputs(self):
        """
        Returns the number of outputs for this problem (always 1).
        """
        return 1

    def times(self):
        """
        Returns this problem's times.

        The returned value is a read-only numpy array of shape ``(n_times, )``,
        where ``n_times`` is the number of time points.
        """
        return self._times

    def values(self):
        """
        Returns this problem's values.

        The returned value is a read-only numpy array of shape ``(n_times, )``,
        where ``n_times`` is the number of time points.
        """
        return self._values


class MultiOutputProblem(object):
    """
    Represents an inference problem where a model is fit to a multi-valued time
    series, such as measured from a system with multiple outputs.

    Arguments:

    ``model``
        A model or model wrapper extending :class:`ForwardModel`.
    ``times``
        A sequence of points in time. Must be non-negative and non-decreasing.
    ``values``
        A sequence of multi-valued measurements. Must have shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of points in
        ``times`` and ``n_outputs`` is the number of outputs in the model.

    """
    def __init__(self, model, times, values):

        # Check model
        self._model = model
        self._dimension = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be non-decreasing.')

        # Check values, copy so that they can no longer be changed
        self._values = pints.matrix2d(values)

        # Check for correct shape
        if self._values.shape != (len(self._times), self._n_outputs):
            raise ValueError(
                'Values array must have shape `(n_times, n_outputs)`.')

    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._dimension

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        """
        return self._model.simulate(parameters, self._times)

    def n_outputs(self):
        """
        Returns the number of outputs for this problem.
        """
        return self._n_outputs

    def times(self):
        """
        Returns this problem's times.

        The returned value is a read-only numpy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._times

    def values(self):
        """
        Returns this problem's values.

        The returned value is a read-only numpy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._values
