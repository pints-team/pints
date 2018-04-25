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

    def max_derivatives(self):
        """
        Returns the highest order of derivatives this model can return.
        The default is ``0`` (no derivatives).

        By returning ``1`` models declare that they support calculation of
        first-order derivatives ``dy/dp`` of the simulated values ``y`` with
        respect to the parameters ``p``.

        By returning ``2`` models declare that they support calculation of
        first-order derivatives ``dy/dp`` and second-order derivatives
        ``d2y/dp2`` of the simulated values ``y`` with respect to the
        parameters ``p``.
        """
        return 0

    def n_outputs(self):
        """
        Returns the number of outputs this model has. The default is 1.
        """
        return 1

    def n_parameters(self):
        """
        Returns the dimension of the parameter space.
        """
        raise NotImplementedError

    def simulate(self, parameters, times, n_derivatives=0):
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
        ``n_derivatives``
            An optional argument specifying the highest order of derivatives
            to include in the results (default=0, see below).
            Models that do not support derivatives can ignore this argument.

        Returns:

        For models that don't support derivatives, this method should return a
        NumPy array ``y`` of length ``len(times)`` representing the values of
        the model at the given time points.

        For models that support first-order derivatives (i.e. that return
        :meth:`max_derivatives() > 0`), the return type depends on the value of
        the optional argument ``n_derivatives``.
        In this case, for ``n_derivatives=0`` the returned value is ``y``, for
        ``n_dervatives=1`` the returned value is a tuple  ``(y, y')``, and for
        ``n_derivatives=2`` a tuple ``(y, y', y'')`` is returned.
        Requests for ``n_derivatives`` higher than the model supports can
        safely be ignored.

        Note: For efficiency, both ``parameters`` and ``times`` will be passed
        in as read-only numpy arrays.
        """
        raise NotImplementedError


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

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_times = len(self._times)

        # Check highest supported order of derivatives
        self._max_derivatives = int(model.max_derivatives())

        # Check times and values array have write shape
        if len(self._values) != self._n_times:
            raise ValueError(
                'Times and values arrays must have same length.')

    def evaluate(self, parameters, n_derivatives=0):
        """
        Runs a simulation using the given ``parameters``, and returns the
        simulated values.

        For problems that support first-order derivatives, the optional
        argument ``n_derivatives`` can be set to ``1`` to return a tuple
        ``(values, jacobian)`` instead, where ``values`` is a vector containing
        the simulated values and ``jacobian`` is a matrix containing the
        first-order derivatives of those values with respect to the
        ``parameters``.

        For problems that support second-order derivatives, ``n_derivatives``
        can be set to ``2`` to return a tuple ``(values, jacobian, hessian)``
        where ``hessian`` is a matrix containing the second-order derivatives
        of ``values`` with respect to ``parameters``.

        See also :meth:`max_derivatives()`.
        """
        return self._model.simulate(parameters, self._times, n_derivatives=0)

    def max_derivatives(self):
        """
        Returns the highest order of derivatives this problem can evaluate (see
        :meth:`ForwardModel.max_derivatives`).
        """
        return self._max_derivatives

    def n_outputs(self):
        """
        Returns the number of outputs for this problem (always 1).
        """
        return 1

    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._n_parameters

    def n_times(self):
        """
        Returns the number of sampling points, i.e. the length of the vectors
        returned by :meth:`times()` and :meth:`values()`.
        """
        return self._n_times

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

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be non-decreasing.')

        # Check values, copy so that they can no longer be changed
        self._values = pints.matrix2d(values)

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times)

        # Check highest supported order of derivatives
        self._max_derivatives = int(model.max_derivatives())

        # Check for correct shape
        if self._values.shape != (self._n_times, self._n_outputs):
            raise ValueError(
                'Values array must have shape `(n_times, n_outputs)`.')

    def evaluate(self, parameters, n_derivatives=0):
        """
        Runs a simulation using the given ``parameters``, and returns the
        simulated values.

        For problems that support first-order derivatives, the optional
        argument ``n_derivatives`` can be set to ``1`` to return a tuple
        ``(values, jacobian)`` instead, where ``values`` is a vector containing
        the simulated values and ``jacobian`` is a matrix containing the
        first-order derivatives of those values with respect to the
        ``parameters``.

        For problems that support second-order derivatives, ``n_derivatives``
        can be set to ``2`` to return a tuple ``(values, jacobian, hessian)``
        where ``hessian`` is a matrix containing the second-order derivatives
        of ``values`` with respect to ``parameters``.

        See also :meth:`max_derivatives()`.
        """
        return self._model.simulate(parameters, self._times, n_derivatives=0)

    def max_derivatives(self):
        """
        Returns the highest order of derivatives this problem can evaluate (see
        :meth:`ForwardModel.max_derivatives`).
        """
        return self._max_derivatives

    def n_outputs(self):
        """
        Returns the number of outputs for this problem.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._n_parameters

    def n_times(self):
        """
        Returns the number of sampling points, i.e. the length of the vectors
        returned by :meth:`times()` and :meth:`values()`.
        """
        return self._n_times

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
