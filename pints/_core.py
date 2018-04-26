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

        Returns:

        A sequence of length ``len(times)`` representing the values of the
        model at the given ``times``.

        Note: For efficiency, both ``parameters`` and ``times`` will be passed
        in as read-only numpy arrays.
        """
        raise NotImplementedError

    def n_outputs(self):
        """
        Returns the number of outputs this model has. The default is 1.
        """
        return 1

    def n_parameters(self):
        """
        Returns the number of parameters this model takes as input.
        """
        raise NotImplementedError


class ForwardModelS1(object):
    """
    Defines an interface for user-supplied forward models that calculate both a
    set of values ``y``, and a set of derivatives ``dy/dp`` of those variables
    with respect to the parameters.
    """
    def __init__(self):
        super(ForwardModelS1, self).__init__()

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

        Returns:

        A tuple ``(y, y')``, where ``y`` is a sequence of length ``len(times)``
        representing the values of the model at the given ``times``, and where
        ``y'`` contains the derivatives of each point in ``y'`` with respect to
        the ``parameters``.

        Note: For efficiency, both ``parameters`` and ``times`` will be passed
        in as read-only numpy arrays.
        """
        raise NotImplementedError

    def n_outputs(self):
        """
        Returns the number of outputs this model has. The default is 1.
        """
        return 1

    def n_parameters(self):
        """
        Returns the number of parameters this model takes as input.
        """
        raise NotImplementedError


class AbstractProblem(object):
    """
    Abstract base class for different problem types.
    """
    def __init__(self, model, times, values):

        # Check model
        self._model = self._check_model(model)

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be non-decreasing.')

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times)

        # Check values, copy so that they can no longer be changed
        self._values = self._check_values(values)

    def _check_model(self, model):
        """
        Checks if the given model is appropriate for this problem class, and
        converts it if possible.
        """
        raise NotImplementedError(
            'AbstractProblem._check_model not implemented, or attempted to'
            ' instantiate AbstractModel directly.')

    def _check_values(self, values):
        """
        Checks if the given values are appropriate for this problem, and
        returns a read-only version of the values.
        """
        raise NotImplementedError(
            'AbstractProblem._check_values not implemented, or attempted to'
            ' instantiate AbstractModel directly.')

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


class SingleOutputProblem(AbstractProblem):
    """
    Represents an inference problem where a model is fit to a single time
    series, such as measured from a system with a single output.

    Arguments:

    ``model``
        A single-output model or model wrapper extending :class:`ForwardModel`.
    ``times``
        A sequence of points in time. Must be non-negative and increasing.
    ``values``
        A sequence of scalar output values, measured at the times in ``times``.

    """
    def _check_model(self, model):
        """ See :meth:`AbstractProblem._check_model()`. """
        if not isinstance(model, pints.ForwardModel):
            raise TypeError('SingleOutputProblem requires a ForwardModel.')
        if model.n_outputs() != 1:
            raise ValueError(
                'Only single-output models can be used for a'
                ' SingleOutputProblem.')
        return model

    def _check_values(self, values):
        """ See :meth:`AbstractProblem._check_values()`. """
        # Check times and values array have write shape
        if len(values) != self._n_times:
            raise ValueError(
                'Times and values arrays must have same length.')
        return pints.vector(values)

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        """
        return self._model.simulate(parameters, self._times)


class SingleOutputProblemS1(AbstractProblem):
    """
    A single-output problem definition for models that provide first-order
    sensitivities (derivatives with resepct to the parameters).

    See also: :class:`SingleOutputProblem`.

    Arguments:

    ``model``
        A single-output model or model wrapper extending
        :class:`ForwardModelS1`.
    ``times``
        A sequence of points in time. Must be non-negative and increasing.
    ``values``
        A sequence of scalar output values, measured at the times in ``times``.

    """
    def _check_model(self, model):
        """ See :meth:`AbstractProblem._check_model()`. """
        if not isinstance(model, pints.ForwardModelS1):
            raise TypeError('SingleOutputProblemS1 requires a ForwardModelS1.')
        if model.n_outputs() != 1:
            raise ValueError(
                'Only single-output models can be used for a'
                ' SingleOutputProblem.')
        return model

    def _check_values(self, values):
        """ See :meth:`AbstractProblem._check_values()`. """
        # Check times and values array have write shape
        if len(values) != self._n_times:
            raise ValueError(
                'Times and values arrays must have same length.')
        return pints.vector(values)

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning a tuple with
        the simulated values and their first-order derivatives.
        """
        return self._model.simulate(parameters, self._times)


class MultiOutputProblem(AbstractProblem):
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
    def _check_model(self, model):
        """ See :meth:`AbstractProblem._check_model()`. """
        if not isinstance(model, pints.ForwardModel):
            raise TypeError('MultiOutputProblem requires a ForwardModel.')
        return model

    def _check_values(self, values):
        """ See :meth:`AbstractProblem._check_values()`. """
        values = pints.matrix2d(values)
        if values.shape != (self._n_times, self._n_outputs):
            raise ValueError(
                'Values array must have shape `(n_times, n_outputs)`.')
        return values

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        """
        return self._model.simulate(parameters, self._times)


class MultiOutputProblemS1(MultiOutputProblem):
    """
    A multi-output problem definition for models that provide first-order
    sensitivities (derivatives with resepct to the parameters).

    See also: :class:`MultiOutputProblem`.


    Arguments:

    ``model``
        A model or model wrapper extending :class:`ForwardModelS1`.
    ``times``
        A sequence of points in time. Must be non-negative and non-decreasing.
    ``values``
        A sequence of multi-valued measurements. Must have shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of points in
        ``times`` and ``n_outputs`` is the number of outputs in the model.

    """
    def _check_model(self, model):
        """ See :meth:`AbstractProblem._check_model()`. """
        if not isinstance(model, pints.ForwardModelS1):
            raise TypeError('SingleOutputProblemS1 requires a ForwardModelS1.')
        return model

    def _check_values(self, values):
        """ See :meth:`AbstractProblem._check_values()`. """
        values = pints.matrix2d(values)
        if values.shape != (self._n_times, self._n_outputs):
            raise ValueError(
                'Values array must have shape `(n_times, n_outputs)`.')
        return values

