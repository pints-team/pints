#
# Core modules and methods
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints


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

        Returns a sequence of length ``n_times`` (for single output problems)
        or a NumPy array of shape ``(n_times, n_outputs)`` (for multi-output
        problems), representing the values of the model at the given ``times``.

        Parameters
        ----------
        parameters
            An ordered sequence of parameter values.
        times
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.
        """
        raise NotImplementedError

    def n_outputs(self):
        """
        Returns the number of outputs this model has. The default is 1.
        """
        return 1


class ForwardModelS1(ForwardModel):

    """
    Defines an interface for user-supplied forward models which can calculate
    the first-order derivative of the simulated values with respect to the
    parameters.

    Extends :class:`pints.ForwardModel`.
    """

    def __init__(self):
        super(ForwardModelS1, self).__init__()

    def simulateS1(self, parameters, times):
        """
        Runs a forward simulation with the given ``parameters`` and returns a
        time-series with data points corresponding to the given ``times``,
        along with the sensitivities of the forward simulation with respect to
        the parameters.

        Parameters
        ----------
        parameters
            An ordered list of parameter values.
        times
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.

        Returns
        -------
        y
            The simulated values, as a sequence of ``n_times`` values, or
            a NumPy array of shape ``(n_times, n_outputs)``.
        y'
            The corresponding derivatives, as a NumPy array of shape
            ``(n_times, n_parameters)`` or an array of shape
            ``(n_times, n_outputs, n_parameters)``.
        """
        raise NotImplementedError


class SingleOutputProblem(object):

    """
    Represents an inference problem where a model is fit to a single time
    series, such as measured from a system with a single output.

    Parameters
    ----------
    model
        A model or model wrapper extending :class:`ForwardModel`.
    times
        A sequence of points in time. Must be non-negative and increasing.
    values
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

        # Check times and values array have write shape
        if len(self._values) != self._n_times:
            raise ValueError(
                'Times and values arrays must have same length.')

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values as a NumPy array of shape ``(n_times,)``.
        """
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape((self._n_times,))

    def evaluateS1(self, parameters):
        """
        Runs a simulation with first-order sensitivity calculation, returning
        the simulated values and derivatives.

        The returned data is a tuple of NumPy arrays ``(y, y')``, where ``y``
        has shape ``(self._n_times,)`` while ``y'`` has shape
        ``(n_times, n_parameters)``.

        *This method only works for problems with a model that implements the
        :class:`ForwardModelS1` interface.*
        """
        y, dy = self._model.simulateS1(parameters, self._times)
        return (
            np.asarray(y).reshape((self._n_times,)),
            np.asarray(dy).reshape((self._n_times, self._n_parameters))
        )

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

        The returned value is a read-only NumPy array of shape ``(n_times, )``,
        where ``n_times`` is the number of time points.
        """
        return self._times

    def values(self):
        """
        Returns this problem's values.

        The returned value is a read-only NumPy array of shape ``(n_times, )``,
        where ``n_times`` is the number of time points.
        """
        return self._values


class MultiOutputProblem(object):

    """
    Represents an inference problem where a model is fit to a multi-valued time
    series, such as measured from a system with multiple outputs.

    Parameters
    ----------
    model
        A model or model wrapper extending :class:`ForwardModel`.
    times
        A sequence of points in time. Must be non-negative and non-decreasing.
    values
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

        # Check for correct shape
        if self._values.shape != (self._n_times, self._n_outputs):
            raise ValueError(
                'Values array must have shape `(n_times, n_outputs)`.')

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.

        The returned data is a NumPy array with shape ``(n_times, n_outputs)``.
        """
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape(self._n_times, self._n_outputs)

    def evaluateS1(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.

        The returned data is a tuple of NumPy arrays ``(y, y')``, where ``y``
        has shape ``(n_times, n_outputs)``, while ``y'`` has shape
        ``(n_times, n_outputs, n_parameters)``.

        *This method only works for problems whose model implements the
        :class:`ForwardModelS1` interface.*
        """
        y, dy = self._model.simulateS1(parameters, self._times)
        return (
            np.asarray(y).reshape(self._n_times, self._n_outputs),
            np.asarray(dy).reshape(
                self._n_times, self._n_outputs, self._n_parameters)
        )

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

        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._times

    def values(self):
        """
        Returns this problem's values.

        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._values


class TunableMethod(object):

    """
    Defines an interface for a numerical method with a given number of
    hyper-parameters.

    Each optimiser or sampler method implemented in pints has a number of
    parameters which alters its behaviour, which can be called
    "hyper-parameters". The optimiser/sampler method will provide member
    functions to set each of these hyper-parameters individually. In contrast,
    this interface provides a generic way to set the hyper-parameters, which
    allows the user to, for example, use an optimiser to tune the
    hyper-parameters of the method.

    Note that :meth:`set_hyper_parameters` takes an array of parameters, which
    might be of the same type (e.g. a NumPy array). So derived classes should
    not raise any errors if individual hyper parameters are set using the wrong
    type (e.g. float rather than int), but should instead implicitly convert
    the argument to the correct type.
    """

    def n_hyper_parameters(self):
        """
        Returns the number of hyper-parameters for this method (see
        :class:`TunableMethod`).
        """
        return 0

    def set_hyper_parameters(self, x):
        """
        Sets the hyper-parameters for the method with the given vector of
        values (see :class:`TunableMethod`).

        Parameters
        ----------
        x
            An array of length ``n_hyper_parameters`` used to set the
            hyper-parameters.
        """
        pass
