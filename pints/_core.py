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


class ProblemCollection(object):
    """
    Represents an inference problem where a model is fit to a multi-valued time
    series, such as when measured from a system with multiple outputs, where
    the different time series are potentially measured at different time
    intervals.

    This class is also of use when different outputs are modelled with
    different likelihoods or score functions.

    Parameters
    ----------
    model
        A model or model wrapper extending :class:`ForwardModel`.
    args
        Consecutive times, values lists for each output chunk. For example,
        times_1, values_1, times_2, values_2: where times_1 = [1.2, 2.5, 3] and
        values_1 = [2.3, 4.5, 4.5]; times_2 = [4, 5, 6, 7] and
        values_2 = [[3.4, 1.1, 0.5, 0.6], [1.2, 3.3, 4.5, 5.5]].
    """
    def __init__(self, model, *args):
        self._model = model
        print("pints length = ", len(args))
        if len(args) < 2:
            raise ValueError('Must supply at least one time series.')
        if len(args) % 2 != 0:
            raise ValueError(
                'Must supply times and values for each time series.')
        self._timeses = []
        self._valueses = []
        self._output_indices = []

        k = 0
        self._n_output_sets = len(args) // 2
        for i in range(self._n_output_sets):
            times = np.array(args[k])
            times_shape = times.shape
            if len(times_shape) != 1:
                raise ValueError('Times must be one-dimensional.')
            values = np.array(args[k + 1])
            values_shape = values.shape
            if values_shape[0] != times_shape[0]:
                raise ValueError('Outputs must be of same length as times.')
            self._timeses.append(times)
            self._valueses.append(values)
            if len(values_shape) > 1:
                n_outputs = values_shape[1]
            else:
                n_outputs = 1
            self._output_indices.extend([i] * n_outputs)
            k += 2
        self._times_all = np.sort(list(set(np.concatenate(self._timeses))))
        self._output_indices = np.array(self._output_indices)

        # vars to handle caching across multiple output chunks
        self._cached_output = None
        self._cached_sensitivities = None
        self._cached_parameters = None

    def _output_sorter(self, y, index):
        """
        Returns output(s) corresponding to a given index at times corresponding
        to that output.
        """
        # lookup times in times array
        times = self._timeses[index]
        time_indices = [np.where(self._times_all == x)[0][0] for x in times]

        # find relevant output indices
        output_indices = np.where(self._output_indices == index)[0]

        # pick rows then columns
        y_short = y[time_indices, :]
        y_short = y_short[:, output_indices]
        if y_short.shape[1] == 1:
            y_short = y_short.reshape((len(self._timeses[index]),))
        return y_short

    def _output_and_sensitivity_sorter(self, y, dy, index):
        """
        Returns output(s) and sensitivities corresponding to a given index at
        times corresponding to that output.
        """
        # lookup times in times array
        times = self._timeses[index]
        time_indices = [np.where(self._times_all == x)[0][0] for x in times]

        # find relevant output indices
        output_indices = np.where(self._output_indices == index)[0]

        # pick rows then columns
        y_short = y[time_indices, :]
        y_short = y_short[:, output_indices]
        if y_short.shape[1] == 1:
            y_short = y_short.reshape((len(self._timeses[index]),))

        # sort sensitivities
        dy_short = dy[time_indices, :, :]
        dy_short = dy_short[:, output_indices, :]

        if len(output_indices) == 1:
            dy_short = dy_short.reshape(
                len(time_indices), 1, dy_short.shape[2])
        return y_short, dy_short

    def _evaluate(self, parameters, index):
        """ Evaluates model or returns cached result. """
        parameters = pints.vector(parameters)
        if not np.array_equal(self._cached_parameters, parameters):
            y = np.asarray(self._model.simulate(parameters, self._times_all))
            self._cached_output = y
            self._cached_parameters = parameters
        return self._output_sorter(self._cached_output, index)

    def _evaluateS1(self, parameters, index):
        """ Evaluates model with sensitivities or returns cached result. """
        parameters = pints.vector(parameters)

        # extra or here catches if evaluate has been called before evaluateS1
        if (not np.array_equal(self._cached_parameters, parameters) or
                self._cached_sensitivities is None):
            y, dy = self._model.simulateS1(parameters, self._times_all)
            self._cached_output = y
            self._cached_sensitivities = dy
            self._cached_parameters = parameters
        return self._output_and_sensitivity_sorter(
            self._cached_output, self._cached_sensitivities, index)

    def model(self):
        """ Returns forward model. """
        return self._model

    def subproblem(self, index):
        """
        Creates a `pints.SubProblem` corresponding to a particular output
        index.
        """
        if index >= self._n_output_sets:
            raise ValueError('Index must be less than number of output sets.')
        return pints.SubProblem(self, index)

    def timeses(self):
        """ Returns list of times sequences: one for each output chunk. """
        return self._timeses

    def valueses(self):
        """ Returns list of value chunks: one for each output chunk. """
        return self._valueses


class SubProblem(object):
    """
    Represents an inference problem for a subset of outputs from a multi-output
    model. This is likely to be used either when the measurement times across
    outputs are differ or when different outputs require different objective
    functions (i.e. log-likelihoods or score functions).

    Parameters
    ----------
    collection
        An object of :class:`ProblemCollection`.
    index
        An integer index corresponding to the particular output chunk in the
        collection.
    """
    def __init__(self, collection, index):

        # Get items from collection
        self._collection = collection
        self._index = index
        model = collection.model()
        self._model = model
        timeses = collection.timeses()
        self._times = pints.vector(timeses[index])
        values = collection.valueses()
        values = values[index]

        self._n_parameters = int(model.n_parameters())
        self._n_times = len(self._times)

        values = np.array(values)
        values_shape = values.shape

        # here don't check array sizes as this will be done in the
        # problemcollection.subproblem method
        if len(values_shape) == 1:
            self._n_outputs = 1

            # copy so that they can no longer be changed
            self._values = pints.vector(values)

        else:
            self._n_outputs = values_shape[1]
            self._values = pints.matrix2d(values)

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        """
        return self._collection._evaluate(parameters, self._index)

    def evaluateS1(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        """
        return self._collection._evaluateS1(parameters, self._index)

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
