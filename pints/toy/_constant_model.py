#
# Constant model with multiple outputs.
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


class ConstantModel(pints.ForwardModel):
    """
    *Extends:* :class:`pints.ForwardModel`.

    Toy model that's constant over time, linear over the parameters.

    For an `n`-dimensional model, evaluated with parameters
    ``p = [p_1, p_2, ..., p_n]``, the simulated value at each time t equals:

    .. math::
        f(t) = (p_1, p_2, ..., p_n)

    By default, the simulated output will have shape ``(n_times, )`` if
    ``n == 1`` and shape ``(n_times, n_outputs)`` if ``n > 1``. This can be
    tweaked using ``force_multi_output``.

    This model is mostly useful for unit testing.

    Arguments:

    ``n``
        The number of parameters (and outputs) the model should have.
    ``force_multi_output``
        Set to ``True`` to always return output of the shape
        ``(n_times, n_outputs)``, even if ``n_outputs == 1``.

    Example::

        times = np.linspace(0, 1, 100)
        m = pints.ConstantModel(2)
        m.simulate([1, 2], times)

    In this example, the returned output is ``[1, 2]`` at every point in time.

    """

    def __init__(self, n, force_multi_output=False):
        super(ConstantModel, self).__init__()

        self._n = int(n)
        if self._n < 1:
            raise ValueError('Number of parameters must be 1 or greater.')

        # Reshape for single-output models?
        self._reshape = (n == 1 and not force_multi_output)

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return self._n

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.outputs()`. """
        return self._n

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """

        # Check input
        parameters = np.asarray(parameters)
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if len(parameters) != self._n:
            raise ValueError('Expected ' + str(self._n) + ' parameters.')
        if not np.all(np.isfinite(parameters)):
            raise ValueError('All parameters must be finite.')

        # Calculate

        out = parameters.reshape((1, self._n)).repeat(len(times), axis=0)
        if self._reshape:
            out = out.reshape((len(times), ))
        return out

