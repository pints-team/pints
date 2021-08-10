#
# Constant model with multiple outputs.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints


class ConstantModel(pints.ForwardModelS1):
    r"""
    Toy model that's constant over time, linear over the parameters, mostly
    useful for unit testing.

    For an `n`-dimensional model, evaluated with parameters
    ``p = [p_1, p_2, ..., p_n]``, the simulated values are time-invariant, so
    that for any time ``t``

    .. math::
        f(t) = (p_1, 2 p_2, 3 p_3, ..., n p_n)

    The derivatives with respect to the parameters are time-invariant, and
    simply equal

    .. math::

        \frac{\partial{f_i(t)}}{dp_j} =
            \begin{cases} i, i = j\\0, i \neq j \end{cases}

    Extends :class:`pints.ForwardModelS1`.

    Parameters
    ----------
    n : int
        The number of parameters (and outputs) the model should have.
    force_multi_output : boolean
        Set to ``True`` to always return output of the shape
        ``(n_times, n_outputs)``, even if ``n_outputs == 1``.

    Example
    -------
    ::

        times = np.linspace(0, 1, 100)
        m = pints.ConstantModel(2)
        m.simulate([1, 2], times)

    In this example, the returned output is ``[1, 4]`` at every point in time.
    """

    def __init__(self, n, force_multi_output=False):
        super(ConstantModel, self).__init__()

        n = int(n)
        if n < 1:
            raise ValueError('Number of parameters must be 1 or greater.')
        self._r = np.arange(1, 1 + n)
        self._n = n

        # Reshape for single-output models?
        self._reshape = (n == 1 and not force_multi_output)

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return self._n

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
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
        out = parameters.reshape((1, self._n)) * self._r
        out = out.repeat(len(times), axis=0)
        if self._reshape:
            out = out.reshape((len(times), ))
        return out

    def simulateS1(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulateS1()`. """
        y = self.simulate(parameters, times)
        if self._reshape:
            dy = np.ones(len(times))
        else:
            # Output has shape (times, outputs, parameters)
            # At every time point, there is a matrix:
            #  [[df1/dp1, df1/dp2],
            #   [df2/dp1, df2/dp2]]  (for 2d...)
            # i.e.
            #  [[df1/dp1, df1/dp2],
            #   [df2/dp1, df2/dp2]]
            # i.e.
            #  [[1, 0],
            #   [0, 2]]
            dy = np.tile(
                np.diag(np.arange(1, self._n + 1)), (len(times), 1, 1))
        return (y, dy)
