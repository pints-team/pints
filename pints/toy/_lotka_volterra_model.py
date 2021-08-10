#
# Lotka-Volterra model of Predatory-Prey relationships.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints
from . import ToyODEModel


class LotkaVolterraModel(ToyODEModel, pints.ForwardModelS1):
    """
    Lotka-Volterra model of Predatory-Prey relationships [1]_.

    This model describes cyclical fluctuations in the populations of two
    interacting species.

    .. math::
        \\frac{dx}{dt} = ax - bxy \\\\
        \\frac{dy}{dt} = -cy + dxy

    where ``x`` is the number of prey, and ``y`` is the number of predators.

    Real data is included via :meth:`suggested_values`, which was taken from
    [2]_, and includes hare and lynx pelt count data collected by the Hudson's
    Bay Company, in Canada in the early twentieth century.

    Extends :class:`pints.ForwardModelS1`, :class:`pints.toy.ToyODEModel`.

    Parameters
    ----------
    y0
        The initial population, given as a vector ``[a, b]`` such that
        ``a >= 0`` and ``b >= 0``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Lotka-Volterra_equations
    .. [2] Howard, P. (2009). Modeling basics. Lecture Notes for Math 442,
           Texas A&M University
    """

    def __init__(self, y0=None):
        if y0 is None:
            self.set_initial_conditions(np.log([30, 4]))
        else:
            self.set_initial_conditions(y0)

    def _dfdp(self, z, t, p):
        """ See :meth:`pints.ToyModel.jacobian()`. """
        x, y = z
        a, b, c, d = [float(param) for param in p]
        ret = np.empty((2, 4))
        ret[0, 0] = x
        ret[0, 1] = -x * y
        ret[0, 2] = 0
        ret[0, 3] = 0
        ret[1, 0] = 0
        ret[1, 1] = 0
        ret[1, 2] = -y
        ret[1, 3] = x * y
        return ret

    def jacobian(self, z, t, p):
        """ See :meth:`pints.ToyModel.jacobian()`. """
        x, y = z
        a, b, c, d = [float(param) for param in p]
        ret = np.empty((2, 2))
        ret[0, 0] = a - b * y
        ret[0, 1] = -b * x
        ret[1, 0] = d * y
        ret[1, 1] = -c + d * x
        return ret

    def initial_conditions(self):
        """
        Returns the current initial conditions.
        """
        return np.array(self._y0, copy=True)

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 2

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 4

    def _rhs(self, state, time, parameters):
        """
        Right-hand side equation of the ode to solve.
        """
        x, y = state
        a, b, c, d = parameters
        return np.array([a * x - b * x * y, -c * y + d * x * y])

    def set_initial_conditions(self, y0):
        """
        Changes the initial conditions for this model.
        """
        a, b = y0
        if a < 0 or b < 0:
            raise ValueError('Initial populations cannot be negative.')
        self._y0 = [a, b]

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([0.5, 0.15, 1.0, 0.3])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        return np.linspace(0, 20, 21)

    def suggested_values(self):
        """
        Returns hare-lynx pelt count data collected by the Hudson's Bay Company
        in Canada in the early twentieth century, which is taken from [2]_.
        The data given here corresponds to annual observations taken from
        1900-1920 (inclusive).
        """
        return np.array([
            [30.0, 4.0],  # 1900
            [47.2, 6.1],
            [70.2, 9.8],
            [77.4, 35.2],
            [36.3, 59.4],  # 1904
            [20.6, 41.7],
            [18.1, 19.0],
            [21.4, 13.0],
            [22.0, 8.3],
            [25.4, 9.1],  # 1909
            [27.1, 7.4],
            [40.3, 8.0],
            [57.0, 12.3],
            [76.6, 19.5],
            [52.3, 45.7],  # 1914
            [19.5, 51.1],
            [11.2, 29.7],
            [7.6, 15.8],
            [14.6, 9.7],
            [16.2, 10.1],
            [24.7, 8.1],  # 1920
        ])
