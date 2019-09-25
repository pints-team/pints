#
# Lotka-Volterra model of Predatory-Prey relationships.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import print_function
import numpy as np
import pints
import scipy

from . import ToyModel


class LotkaVolterraModel(pints.ForwardModel, ToyModel):
    """
    Lotka-Volterra model of Predatory-Prey relationships [1]_.

    This model describes cyclical fluctuations in the populations of two
    interacting species.

    .. math::
        \\frac{dx}{dt} = ax - bxy \\\\
        \\frac{dy}{dt} = -cy + dxy

    where ``x`` is the number of prey, and ``y`` is the number of predators.

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    y0
        The initial population, given as a vector ``[a, b]`` such that
        ``a >= 0`` and ``b >= 0``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
    """
    def __init__(self, y0=None):
        if y0 is None:
            self.set_initial_conditions([2, 2])
        else:
            self.set_initial_conditions(y0)

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

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        return scipy.integrate.odeint(
            self._rhs, self._y0, times, args=(parameters,))

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([3, 2, 3, 2])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        return np.linspace(0, 3, 300)

