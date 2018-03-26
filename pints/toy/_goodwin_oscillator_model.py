#
# Three-state Goodwin oscillator toy model.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import print_function
import pints
import numpy as np
import scipy


class GoodwinOscillatorModel(pints.ForwardModel):
    """
    *Extends:* :class:`pints.ForwardModel`.

    Three-state Goodwin oscillator toy model [1, 2].

    In this implementation of the model, only the last state is visible,
    making it very difficult to identify.

    [1] Oscillatory behavior in enzymatic control processes."
    Goodwin (1965) Advances in enzyme regulation.

    [2] Mathematics of cellular control processes I. Negative feedback to one
    gene. Griffith (1968) Journal of theoretical biology.
    """

    def dimension(self):
        """ See :meth:`pints.ForwardModel.dimension()`. """
        return 5

    def _rhs(self, state, time, parameters):
        """
        Right-hand side equation of the ode to solve.
        """
        x, y, z = state
        k2, k3, m1, m2, m3 = parameters
        dxdt = 1 / (1 + z**10) - m1 * x
        dydt = k2 * x - m2 * y
        dzdt = k3 * y - m3 * z
        return dxdt, dydt, dzdt

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        y0 = [0.0054, 0.053, 1.93]
        solution = scipy.integrate.odeint(
            self._rhs, y0, times, args=(parameters,))
        return solution[:, -1]   # Only observe the last state

    def suggested_parameters(self):
        """
        Returns a suggested array of parameter values.
        """
        return np.array([2, 4, 0.12, 0.08, 0.1])

    def suggested_times(self):
        """
        Returns a suggested set of sampling times.
        """
        return np.linspace(0, 100, 200)

