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
    Three-state Goodwin oscillator toy model [1].

    In this implementation of the model, only the first state is visible,
    making it very difficult to identify.

    [1] TODO TODO TODO TODO TODO Needs reference!
    """

    def dimension(self):
        """ See :meth:`pints.ForwardModel.dimension`. """
        return 5

    def _rhs(self, state, time, parameters):
        """
        Right-hand side equation of the ode to solve.
        """
        x, y, z = state
        a1, a2, alpha, k1, k2 = parameters
        dxdt = a1 / (1 + a2 * z**10) - alpha * x
        dydt = k1 * x - alpha * y
        dzdt = k2 * y - alpha * z
        return dxdt, dydt, dzdt

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate`. """
        y0 = [0, 0, 0]
        solution = scipy.integrate.odeint(
            self._rhs, y0, times, args=(parameters,))
        return solution[:, 0]   # Only observe the first state

    def suggested_parameters(self):
        """
        Returns a suggested array of parameter values.
        """
        return np.array([1.97, 0.15, 0.53, 0.46, 1.49])

    def suggested_times(self):
        """
        Returns a suggested set of sampling times.
        """
        return np.linspace(0, 50, 120)

