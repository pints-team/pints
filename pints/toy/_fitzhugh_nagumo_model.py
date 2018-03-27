#
# Logistic model.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints
from scipy.integrate import odeint


class FitzhughNagumoModel(pints.ForwardModel):
    """
    Fitzhugh Nagumo model of action potential.

    Has two states, and three phenomenological parameters: ``a`` , ``b``,
    ``c``. All states are visible

    Arguments:

    ``y0``
        The system's initial
    """
    def __init__(self, y0=None):
        super(FitzhughNagumoModel, self).__init__()

        # Check initial values
        if y0 is None:
            self._y0 = np.array([-1, 1], dtype=float)
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError('Initial value must have size 2.')

    def dimension(self):
        """ See :meth:`pints.ForwardModel.dimension()`. """
        return 3

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.outputs()`. """
        return 2

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        a, b, c = [float(x) for x in parameters]

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        def r(y, t, p):
            V, R = y
            dV_dt = (V - V**3 / 3 + R) * c
            dR_dt = (V - a + b * R) / -c
            return dV_dt, dR_dt

        return odeint(r, self._y0, times, (parameters,))
