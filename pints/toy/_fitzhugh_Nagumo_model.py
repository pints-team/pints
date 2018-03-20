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
from scipy.integrate import odeint
import pints


class FitzhughNagumoModel(pints.ForwardModel):
    """
    Fitzhugh Nagumo model of action potential.

    Has three phenomenological parameters: ``a`` , ``b``, ``c``.
    """
    _stateDim = 2
    _dimension = 3
    def __init__(self, init_vals=[-1.,1.]):
        super(FitzhughNagumoModel, self).__init__()

        if init_vals == None:
            self._y0 = np.array([-1.,1.])
        else:
            self._y0 = np.array(init_vals)
    def dimension(self):
        return self._dimension

    def stateDimension(self):
        return self._stateDim

    def simulate(self, parameters, times):
        a, b, c = [float(x) for x in parameters]
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        y0 = self._y0
        def r(y, t, p):
            V, R = y
            dV_dt=(V-((V**3)/3) + R)*c
            dR_dt=-(V-a+b*R)/c
            return dV_dt,dR_dt
        sol = odeint(r, y0, times, (parameters,))
        assert(sol.shape==(len(times),self._stateDim))
        return sol
