#
# Three-state Goodwin oscillator toy model.
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


class GoodwinOscillatorModel(pints.ForwardModel, ToyModel):
    r"""
    Three-state Goodwin oscillator toy model introduced in [1]_, [2]_, but
    best described in [3]_. The model considers level of mRNA, ``x``, which
    is translated into protein ``y``, which, in turn, stimulated production of
    protein ``z`` that inhibits production of mRNA. The ODE system is described
    by the following equations,

    .. math::
        \dot{x} = 1 / (1 + z^{10}) - m1 x

        \dot{y} = k2 x - m2 y

        \dot{z} = k3 y - m3 z

    Parameters are ``[k2, k3, m1, m2, m3]``.

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    References
    ----------
    .. [1] Oscillatory behavior in enzymatic control processes.
           Goodwin (1965) Advances in enzyme regulation.
           https://doi.org/10.1016/0065-2571(65)90067-1

    .. [2] Mathematics of cellular control processes I. Negative feedback to
           one gene. Griffith (1968) Journal of theoretical biology.
           https://doi.org/10.1016/0022-5193(68)90189-6

    .. [3] Estimating Bayes factors via thermodynamic integration and
           population MCMC. Ben Calderhead and Mark Girolami, 2009,
           Computational Statistics and Data Analysis.
    """

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 3

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
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
        return solution

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """

        return np.array([2, 4, 0.12, 0.08, 0.1])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """

        return np.linspace(0, 100, 200)
