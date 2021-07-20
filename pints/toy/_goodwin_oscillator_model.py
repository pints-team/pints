#
# Three-state Goodwin oscillator toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints
from . import ToyODEModel


class GoodwinOscillatorModel(ToyODEModel, pints.ForwardModelS1):
    r"""
    Three-state Goodwin oscillator toy model introduced in [1]_, [2]_, but
    best described in [3]_. The model considers level of mRNA, :math:`x`, which
    is translated into protein :math:`y`, which, in turn, stimulated production
    of protein :math:`z` that inhibits production of mRNA. The ODE system is
    described by the following equations,

    .. math::
        \dot{x} = 1 / (1 + z^{10}) - m_1 x

        \dot{y} = k_2 x - m_2 y

        \dot{z} = k_3 y - m_3 z

    Parameters are :math:`[k_2, k_3, m_1, m_2, m_3]`. The initial conditions
    are hard-coded at ``[0.0054, 0.053, 1.93]``.

    Extends :class:`pints.ForwardModelS1`, :class:`pints.toy.ToyODEModel`.

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
    def __init__(self):
        super(GoodwinOscillatorModel, self).__init__()
        self._y0 = [0.0054, 0.053, 1.93]

    def _dfdp(self, state, time, parameters):
        """ See :meth:`pints.ToyODEModel._dfdp()`. """
        x, y, z = state
        k2, k3, m1, m2, m3 = parameters
        ret = np.empty((self.n_outputs(), self.n_parameters()))
        ret[0, 0] = 0
        ret[0, 1] = 0
        ret[0, 2] = -x
        ret[0, 3] = 0
        ret[0, 4] = 0
        ret[1, 0] = x
        ret[1, 1] = 0
        ret[1, 2] = 0
        ret[1, 3] = -y
        ret[1, 4] = 0
        ret[2, 0] = 0
        ret[2, 1] = y
        ret[2, 2] = 0
        ret[2, 3] = 0
        ret[2, 4] = -z
        return ret

    def jacobian(self, state, time, parameters):
        """ See :meth:`pints.ToyODEModel.jacobian()`. """
        x, y, z = state
        k2, k3, m1, m2, m3 = parameters
        ret = np.empty((self.n_outputs(), self.n_outputs()))
        ret[0, 0] = -m1
        ret[0, 1] = 0
        ret[0, 2] = -10 * z**9 / ((1 + z**10)**2)
        ret[1, 0] = k2
        ret[1, 1] = -m2
        ret[1, 2] = 0
        ret[2, 0] = 0
        ret[2, 1] = k3
        ret[2, 2] = -m3
        return ret

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 3

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 5

    def _rhs(self, state, time, parameters):
        """ See :meth:`pints.ToyODEModel._rhs()`. """
        x, y, z = state
        k2, k3, m1, m2, m3 = parameters
        dxdt = 1 / (1 + z**10) - m1 * x
        dydt = k2 * x - m2 * y
        dzdt = k3 * y - m3 * z
        return dxdt, dydt, dzdt

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([2, 4, 0.12, 0.08, 0.1])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        return np.linspace(0, 100, 200)
