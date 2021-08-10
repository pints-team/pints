#
# Fitzhugh-Nagumo toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
#
import numpy as np

import pints
from . import ToyODEModel


class FitzhughNagumoModel(ToyODEModel, pints.ForwardModelS1):
    r"""
    Fitzhugh-Nagumo model of the action potential [1]_.

    Has two states, and three phenomenological parameters: ``a`` , ``b``,
    ``c``. All states are visible

    .. math::
        \frac{d \mathbf{y}}{dt} = \mathbf{f}(\mathbf{y},\mathbf{p},t)

    where

    .. math::
        \mathbf{y} &= (V,R)\\
        \mathbf{p} &= (a,b,c)

    The RHS, jacobian and change in RHS with the parameters are given by

    .. math::
        \begin{align}
        \mathbf{f}(\mathbf{y},\mathbf{p},t) &=
            \left[\begin{matrix}
                c \left(R - V^{3}/3+V\right) \\
                - \frac{1}{c} \left(R b + V - a\right)
            \end{matrix}\right] \\
        \frac{\partial \mathbf{f}}{\partial \mathbf{y}} &=
            \left[\begin{matrix}
                c \left(1- V^{2}\right) & c \\
                - \frac{1}{c} & - \frac{b}{c}
            \end{matrix}\right] \\
        \frac{\partial \mathbf{f}}{\partial \mathbf{p}} &=
            \left[\begin{matrix}
                0 & 0 & R - V^{3}/3 + V\\
                \frac{1}{c} & - \frac{R}{c} &
                    \frac{1}{c^{2}} \left(R b + V - a\right)
            \end{matrix}\right]
        \end{align}

    Extends :class:`pints.ForwardModelS1`, `pints.toy.ToyODEModel`.

    Parameters
    ----------
    y0
        The system's initial state. If not given, the default ``[-1, 1]`` is
        used.

    References
    ----------
    .. [1] A kinetic model of the conductance changes in nerve membrane
           Fitzhugh (1965) Journal of Cellular and Comparative Physiology.
           https://doi.org/10.1002/jcp.1030660518
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

    def _dfdp(self, y, t, p):
        """ See :meth:`pints.ToyODEModel._dfdp()`. """
        V, R = y
        a, b, c = [float(param) for param in p]
        ret = np.empty((2, 3))
        ret[0, 0] = 0
        ret[0, 1] = 0
        ret[0, 2] = R - V**3 / 3 + V
        ret[1, 0] = 1 / c
        ret[1, 1] = -R / c
        ret[1, 2] = (R * b + V - a) / c**2
        return ret

    def jacobian(self, y, t, p):
        """ See :meth:`pints.ToyODEModel.jacobian()`. """
        V, R = y
        a, b, c = [float(param) for param in p]
        ret = np.empty((2, 2))
        ret[0, 0] = c * (1 - V**2)
        ret[0, 1] = c
        ret[1, 0] = -1 / c
        ret[1, 1] = -b / c
        return ret

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 2

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 3

    def _rhs(self, y, t, p):
        """ See :meth:`pints.ToyODEModel._rhs()`. """
        V, R = y
        a, b, c = [float(x) for x in p]
        dV_dt = (V - V**3 / 3 + R) * c
        dR_dt = (V - a + b * R) / -c
        return dV_dt, dR_dt

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([0.1, 0.5, 3])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        return np.linspace(0, 20, 200)
