#
# HES1 Michaelis-Menten model of regulatory dynamics.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints
import scipy
from . import ToyODEModel


class Hes1Model(ToyODEModel, pints.ForwardModelS1):
    """
    HES1 Michaelis-Menten model of regulatory dynamics [1]_.

    This model describes the expression level of the transcription factor
    Hes1.

    .. math::
        \\frac{dm}{dt} &= -k_{deg}m + \\frac{1}{1 + (p_2/P_0)^h} \\\\
        \\frac{dp_1}{dt} &= -k_{deg} p_1 + \\nu m - k_1 p_1 \\\\
        \\frac{dp_2}{dt} &= -k_{deg} p_2 + k_1 p_1

    The system is determined by 3 state variables :math:`m`, :math:`p_1`, and
    :math:`p_2`. It is assumed that only :math:`m` can be observed, that is
    only :math:`m` is an observable. The initial condition of the other two
    state variables and :math:`k_{deg}` are treated as implicit parameters of
    the system. The input order of parameters of interest is
    :math:`\\{ P_0, \\nu, k_1, h \\}`.

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    m0 : float
        The initial condition of the observable ``m``. Requires ``m0 >= 0``.
    fixed_parameters
        The fixed parameters of the model which are not inferred, given as a
        vector ``[p1_0, p2_0, k_deg]`` with ``p1_0, p2_0, k_deg >= 0``.

    References
    ----------
    .. [1] Silk, D., el al. 2011. Designing attractive models via automated
           identification of chaotic and oscillatory dynamical regimes. Nature
           communications, 2, p.489.
           https://doi.org/10.1038/ncomms1496
    """
    def __init__(self, m0=None, fixed_parameters=None):
        if fixed_parameters is None:
            self.set_fixed_parameters([5., 3., 0.03])
        else:
            self.set_fixed_parameters(fixed_parameters)
        if m0 is None:
            self.set_m0(2)
        else:
            self.set_m0(m0)

    def _dfdp(self, state, time, parameters):
        """ See :meth:`pints.ToyModel.jacobian()`. """
        m, p1, p2 = state
        P0, v, k1, h = parameters
        p2_over_p0 = p2 / P0
        p2_over_p0_h = p2_over_p0**h
        one_plus_p2_expression_sq = (1 + p2_over_p0_h)**2
        ret = np.empty((self.n_states(), self.n_parameters()))
        ret[0, 0] = h * p2 * p2_over_p0**(h - 1) / (
            P0**2 * one_plus_p2_expression_sq)
        ret[0, 1] = 0
        ret[0, 2] = 0
        ret[0, 3] = - (p2_over_p0_h * np.log(p2_over_p0)) / (
            one_plus_p2_expression_sq
        )
        ret[1, 0] = 0
        ret[1, 1] = m
        ret[1, 2] = -p1
        ret[1, 3] = 0
        ret[2, 0] = 0
        ret[2, 1] = 0
        ret[2, 2] = p1
        ret[2, 3] = 0
        return ret

    def m0(self):
        """
        Returns the initial conditions of the ``m`` variable.
        """
        return self._y0[0]

    def fixed_parameters(self):
        """
        Returns the fixed parameters of the model which are not inferred, given
        as a vector ``[p1_0, p2_0, k_deg]``.
        """
        return [self._p0[0], self._p0[1], self._kdeg]

    def jacobian(self, state, time, parameters):
        """ See :meth:`pints.ToyModel.jacobian()`. """
        m, p1, p2 = state
        P0, v, k1, h = parameters
        k_deg = self._kdeg
        p2_over_p0 = p2 / P0
        p2_over_p0_h = p2_over_p0**h
        one_plus_p2_expression_sq = (1 + p2_over_p0_h)**2
        ret = np.zeros((self.n_states(), self.n_states()))
        ret[0, 0] = -k_deg
        ret[0, 1] = 0
        ret[0, 2] = -h * p2_over_p0**(h - 1) / (P0 * one_plus_p2_expression_sq)
        ret[1, 0] = v
        ret[1, 1] = -k1 - k_deg
        ret[1, 2] = 0
        ret[2, 0] = 0
        ret[2, 1] = k1
        ret[2, 2] = -k_deg
        return ret

    def n_states(self):
        """ See :meth:`pints.ToyODEModel.n_states()`. """
        return 3

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 1

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 4

    def _rhs(self, state, time, parameters):
        """
        Right-hand side equation of the ode to solve.
        """
        m, p1, p2 = state
        P0, v, k1, h = parameters
        output = np.array([
            - self._kdeg * m + 1. / (1. + (p2 / P0)**h),
            - self._kdeg * p1 + v * m - k1 * p1,
            - self._kdeg * p2 + k1 * p1])
        return output

    def set_m0(self, m0):
        """
        Sets the initial conditions of the ``m`` variable.
        """
        if m0 < 0:
            raise ValueError('Initial condition cannot be negative.')
        y0 = [m0, self._p0[0], self._p0[1]]
        super(Hes1Model, self).set_initial_conditions(y0)

    def set_fixed_parameters(self, k):
        """
        Changes the implicit parameters for this model.
        """
        a, b, c = k
        if a < 0 or b < 0 or c < 0:
            raise ValueError('Implicit parameters cannot be negative.')
        self._p0 = [a, b]
        self._kdeg = c

    def simulate_all_states(self, parameters, times):
        """
        Returns all state variables that ``simulate()`` does not return.
        """
        solved_states = scipy.integrate.odeint(
            self._rhs, self._y0, times, args=(parameters,))
        # Return all states
        return solved_states

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([2.4, 0.025, 0.11, 6.9])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        return np.arange(0, 270, 30)

    def suggested_values(self):
        """
        Returns a suggested set of values that matches
        :meth:`suggested_times()`.
        """
        return np.array([2, 1.20, 5.90, 4.58, 2.64, 5.38, 6.42, 5.60, 4.48])
