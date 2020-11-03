#
# HES1 Michaelis-Menten model of regulatory dynamics.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import print_function
import numpy as np
import pints
import scipy

from . import ToyModel


class Hes1Model(pints.ForwardModel, ToyModel):
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

    References
    ----------
    .. [1] Silk, D., el al. 2011. Designing attractive models via automated
           identification of chaotic and oscillatory dynamical regimes. Nature
           communications, 2, p.489.
           https://doi.org/10.1038/ncomms1496

    Parameters
    ----------
    y0 : float
        The initial condition of the observable. Requires ``y0 >= 0``.
    implicit_parameters
        The implicit parameter of the model that is not inferred, given as a
        vector ``[p1_0, p2_0, k_deg]`` with ``p1_0, p2_0, k_deg >= 0``.
    """
    def __init__(self, y0=None, implicit_parameters=None):
        if y0 is None:
            self.set_initial_conditions(2)
        else:
            self.set_initial_conditions(y0)
        if implicit_parameters is None:
            self.set_implicit_parameters([5., 3., 0.03])
        else:
            self.set_implicit_parameters(implicit_parameters)

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

    def set_initial_conditions(self, y0):
        """
        Changes the initial conditions for this model.
        """
        if y0 < 0:
            raise ValueError('Initial condition cannot be negative.')
        self._y0 = y0

    def set_implicit_parameters(self, k):
        """
        Changes the implicit parameters for this model.
        """
        a, b, c = k
        if a < 0 or b < 0 or c < 0:
            raise ValueError('Implicit parameters cannot be negative.')
        self._p0 = [a, b]
        self._kdeg = c

    def initial_conditions(self):
        """
        Returns the initial conditions of this model.
        """
        return self._y0

    def implicit_parameters(self):
        """
        Returns the implicit parameters of this model.
        """
        return [self._p0[0], self._p0[1], self._kdeg]

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        y0 = [self._y0, self._p0[0], self._p0[1]]
        solved_states = scipy.integrate.odeint(
            self._rhs, y0, times, args=(parameters,))
        # Only return the observable
        return solved_states[:, 0]

    def simulate_all_states(self, parameters, times):
        """
        Returns all state variables that ``simulate()`` does not return.
        """
        y0 = [self._y0, self._p0[0], self._p0[1]]
        solved_states = scipy.integrate.odeint(
            self._rhs, y0, times, args=(parameters,))
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

