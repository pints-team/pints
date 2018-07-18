#
# Beeler-Reuter model for mammalian ventricular action potential
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import print_function
import pints
import numpy as np
import scipy.integrate


class ActionPotentialModel(pints.ForwardModel):
    """
    *Extends:* :class:`pints.ForwardModel`.

    The Beeler-Reuter model for mammalian ventricular action potential [1].

    The action potential (cell's transmembrane voltage) model contains
    multiple currents which their magnitude is determined by the maximum-
    conductance while their shape is controlled by other parameters. In this
    simplified (but not trivial) action potential toy model, we define the
    maximum-conductance values of all the currents as the parameter of
    interest, and assume all other parameters of the model are known and well-
    defined. We also define the maximum-conductance values in logarithmic
    scale.

    The observables of this model are the transmembrane voltage (the action
    potential itsefl) and the calcium concentration of the cell (also known as
    the calcium transient).

    References:

    [1] Reconstruction of the action potential of ventricular myocardial
    fibres
    Beeler, Reuter (1977) Journal of Physiology

    Arguments:

    ``y0``
        (Optional) The initial condition of the observables ``v``, ``cai`` and
        requires ``cai0 >= 0``.
    ``implicit_parameters``
        (Optional) The implicit parameter of the model that is not inferred,
        given as a vector ``[m0, h0, j0, d0, f0, x10, C_m, E_Na, I_Stim_amp,
        I_Stim_period, I_Stim_length]``. All implicit parameters have to be
        greater than zero except ``E_Na``.
    """
    def __init__(self, y0=None, implicit_parameters=None):
        if y0 is None:
            self.set_initial_conditions([-84.622, 2e-7])
        else:
            self.set_initial_conditions(y0)
        if implicit_parameters is None:
            self.set_implicit_parameters(self.suggested_implicit_parameters())
        else:
            self.set_implicit_parameters(implicit_parameters)

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        # 5 conductance values
        return 5

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        # membrane voltage and calcium concentration
        return 2

    def _rhs(self, states, time, parameters):
        """
        Right-hand side equation of the ode to solve.
        """
        # Set-up
        V, Cai, m, h, j, d, f, x1 = states
        gNaBar, gNaC, gCaBar, gK1Bar, gx1Bar = np.exp(parameters)
        # Equations
        # INa
        INa = (gNaBar * m**3 * h * j + gNaC) * (V - self._E_Na)
        alpha = (V + 47) / (1 - np.exp(-0.1 * (V + 47)))
        beta = 40 * np.exp(-0.056 * (V + 72))
        dmdt = alpha * (1 - m) - beta * m
        alpha = 0.126 * np.exp(-0.25 * (V + 77))
        beta = 1.7 / (1 + np.exp(-0.082 * (V + 22.5)))
        dhdt = alpha * (1 - h) - beta * h
        alpha = 0.055 * np.exp(-0.25 * (V + 78)) \
            / (1 + np.exp(-0.2 * (V + 78)))
        beta = 0.3 / (1 + np.exp(-0.1 * (V + 32)))
        djdt = alpha * (1 - j) - beta * j
        # ICa
        E_Ca = -82.3 - 13.0287 * np.log(Cai)
        ICa = gCaBar * d * f * (V - E_Ca)
        alpha = 0.095 * np.exp(-0.01 * (V + -5)) \
            / (np.exp(-0.072 * (V + -5)) + 1)
        beta = 0.07 * np.exp(-0.017 * (V + 44)) \
            / (np.exp(0.05 * (V + 44)) + 1)
        dddt = alpha * (1 - d) - beta * d
        alpha = 0.012 * np.exp(-0.008 * (V + 28)) \
            / (np.exp(0.15 * (V + 28)) + 1)
        beta = 0.0065 * np.exp(-0.02 * (V + 30)) \
            / (np.exp(-0.2 * (V + 30)) + 1)
        dfdt = alpha * (1 - f) - beta * f
        # Cai
        dCaidt = -1e-7 * ICa + 0.07 * (1e-7 - Cai)
        # IK1
        IK1 = gK1Bar * (
            4 * (np.exp(0.04 * (V + 85)) - 1)
            / (np.exp(0.08 * (V + 53)) + np.exp(0.04 * (V + 53)))
            + 0.2 * (V + 23)
            / (1 - np.exp(-0.04 * (V + 23)))
        )
        # IX1
        Ix1 = gx1Bar * x1 * (np.exp(0.04 * (V + 77)) - 1) \
            / np.exp(0.04 * (V + 35))
        alpha = 0.0005 * np.exp(0.083 * (V + 50)) \
            / (np.exp(0.057 * (V + 50)) + 1)
        beta = 0.0013 * np.exp(-0.06 * (V + 20)) \
            / (np.exp(-0.04 * (V + 333)) + 1)
        dx1dt = alpha * (1 - x1) - beta * x1
        # I_Stim
        if (time % self._I_Stim_period) < self._I_Stim_length:
            IStim = self._I_Stim_amp
        else:
            IStim = 0
        # V
        dVdt = -(1 / self._C_m) * (IK1 + Ix1 + INa + ICa - IStim)
        # Output
        output = np.array([dVdt,
                           dCaidt,
                           dmdt,
                           dhdt,
                           djdt,
                           dddt,
                           dfdt,
                           dx1dt])
        return output

    def set_initial_conditions(self, y0):
        """
        Changes the initial conditions for this model.
        """
        if y0[1] < 0:
            raise ValueError('Initial condition of ``cai`` cannot be'
                             ' negative.')
        self._v0 = y0[0]
        self._cai0 = y0[1]

    def set_implicit_parameters(self, k):
        """
        Changes the implicit parameters for this model.
        """
        if not (k[:7] > 0).all() or not (k[8:] > 0).all():
            raise ValueError('Implicit parameters cannot be negative except'
                             ' ``E_Na``.')
        # Initial condition for non-observable states
        self._m0 = k[0]
        self._h0 = k[1]
        self._j0 = k[2]
        self._d0 = k[3]
        self._f0 = k[4]
        self._x10 = k[5]
        # membrane capacitance, in uF/cm^2
        self._C_m = k[6]
        # Nernst reversal potentials, in mV
        self._E_Na = k[7]
        # Stimulus current
        self._I_Stim_amp = k[8]
        self._I_Stim_period = k[9]
        self._I_Stim_length = k[10]

    def initial_conditions(self):
        """
        Returns the initial conditions of this model.
        """
        return [self._v0, self._cai0]

    def implicit_parameters(self):
        """
        Returns the implicit parameters of this model.
        """
        return [self._m0, self._h0, self._j0, self._d0, self._f0, self._x10,
                self._C_m, self._E_Na]

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        y0 = [self._v0,
              self._cai0,
              self._m0,
              self._h0,
              self._j0,
              self._d0,
              self._f0,
              self._x10]
        solved_states = scipy.integrate.odeint(
            self._rhs, y0, times, args=(parameters,), hmax=self._I_Stim_length)
        # Only return the observable (V, Cai)
        return solved_states[:, 0:2]

    def simulate_all_states(self, parameters, times):
        """
        Returns all state variables that ``simulate()`` does not return.
        """
        y0 = [self._v0,
              self._cai0,
              self._m0,
              self._h0,
              self._j0,
              self._d0,
              self._f0,
              self._x10]
        solved_states = scipy.integrate.odeint(
            self._rhs, y0, times, args=(parameters,), hmax=self._I_Stim_length)
        # Return all states
        return solved_states

    def suggested_parameters(self):
        """
        Returns a suggested array of parameter values.
        """
        # maximum conducances, in mS/cm^2
        g_Na = 4.0
        g_NaC = 0.003
        g_Ca = 0.09
        g_K1 = 0.35
        g_x1 = 0.8
        return np.log([g_Na, g_NaC, g_Ca, g_K1, g_x1])

    def suggested_implicit_parameters(self):
        """
        Returns a suggested array of implicit parameter values.
        """
        # membrane capacitance, in uF/cm^2
        C_m = 1.0
        # Nernst reversal potentials, in mV
        E_Na = 50.0
        # Stimulus current, in uA/cm^2
        I_Stim_amp = 25.0
        I_Stim_period = 1000.0
        I_Stim_length = 2.0
        # Initial condition for non-observable states
        m0 = 0.01
        h0 = 0.99
        j0 = 0.98
        d0 = 0.003
        f0 = 0.99
        x10 = 0.0004
        return np.array([m0, h0, j0, d0, f0, x10, C_m, E_Na, I_Stim_amp,
                         I_Stim_period, I_Stim_length])

    def suggested_times(self):
        """
        Returns a suggested set of sampling times.
        """
        return np.arange(0, 500, 0.1)

