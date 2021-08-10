#
# SIR Epidemiology toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
#
import numpy as np
import pints
from scipy.integrate import odeint

from . import ToyModel


class SIRModel(pints.ForwardModel, ToyModel):
    r"""
    The SIR model of infectious disease models the number of susceptible (S),
    infected (I), and recovered (R) people in a population [1]_, [2]_.

    The particular model given here is analysed in [3],_ and is described by
    the following three-state ODE:

    .. math::
        \dot{S} = -\gamma S I

        \dot{I} = \gamma S I - v I

        \dot{R} = v I

    Where the parameters are ``gamma`` (infection rate), and ``v``, recovery
    rate. In addition, we assume the initial value of S, ``S0``, is unknwon,
    leading to a three parameter model ``(gamma, v, S0)``.

    The number of infected people and recovered people are observable, making
    this a 2-output system. S can be thought of as an unknown number of
    susceptible people within a larger population.

    The model does not account for births and deaths, which are assumed to
    happen much slower than the spread of the (non-lethal) disease.

    Real data is included via :meth:`suggested_values`, which was taken from
    [3]_, [4]_, [5]_.

    Extends :class:`pints.ForwardModel`, `pints.toy.ToyModel`.

    Parameters
    ----------
    y0
        The system's initial state, must have 3 entries all >=0.

    References
    ----------
    .. [1] A Contribution to the Mathematical Theory of Epidemics. Kermack,
           McKendrick (1927) Proceedings of the Royal Society A.
           https://doi.org/10.1098/rspa.1927.0118

    .. [2] https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

    .. [3] Approximate Bayesian computation scheme for parameter inference and
           model selection in dynamical systems. Toni, Welch, Strelkowa, Ipsen,
           Stumpf (2009) J. R. Soc. Interface.
           https://doi.org/10.1098/rsif.2008.0172

    .. [4] A mathematical model of common-cold epidemics on Tristan da Cunha.
           Hammond, Tyrrell (1971) Epidemiology & Infection.
           https://doi.org/10.1017/S0022172400021677

    .. [5] Common colds on Tristan da Cunha. Shybli, Gooch, Lewis, Tyrell
           (1971) Epidemiology & Infection.
           https://doi.org/10.1017/S0022172400021483
    """

    def __init__(self, y0=None):
        super(SIRModel, self).__init__()

        # Check initial values
        if y0 is None:
            # Toni et al.:
            self._y0 = np.array([38, 1, 0])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 3:
                raise ValueError('Initial value must have size 3.')
            if np.any(self._y0 < 0):
                raise ValueError('Initial states can not be negative.')

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 2

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 3

    def _rhs(self, y, t, gamma, v):
        """
        Calculates the model RHS.
        """
        dS = -gamma * y[0] * y[1]
        dI = gamma * y[0] * y[1] - v * y[1]
        dR = v * y[1]
        return np.array([dS, dI, dR])

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        gamma, v, S0 = parameters
        y0 = np.array(self._y0, copy=True)
        y0[0] = S0
        y = odeint(self._rhs, y0, times, (gamma, v))
        return y[:, 1:]

    def suggested_parameters(self):
        """
        Returns a suggested set of parameters for this toy model.
        """
        # Guesses based on Toni et al.:
        return [0.026, 0.285, 38]

    def suggested_times(self):
        """
        Returns a suggested set of simulation times for this toy model.
        """
        # Toni et al.:
        return np.arange(1, 22)

    def suggested_values(self):
        """
        Returns the data from a common-cold outbreak on the remote island of
        Tristan da Cunha, as given in [3]_, [4]_, [5]_.
        """
        # Toni et al.
        return np.array([
            [1, 0],     # day 1
            [1, 0],
            [3, 0],
            [7, 0],
            [6, 5],     # day 5
            [10, 7],
            [13, 8],
            [13, 13],
            [14, 13],
            [14, 16],    # day 10
            [17, 17],
            [10, 24],
            [6, 30],
            [6, 31],
            [4, 33],    # day 15
            [3, 34],
            [1, 36],
            [1, 36],
            [1, 36],
            [1, 36],    # day 20
            [0, 37],    # day 21
        ])
