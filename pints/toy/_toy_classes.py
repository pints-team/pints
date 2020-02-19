#
# Toy base classes.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from scipy.integrate import odeint


class ToyLogPDF(pints.LogPDF):
    """
    Abstract base class for toy distributions.

    Extends :class:`pints.LogPDF`.
    """

    def distance(self, samples):
        """
        Calculates a measure of distance from ``samples`` to some
        characteristic of the underlying distribution.
        """
        raise NotImplementedError

    def sample(self, n_samples):
        """
        Generates independent samples from the underlying distribution.
        """
        raise NotImplementedError

    def suggested_bounds(self):
        """
        Returns suggested boundaries for prior.
        """
        raise NotImplementedError


class ToyModel(object):
    """
    Defines an interface for toy problems.

    Note that toy models should extend both ``ToyModel`` and one of the forward
    model classes, e.g. :class:`pints.ForwardModel`.
    """
    def suggested_parameters(self):
        """
        Returns an numpy array of the parameter values that are representative
        of the model.

        For example, these parameters might reproduce a particular result that
        the model is famous for.
        """
        raise NotImplementedError

    def suggested_times(self):
        """
        Returns an numpy array of time points that is representative of the
        model
        """
        raise NotImplementedError

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        raise NotImplementedError


class ToyODEModel(ToyModel):
    """
    Defines an interface for toy problems where the underlying model is an
    ordinary differential equation (ODE).

    Note that toy ODE models should extend both ``ToyModel`` and one of the
    forward model classes, e.g. :class:`pints.ForwardModel`.
    """
    def _dfdp(self, y, t, p):
        """
        Returns the derivative of the ODE RHS with respect to parameters, this
        should be a matrix of dimensions ``n_parameters`` by ``n_parameters``.

        Parameters
        ----------
        y
            State vector
        t
            Time
        p
            Model parameters
        """
        raise NotImplementedError

    def jacobian(self, y, t, p):
        """
        Returns the Jacobian (the derivative of the RHS ODE with respect to the
        outputs), this should be a matrix of dimensions ``n_outputs`` by
        ``n_outputs``.

        Parameters
        ----------
        y
            State vector
        t
            Time
        p
            Model parameters
        """
        raise NotImplementedError

    def n_states(self):
        """
        Returns number of states in underlying ODE.
        Note:will not be same as ``n_outputs()`` for models where only a subset
        of states are observed.
        """
        return self.n_outputs()

    def _rhs(self, y, t, p):
        """
        Returns RHS of ODE for numerical integration to obtain outputs and
        sensitivities, this should be a vector of length ``n_outputs``.

        Parameters
        ----------
        y
            State vector
        t
            Time
        p
            Model parameters
        """
        raise NotImplementedError

    def _rhs_S1(self, y_and_dydp, t, p):
        """
        Forms the RHS of ODE for numerical integration to obtain both outputs
        and sensitivities, this should be a vector of length
        ``n_outputs + n_parameters``.

        Parameters
        ----------
        y_and_dydp
            Combined vector of states (elements ``0`` to ``n_states - 1``) and
            sensitivities (elements ``n_states`` onwards)
        t
            Time
        p
            Model parameters
        """
        n_outputs = self.n_states()
        y = y_and_dydp[0:n_outputs]
        dydp = y_and_dydp[n_outputs:].reshape((n_outputs, self.n_parameters()))
        dydt = self._rhs(y, t, p)
        d_dydp_dt = (
            np.matmul(self.jacobian(y, t, p), dydp) +
            self._dfdp(y, t, p))
        return np.concatenate((dydt, d_dydp_dt.reshape(-1)))

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        return self._simulate(parameters, times, False)

    def _simulate(self, parameters, times, sensitivities):
        """
        Private helper function that uses ``scipy.integrate.odeint`` to
        simulate a model (with or without sensitivities).

        Parameters
        ----------
        parameters
            With dimensions ``n_parameters``.
        times
            The times at which to calculate the model output / sensitivities.
        sensitivities
            If set to ``True`` the function returns the model outputs and
            sensitivities ``(values, sensitivities)``. If set to ``False`` the
            function only returns the model outputs ``values``. See
            :meth:`pints.ForwardModel.simulate()` and
            :meth:`pints.ForwardModel.simulate_with_sensitivities()` for
            details.
        """
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        if sensitivities:
            n_params = self.n_parameters()
            n_outputs = self.n_states()
            y0 = np.zeros(n_params * n_outputs + n_outputs)
            y0[0:n_outputs] = self._y0
            result = odeint(self._rhs_S1, y0, times, (parameters,))
            values = result[:, 0:n_outputs]
            dvalues_dp = (result[:, n_outputs:].reshape(
                (len(times), n_outputs, n_params)))
            return values, dvalues_dp
        else:
            values = odeint(self._rhs, self._y0, times, (parameters,))
            return values

    def simulateS1(self, parameters, times):
        """ See :meth:`pints.ForwardModelS1.simulateS1()`. """
        values, dvalues_dp = self._simulate(parameters, times, True)
        n_outputs = self.n_outputs()
        return values[:, :n_outputs], dvalues_dp[:, :n_outputs, :]
