#
# Toy base classes.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints
import scipy


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
        Returns an NumPy array of the parameter values that are representative
        of the model.

        For example, these parameters might reproduce a particular result that
        the model is famous for.
        """
        raise NotImplementedError

    def suggested_times(self):
        """
        Returns an NumPy array of time points that is representative of the
        model
        """
        raise NotImplementedError


class ToyODEModel(ToyModel):
    """
    Defines an interface for toy problems where the underlying model is an
    ordinary differential equation (ODE) that describes some time-series
    generating model.

    Note that toy ODE models should extend both :class:`pints.ToyODEModel` and
    one of the forward model classes, e.g. :class:`pints.ForwardModel` or
    :class:`pints.ForwardModelS1`.

    To use this class as the basis for a :class:`pints.ForwardModel`, the
    method :meth:`_rhs()` should be reimplemented.

    Models implementing :meth:`_rhs()`, :meth:`jacobian()` and :meth:`_dfdp()`
    can be used to create a :class:`pints.ForwardModelS1`.
    """
    def _dfdp(self, y, t, p):
        """
        Returns the derivative of the ODE RHS at time ``t``, with respect to
        model parameters ``p``.

        Parameters
        ----------
        y
            The state vector at time ``t`` (with length ``n_outputs``).
        t
            The time to evaluate at (as a scalar).
        p
            A vector of model parameters (of length ``n_parameters``).

        Returns
        -------
        A matrix of dimensions ``n_outputs`` by ``n_parameters``.
        """
        raise NotImplementedError

    def initial_conditions(self):
        """ Returns the initial conditions of the model. """
        return self._y0

    def jacobian(self, y, t, p):
        """
        Returns the Jacobian (the derivative of the RHS ODE with respect to the
        outputs) at time ``t``.

        Parameters
        ----------
        y
            The state vector at time ``t`` (with length ``n_outputs``).
        t
            The time to evaluate at (as a scalar).
        p
            A vector of model parameters (of length ``n_parameters``).

        Returns
        -------
        A matrix of dimensions ``n_outputs`` by ``n_outputs``.
        """
        raise NotImplementedError

    def n_states(self):
        """
        Returns number of states in underlying ODE. Note: will not be same as
        ``n_outputs()`` for models where only a subset of states are observed.
        """
        return self.n_outputs()

    def _rhs(self, y, t, p):
        """
        Returns the evaluated RHS (``dy/dt``) for a given state vector ``y``,
        time ``t``, and parameter vector ``p``.

        Parameters
        ----------
        y
            The state vector at time ``t`` (with length ``n_outputs``).
        t
            The time to evaluate at (as a scalar).
        p
            A vector of model parameters (of length ``n_parameters``).

        Returns
        -------
        A vector of length ``n_outputs``.
        """
        raise NotImplementedError

    def _rhs_S1(self, y_and_dydp, t, p):
        """
        Forms the RHS of ODE for numerical integration to obtain both outputs
        and sensitivities.

        Parameters
        ----------
        y_and_dydp
            A combined vector of states (elements ``0`` to ``n_outputs - 1``)
            and sensitivities (elements ``n_outputs`` onwards).
        t
            The time to evaluate at (as a scalar).
        p
            A vector of model parameters (of length ``n_parameters``).

        Returns
        -------
        A vector of length ``n_outputs + n_parameters``.
        """

        # separating initial values of model outputs(y) and sensitivities(dydp)
        y = y_and_dydp[0:self.n_states()]
        dydp = y_and_dydp[self.n_states():].reshape((self.n_parameters(),
                                                     self.n_states()))

        # calculating the derivatives w.r.t t of the model outputs
        dydt = self._rhs(y, t, p)

        # calculating sensitivities
        d_dydp_dt = (
            np.matmul(dydp, np.transpose(self.jacobian(y, t, p))) +
            np.transpose(self._dfdp(y, t, p)))
        return np.concatenate((dydt, d_dydp_dt.reshape(-1)))

    def set_initial_conditions(self, y0):
        """ Sets the initial conditions of the model. """
        self._y0 = y0

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
        times = pints.vector(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        # Scipy odeint requires the first element in ``times`` to be the
        # initial point, which ForwardModel says _has to be_ t=0
        offset = 0
        if len(times) < 1 or times[0] != 0:
            times = np.concatenate(([0], times))
            offset = 1

        if sensitivities:
            n_params = self.n_parameters()
            n_outputs = self.n_states()
            y0 = np.zeros(n_params * n_outputs + n_outputs)
            y0[0:n_outputs] = self._y0
            result = scipy.integrate.odeint(
                self._rhs_S1, y0, times, (parameters,))
            values = result[:, 0:n_outputs]
            dvalues_dp = (result[:, n_outputs:].reshape(
                (len(times), n_outputs, n_params), order="F"))
            return values[offset:], dvalues_dp[offset:]
        else:
            values = scipy.integrate.odeint(
                self._rhs, self._y0, times, (parameters,))
            return values[offset:, :self.n_outputs()].squeeze()

    def simulateS1(self, parameters, times):
        """ See :meth:`pints.ForwardModelS1.simulateS1()`. """
        values, dvalues_dp = self._simulate(parameters, times, True)
        n_outputs = self.n_outputs()
        return values[:, :n_outputs].squeeze(), dvalues_dp[:, :n_outputs, :]
