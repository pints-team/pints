# -*- coding: utf-8 -*-
#
# Generalised Elliptical Slice Sampling
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
from scipy import optimize
from scipy import special


class SliceGeneralisedEllipticalMCMC(pints.SingleChainMCMC):
    """
    *Extends:* :class:`SingleChainMCMC`
    """

    def __init__(self, x0, sigma0=None):
        super(SliceGeneralisedEllipticalMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._x0 = np.asarray(x0, dtype=float)
        self._running = False
        self._ready_for_tell = False
        self._active_sample = None
        self._active_sample_pi_log_pdf = None
        self._proposed_sample = None
        self._proposed_sample_pi_log_pdf = None
        self._l_log_y = None
        self._prepare = True
        self._given_starting_points = None

        # Groups used for maximum-likelihood ``t`` parameters
        self._groups = None
        self._starts_mean = np.ones(self._n_parameters)
        self._starts_std = 2
        self._group_size = 10

        # Arrays of ``t`` distribution parameters for both groups
        self._t_mu = []
        self._t_Sigma = []
        self._t_nu = []

        # Group index: False for group 1, True for group 2
        self._index_active_group = False

        # Sample index
        self._index_active_sample = 0

        # Variable used to define new ellipse for ESS
        self._ess_nu = None

        # Initial proposal and angles bracked
        self._phi = None
        self._phi_min = None
        self._phi_max = None

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """

        # Check ask/tell pattern
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        # Initialise on first call
        if not self._running:
            self._running = True

        # Very first iteration
        if self._active_sample is None:

            # Ask for the log pdf of x0
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)

        # Prepare for ESS update
        if self._prepare:
            self._ready_for_tell = True
            return np.array(self._active_sample, copy=True)

        # Draw proposal
        self._proposed_sample = (
            (self._active_sample - self._t_mu[
                not self._index_active_group]) * np.cos(self._phi) +
            (self._ess_nu - self._t_mu[
                not self._index_active_group]) * np.sin(self._phi) +
            self._t_mu[not self._index_active_group])

        # Send new point for to check
        self._ready_for_tell = True
        return np.array(self._proposed_sample, copy=True)

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # Check ask/tell pattern
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        fx = np.asarray(reply, dtype=float)

        # Very first call
        if self._active_sample is None:

            # Check first point is somewhere sensible
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Update current sample, and initialise proposed sample for next
            # iteration
            self._active_sample = np.array(self._x0, copy=True)

            # Initialise array of groups
            if self._given_starting_points is None:
                starts = np.random.normal(
                    loc=self._starts_mean, scale=self._starts_std, size=(
                        2 * self._group_size - 1, self._n_parameters))
            else:
                starts = self._given_starting_points

            starts = np.concatenate(([self._x0], starts))
            self._groups = [starts[:self._group_size, :],
                            starts[self._group_size:, :]]

            # Parameters for t distributions
            for group in self._groups:
                mu, Sigma, nu = self._fit_mvstud(group)
                self._t_mu.append(np.array(mu, copy=True))
                self._t_Sigma.append(np.array(Sigma, copy=True))
                self._t_nu.append(nu)

            self._prepare = True

            # Return first point in chain, which is x0
            return np.array(self._active_sample, copy=True)

        # Index of non-active group
        index = not self._index_active_group

        # t parameters used for the GESS update
        t_nu = self._t_nu[index]
        t_Sigma = np.array(self._t_Sigma[index], copy=True)
        t_invSigma = np.linalg.inv(t_Sigma)
        t_mu = np.array(self._t_mu[index], copy=True)

        # Prepare for ESS update
        if self._prepare:
            # Store pi_log_pdf of active sample
            self._active_sample_pi_log_pdf = fx

            # Obtain parameters for inverse gamma distribution
            ig_alpha = (self._n_parameters + t_nu) / 2
            ig_beta = 0.5 * (
                t_nu + np.dot((self._active_sample - t_mu), np.dot(
                    t_invSigma, (self._active_sample - t_mu))))
            ig_s = 1. / np.random.gamma(ig_alpha, 1. / ig_beta)

            # Covariance matrix for Elliptical Slice Sampling update
            ess_Sigma = ig_s * t_Sigma

            # Draw ``nu`` from Gaussian prior
            self._ess_nu = np.random.multivariate_normal(t_mu, ess_Sigma)

            # Set log-likelihood treshold for ESS update
            u = np.random.uniform()
            self._l_log_y = (
                self._active_sample_pi_log_pdf - self._logt(
                    self._active_sample, t_mu, t_invSigma, t_nu) + np.log(u))

            # Draw an initial proposal and define bracket
            self._phi = np.random.uniform(0, 2 * np.pi)
            self._phi_min = self._phi - 2 * np.pi
            self._phi_max = self._phi

            self._prepare = False
            return None

        # Log likelihood of proposal
        log_pi_proposed = fx
        log_t_proposed = self._logt(
            self._proposed_sample, t_mu, t_invSigma, t_nu)
        log_l_proposed = log_pi_proposed - log_t_proposed

        # Acceptance Check
        if log_l_proposed > self._l_log_y:

            # Replace active sample with new accepted proposal
            self._groups[self._index_active_group][
                self._index_active_sample] = np.array(
                    self._proposed_sample, copy=True)

            # Manage indices
            if self._index_active_sample == self._group_size - 1:
                self._index_active_sample = 0
                self._index_active_group = not self._index_active_group

                # Update MLE parameters for non-active group
                mu, Sigma, nu = self._fit_mvstud(
                    self._groups[not self._index_active_group])
                self._t_mu[
                    not self._index_active_group] = np.array(mu, copy=True)
                self._t_Sigma[
                    not self._index_active_group] = np.array(Sigma, copy=True)
                self._t_nu[not self._index_active_group] = nu

            else:
                self._index_active_sample += 1

            # Update active sample
            self._active_sample = np.array(
                self._groups[self._index_active_group]
                [self._index_active_sample], copy=True)

            self._prepare = True
            return np.array(self._proposed_sample, copy=True)

        else:
            # Shrink bracket
            if self._phi < 0:
                self._phi_min = self._phi
            else:
                self._phi_max = self._phi

        # Draw new sample
        self._phi = np.random.uniform(self._phi_min, self._phi_max)

        return None

    # Function for computing the maximum likelihood for multivariate t
    # distribution parameters
    def _fit_mvstud(self, data, tolerance=1e-6):
        def opt_nu(delta_iobs, nu):
            def func0(nu):
                w_iobs = (nu + dim) / (nu + delta_iobs)
                f = -special.psi(nu / 2) + np.log(nu / 2) + np.sum(
                    np.log(w_iobs)) / n - np.sum(
                        w_iobs) / n + 1 + special.psi((
                            nu + dim) / 2) - np.log((nu + dim) / 2)
                return f

            if func0(1e6) >= 0:
                nu = np.inf
            else:
                nu = optimize.brentq(func0, 1e-6, 1e6)
            return nu

        # Extrapolate information about data: obtain dimention and number of
        # chains in the group
        data = data.T
        (dim, n) = data.shape

        # Initialize mu_0, Sigma_0, nu_0
        mu = np.array([np.median(data, 1)]).T
        Sigma = np.cov(data) * (n - 1) / n + 1e-1 * np.eye(dim)
        nu = 20
        last_nu = 0

        # Loop
        while np.abs(last_nu - nu) > tolerance:

            # Sum the distances of each point from the mean
            diffs = data - mu
            delta_iobs = np.sum(diffs * np.linalg.solve(Sigma, diffs), 0)

            # update nu
            last_nu = nu
            nu = opt_nu(delta_iobs, nu)
            if nu == np.inf:
                nu = 1e6
                return mu.T[0], Sigma, nu

            w_iobs = (nu + dim) / (nu + delta_iobs)

            # update Sigma
            Sigma = np.dot(w_iobs * diffs, diffs.T) / n

            # update mu
            mu = np.sum(w_iobs * data, 1) / sum(w_iobs)
            mu = np.array([mu]).T

        return mu.T[0], Sigma, nu

    # Log density of multivariate ``t`` distribution
    def _logt(self, x, mu, invSigma, nu):
        return - (self._n_parameters + nu) / 2 * np.log(
            1 + np.dot(x - mu, np.dot(invSigma, x - mu)) / nu)

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Generalised Elliptical Slice Sampling'

    def set_starts_mean(self, mean):
        """
        Sets mean of the Gaussian distribution from which we
        draw the starting samples.
        """
        if type(mean) == int or float:
            mean = np.full((len(self._x0)), mean)
        else:
            mean = np.asarray(mean)
        self._starts_mean = mean

    def set_starts_std(self, std):
        """
        Sets standard deviation of the Gaussian distribution from which we
        draw the starting samples.
        """
        if std <= 0:
            raise ValueError("""Standard deviation of the Gaussian distribution
            from which we draw the starting samples should be positive""")
        self._starts_std = std

    def set_group_size(self, group_size):
        """
        Sets size of group of starting points.
        """
        if group_size <= 0:
            raise ValueError("""Each group of starting points should have at least
            one value.""")
        self._group_size = group_size

    def get_starts_mean(self):
        """
        Returns mean of the Gaussian distribution from which we
        draw the starting samples.
        """
        return self._starts_mean

    def get_starts_std(self):
        """
        Returns standard deviation of the Gaussian distribution from which we
        draw the starting samples.
        """
        return self._starts_std

    def get_group_size(self):
        """
        Returns size of the groups of starting points.
        """
        return self._group_size

    def give_initial_points(self, points):
        """
        Sets starting points.
        """
        points = np.asarray(points)
        if points.shape[0] != 2 * self._group_size - 1:
            raise ValueError("""The array of starting points should include ``2 *
            group_size - 1`` values.""")
        if points.shape[1] != self._n_parameters:
            raise ValueError("""The dimensions of each starting point should be equal
            to the number of parameters.""")
        self._given_starting_points = points
