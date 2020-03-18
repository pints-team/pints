#
# Eight schools log-pdf.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints

from . import ToyLogPDF


class EightSchoolsCenteredLogPDF(ToyLogPDF):
    r"""
    The classic Eight Schools example from [1]_. This model was used to
    determine the effects of coaching on SATS scores in 8 schools. This model
    is hierarchical and takes the form,

    .. math::
        \theta_j\sim\mathcal{N}(\mu, \tau)
        y_j\sim\mathcal{N}(\theta_j, \sigma_j)

    where :math:`sigma_j` is known. In this implementation of the model, we
    assume the "centered" parameterisation that mirrors the above statistical
    model.

    Extends :class:`pints.toy.ToyLogPDF`.

    Parameters
    ----------
    mu : float
        Mean population-level score
    tau : float
        Population-level standard deviation
    theta_j : float
        School j's mean score

    References
    ----------
    .. [1] "Bayesian data analysis", 3rd edition, 2014, Gelman, A et al..
    """
    def __init__(self):
        self._n_parameters = 10
        self._y_j = [28, 8, -3, 7, -1, 1, 18, 12]
        self._sigma_j = [15, 10, 16, 11, 9, 11, 10, 18]
        # priors
        self._mu_log_pdf = pints.GaussianLogPrior(0, 5)
        self._tau_log_pdf = pints.HalfCauchyLogPrior(0, 5)

    def __call__(self, x):
        mu = x[0]
        tau = x[1]
        thetas = x[2:]
        log_prior = pints.GaussianLogPrior(mu, tau)
        log_prob = self._mu_log_pdf([mu])
        log_prob += self._tau_log_pdf([tau])
        for i, theta in enumerate(thetas):
            log_prob += log_prior([theta])
            log_prob += (
                pints.GaussianLogPrior(theta, self._sigma_j[i])([self._y_j[i]])
            )
        return log_prob

    def data(self):
        """ Returns data used to fit model from [1]_. """
        return {'J': 8, 'y': self._y_j, 'sigma': self._sigma_j}

    def evaluateS1(self, x):
        """ See :meth:`pints.LogPDF.evaluateS1()`. """
        mu = x[0]
        tau = x[1]
        thetas = x[2:]
        log_prior = pints.GaussianLogPrior(mu, tau)
        log_prob1, dL1 = self._mu_log_pdf.evaluateS1([mu])
        log_prob2, dL2 = self._tau_log_pdf.evaluateS1([tau])
        log_prob = log_prob1 + log_prob2
        dL_theta = []
        for i, theta in enumerate(thetas):
            log_prob_temp, dL_temp = log_prior.evaluateS1([theta])
            log_prob += log_prob_temp
            log_prob += (
                pints.GaussianLogPrior(theta, self._sigma_j[i])([self._y_j[i]])
            )
            dL_temp[0] += (self._y_j[i] - theta) / (self._sigma_j[i]**2)
            dL_theta.append(dL_temp[0])
        return log_prob, ([dL1[0]] + [dL2[0]] + dL_theta)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def suggested_bounds(self):
        """ See :meth:`pints.toy.ToyLogPDF.suggested_bounds()`. """
        magnitude = 50
        bounds = np.tile([-magnitude, magnitude], (self.n_parameters(), 1))
        return np.transpose(bounds).tolist()
