#
# Eight schools log-pdf.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints

from . import ToyLogPDF


class EightSchoolsLogPDF(ToyLogPDF):
    r"""
    The classic Eight Schools example that is discussed in [1]_.

    The aim of this model (implemented as a :class:`pints.ToyLogPDF`) is to
    determine the effects of coaching on SAT scores in 8 schools (each school
    being denoted by subscript j in the following equations). It it used
    by statisticians to illustrate how hierarchical models can quite easily
    become unidentified, making inference hard.

    This model is hierarchical and takes the form,

    .. math::
        \begin{align}
        \mu &\sim \mathcal{N}(0, 5) \\
        \tau &\sim \text{Cauchy}(0, 5) \\
        \theta_j &\sim \mathcal{N}(\mu, \tau) \\
        y_j &\sim \mathcal{N}(\theta_j, \sigma_j), \\
        \end{align}

    where :math:`\sigma_j` is known. The user may choose between the "centered"
    parameterisation of the model (which exactly mirrors the statistical
    model), and the "non-centered" parameterisation, which introduces
    auxillary variables to improve chain mixing. The non-centered model takes
    the form,

    .. math::
        \begin{align}
        \mu &\sim \mathcal{N}(0, 5) \\
        \tau &\sim \text{Cauchy}(0, 5) \\
        \tilde{\theta}_j &\sim \mathcal{N}(0, 1) \\
        \theta_j &= mu + \tilde{\theta}_j \tau \\
        y_j &\sim \mathcal{N}(\theta_j, \sigma_j). \\
        \end{align}

    Note that, in the non-centered case, the parameter samples correspond to
    :math:`\tilde{\theta}` rather than :math:`\theta`.

    The model uses a 10-dimensional parameter vector, composed of

    - ``mu``, the population-level score
    - ``tau``, the population-level standard deviation
    - ``theta_j``, school j's mean score (for each of the 8 schools).

    Extends :class:`pints.toy.ToyLogPDF`.

    Parameters
    ----------
    centered : bool
        Whether or not to use the centered formulation.

    References
    ----------
    .. [1] "Bayesian data analysis", 3rd edition, 2014, Gelman, A et al..
    """
    def __init__(self, centered=True):
        self._n_parameters = 10
        self._y_j = [28, 8, -3, 7, -1, 1, 18, 12]
        self._sigma_j = [15, 10, 16, 11, 9, 11, 10, 18]
        # priors
        self._mu_log_pdf = pints.GaussianLogPrior(0, 5)
        self._tau_log_pdf = pints.HalfCauchyLogPrior(0, 5)
        self._centered = bool(centered)

    def __call__(self, x):
        if len(x) != 10:
            raise ValueError('Input parameters must be of length 10.')
        mu = x[0]
        tau = x[1]
        if tau < 0:  # to handle proposals without having to change log-priors
            return -np.inf

        thetas = x[2:]

        log_prob = self._mu_log_pdf([mu])
        log_prob += self._tau_log_pdf([tau])
        if self._centered:
            log_prior = pints.GaussianLogPrior(mu, tau)
        else:
            log_prior = pints.GaussianLogPrior(0, 1)

        for i, theta_tilde in enumerate(thetas):
            log_prob += log_prior([theta_tilde])
            if self._centered:
                theta = theta_tilde
            else:
                theta = mu + theta_tilde * tau
            log_prior_2 = pints.GaussianLogPrior(theta, self._sigma_j[i])
            log_prob += log_prior_2([self._y_j[i]])

        return log_prob

    def data(self):
        """ Returns data used to fit model from [1]_. """
        return {'J': 8, 'y': self._y_j, 'sigma': self._sigma_j}

    def evaluateS1(self, x):
        """ See :meth:`pints.LogPDF.evaluateS1()`. """
        if len(x) != 10:
            raise ValueError('Input parameters must be of length 10.')
        mu = x[0]
        tau = x[1]
        if tau < 0:  # to handle proposals without having to change log-priors
            return -np.inf, np.full([1, 10], -np.inf)

        thetas = x[2:]
        log_prob1, dL1 = self._mu_log_pdf.evaluateS1([mu])
        log_prob2, dL2 = self._tau_log_pdf.evaluateS1([tau])
        log_prob = log_prob1 + log_prob2

        if self._centered:
            log_prior = pints.GaussianLogPrior(mu, tau)
            dL_theta = []
            for i, theta in enumerate(thetas):
                y_j = self._y_j[i]
                sigma_j = self._sigma_j[i]
                dL1[0] += (theta - mu) / tau**2
                dL2[0] += ((theta - mu)**2 - tau**2) / tau**3
                log_prob_temp, dL_temp = log_prior.evaluateS1([theta])
                log_prob += log_prob_temp
                log_prob += pints.GaussianLogPrior(theta, sigma_j)([y_j])
                dL_temp[0] += (y_j - theta) / sigma_j**2
                dL_theta.append(dL_temp[0])
        else:
            log_prior = pints.GaussianLogPrior(0, 1)
            dL_theta = []
            for i, theta_tilde in enumerate(thetas):
                y_j = self._y_j[i]
                sigma_j = self._sigma_j[i]
                theta = mu + theta_tilde * tau
                y_minus_theta = (y_j - theta) / sigma_j**2
                dL1[0] += y_minus_theta
                dL2[0] += theta_tilde * y_minus_theta
                log_prob_temp, dL_temp = log_prior.evaluateS1([theta_tilde])
                log_prob += log_prob_temp
                log_prob += pints.GaussianLogPrior(theta, sigma_j)([y_j])
                dL_temp[0] += tau * y_minus_theta
                dL_theta.append(dL_temp[0])

        return log_prob, ([dL1[0]] + [dL2[0]] + dL_theta)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def suggested_bounds(self):
        """ See :meth:`pints.toy.ToyLogPDF.suggested_bounds()`. """
        magnitude = 40
        bounds = np.tile([-magnitude, magnitude], (self.n_parameters(), 1))
        bounds[1, 0] = 0
        return np.transpose(bounds).tolist()
