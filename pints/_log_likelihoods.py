#
# Log-likelihood functions
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
import scipy.special


class AR1LogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming AR1 noise model

    .. math::
        \log{L(\theta, \sigma'|\boldsymbol{x})} =
            -\frac{N}{2}\log{2\pi}
            -N\log{\sigma'}
            -\frac{1}{2\sigma'^2}
                \sum_{i=2}^N{(\epsilon_i x_i - \rho \epsilon_{i-1} )^2}

    where

    .. math::
        \epsilon_i = x_i - f_i(\theta)

    and

    .. math::
        \sigma' = \sigma \sqrt{1-\rho^2}

    .

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem two parameters are added (rho, sigma),
        for a multi-output problem 2 * ``n_outputs`` parameters are added.

    """

    def __init__(self, problem):
        super(AR1LogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times) - 1
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + 2 * self._no

        # Pre-calculate parts
        self._logn = 0.5 * (self._nt) * np.log(2 * np.pi)

    def __call__(self, x):
        m = 2 * self._no
        parameters = x[-m:]
        rho = np.asarray(parameters[0::2])
        sigma = np.asarray(parameters[1::2])
        sigma = np.asarray(sigma) * np.sqrt(1 - rho**2)
        error = self._values - self._problem.evaluate(x[:-2 * self._no])
        autocorr_error = error[1:] - rho * error[:-1]
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(autocorr_error**2, axis=0) / (2 * sigma**2))


class ARMA11LogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming ARMA(1,1) noise model.

    .. math::
        \log{L(\theta, \sigma|\boldsymbol{x})} =
            -\frac{N}{2}\log{2\pi}
            -N\log{\sigma}
            -\frac{1}{2\sigma^2}
                \sum_{i=3}^N{(\nu_i - \phi \nu_{i-1})^2}

    where

    .. math::
        \nu_i = \epsilon_i - \rho \epsilon_{i-1}

    and

    .. math::
        \epsilon_i = x_i - f_i(\theta)

    and

    .. math::
        \sigma = \sigma\sqrt{\frac{1-\rho^2}{1 + 2\phi\rho + \phi^2}}`

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem three parameters are added (rho, phi, sigma),
        for a multi-output problem 3 * ``n_outputs`` parameters are added.
    """

    def __init__(self, problem):
        super(ARMA11LogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times) - 2
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + 3 * self._no

        # Pre-calculate parts
        self._logn = 0.5 * (self._nt) * np.log(2 * np.pi)

    def __call__(self, x):
        m = 3 * self._no
        parameters = x[-m:]
        rho = np.asarray(parameters[0::3])
        phi = np.asarray(parameters[1::3])
        sigma = np.asarray(parameters[2::3])
        sigma = (
            sigma *
            np.sqrt((1.0 - rho**2) / (1.0 + 2.0 * phi * rho + phi**2))
        )
        error = self._values - self._problem.evaluate(x[:-m])
        v = error[1:] - rho * error[:-1]
        autocorr_error = v[1:] - phi * v[:-1]
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(autocorr_error**2, axis=0) / (2 * sigma**2))


class GaussianIntegratedUniformLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming independent Gaussian-distributed noise
    at each time point where :math:`\sigma\sim U(a,b)` has been integrated out
    of the joint posterior of :math:`p(\theta,\sigma|X)`,

    .. math::
        \begin{align} p(\theta|X) &= \int_{0}^{\infty} p(\theta, \sigma|X)
        \mathrm{d}\sigma\\
        &\propto \int_{0}^{\infty} p(X|\theta, \sigma) p(\theta, \sigma)
        \mathrm{d}\sigma,\end{align}

    Note that this is exactly the same statistical model as
    :class:`pints.GaussianLogLikelihood` with a uniform prior on
    :math:`\sigma`.

    A possible advantage of this log-likelihood compared with using a
    :class:`pints.GaussianLogLikelihood`, is that it has one fewer parameters
    (:math:`sigma`) which may speed up convergence to the posterior
    distribution, especially for multi-output problems which will have
    ``n_outputs`` fewer parameter dimensions.

    The log-likelihood is given in terms of the sum of squared errors:

    .. math::
        SSE = \sum_{i=1}^n (f_i(\theta) - y_i)^2

    and is given up to a normalisation constant by:

    .. math::
        \begin{align}
        \text{log} L =
            & - n / 2 \text{log}(\pi) \\
            & - \text{log}(2 (b - a) \sqrt(2)) \\
            & + (1 / 2 - n / 2) \text{log}(SSE) \\
            & + \text{log}\left[\Gamma((n - 1) / 2, \frac{SSE}{2 b^2}) -
                \Gamma((n - 1) / 2, \frac{SSE}{2 a^2}) \right]
        \end{align}

    where :math:`\Gamma(u,v)` is the upper incomplete gamma function as defined
    here: https://en.wikipedia.org/wiki/Incomplete_gamma_function

    This log-likelihood is inherently a Bayesian method since it assumes a
    uniform prior on :math:`\sigma\sim U(a,b)`. However using this likelihood
    in optimisation routines should yield the same estimates as the full
    :class:`pints.GaussianLogLikelihood`.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`.
    lower
        The lower limit on the uniform prior om `sigma`. Must be
        non-negative.
    upper
        The upper limit on the uniform prior om `sigma`.
    """

    def __init__(self, problem, lower, upper):
        super(GaussianIntegratedUniformLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters()
        a = lower
        if np.isscalar(a):
            a = np.ones(self._no) * float(a)
        else:
            a = pints.vector(a)
            if len(a) != self._no:
                raise ValueError(
                    'Lower limit on uniform prior for sigma must be a ' +
                    ' scalar or a vector of length n_outputs.')
        if np.any(a < 0):
            raise ValueError('Lower limit on uniform prior for sigma ' +
                             'must be non-negative.')
        b = upper
        if np.isscalar(b):
            b = np.ones(self._no) * float(b)
        else:
            b = pints.vector(b)
            if len(b) != self._no:
                raise ValueError(
                    'Upper limit on uniform prior for sigma must be a ' +
                    ' scalar or a vector of length n_outputs.')
        if np.any(b <= 0):
            raise ValueError('Upper limit on uniform prior for sigma ' +
                             'must be positive.')
        diff = b - a
        if np.any(diff <= 0):
            raise ValueError('Upper limit on uniform prior for sigma ' +
                             'must exceed lower limit.')
        self._a = a
        self._b = b

        # Pre-calculate
        n = self._nt
        self._n_minus_1_over_2 = (n - 1.0) / 2.0
        self._const_a_0 = (
            -n * np.log(b) - (n / 2.0) * np.log(np.pi) -
            np.log(2 * np.sqrt(2))
        )
        self._b2 = self._b**2
        self._a2 = self._a**2
        self._const_general = (
            -(n / 2.0) * np.log(np.pi) - np.log(2 * np.sqrt(2) * (b - a))
        )
        self._log_gamma = scipy.special.gammaln(self._n_minus_1_over_2)
        self._two_power = 2**(1 / 2 - n / 2)

    def __call__(self, x):
        error = self._values - self._problem.evaluate(x)
        sse = np.sum(error**2, axis=0)

        # Calculate
        log_temp = np.zeros(len(self._a2))
        sse = pints.vector(sse)
        for i, a in enumerate(self._a2):
            if a != 0:
                log_temp[i] = np.log(
                    scipy.special.gammaincc(self._n_minus_1_over_2,
                                            sse[i] / (2 * self._b2[i])) -
                    scipy.special.gammaincc(self._n_minus_1_over_2,
                                            sse[i] / (2 * a)))
            else:
                log_temp[i] = np.log(
                    scipy.special.gammaincc(self._n_minus_1_over_2,
                                            sse[i] / (2 * self._b2[i])))
        return np.sum(
            self._const_general -
            self._n_minus_1_over_2 * np.log(sse) +
            self._log_gamma +
            log_temp
        )


class CauchyLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming independent Cauchy-distributed noise
    at each time point, and adds one parameter: the scale (``sigma``).

    For a noise characterised by ``sigma``, the log-likelihood is of the form:

    .. math::
        \log{L(\theta, \sigma)} =
              -N\log \pi - N\log \sigma
              -\sum_{i=1}^N\log(1 +
            \frac{x_i - f(\theta)}{\sigma}^2)

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem one parameter is added ``sigma``, where
        ``sigma`` is scale, for a multi-output problem ``n_outputs``
        parameters are added.
    """

    def __init__(self, problem):
        super(CauchyLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem (one for each output)
        self._n_parameters = problem.n_parameters() + self._no

        # Pre-calculate
        self._n = len(self._times)
        self._n_log_pi = self._n * np.log(np.pi)

    def __call__(self, x):
        # For multiparameter problems the parameters are stored as
        # (model_params_1, model_params_2, ..., model_params_k,
        # sigma_1, sigma_2,...)
        n = self._n
        m = self._no

        # problem parameters
        problem_parameters = x[:-m]
        error = self._values - self._problem.evaluate(problem_parameters)

        # Distribution parameters
        sigma = np.asarray(x[-m:])

        # Calculate
        return np.sum(
            - self._n_log_pi
            - n * np.log(sigma)
            - np.sum(np.log(1 + (error / sigma)**2), axis=0)
        )


class GaussianKnownSigmaLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming independent Gaussian noise at each
    time point, using a known value for the standard deviation (sigma) of that
    noise:

    .. math::
        \log{L(\theta | \sigma,\boldsymbol{x})} =
            -\frac{N}{2}\log{2\pi}
            -N\log{\sigma}
            -\frac{1}{2\sigma^2}\sum_{i=1}^N{(x_i - f_i(\theta))^2}

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`.
    sigma
        The standard devation(s) of the noise. Can be a single value or a
        sequence of sigma's for each output. Must be greater than zero.
    """

    def __init__(self, problem, sigma):
        super(GaussianKnownSigmaLogLikelihood, self).__init__(problem)

        # Store counts
        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times()

        # Check sigma
        if np.isscalar(sigma):
            sigma = np.ones(self._no) * float(sigma)
        else:
            sigma = pints.vector(sigma)
            if len(sigma) != self._no:
                raise ValueError(
                    'Sigma must be a scalar or a vector of length n_outputs.')
        if np.any(sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')

        # Pre-calculate parts
        self._offset = -0.5 * self._nt * np.log(2 * np.pi)
        self._offset -= self._nt * np.log(sigma)
        self._multip = -1 / (2.0 * sigma**2)

        # Pre-calculate S1 parts
        self._isigma2 = sigma**-2

    def __call__(self, x):
        error = self._values - self._problem.evaluate(x)
        return np.sum(self._offset + self._multip * np.sum(error**2, axis=0))

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        # Evaluate, and get residuals
        y, dy = self._problem.evaluateS1(x)

        # Reshape dy, in case we're working with a single-output problem
        dy = dy.reshape(self._nt, self._no, self._np)

        # Note: Must be (data - simulation), sign now matters!
        r = self._values - y

        # Calculate log-likelihood
        L = np.sum(self._offset + self._multip * np.sum(r**2, axis=0))

        # Calculate derivative
        dL = np.sum(
            (self._isigma2 * np.sum((r.T * dy.T).T, axis=0).T).T, axis=0)

        # Return
        return L, dL


class GaussianLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming independent Gaussian noise at each
    time point, and adds a parameter representing the standard deviation
    (sigma) of the noise on each output.

    For a noise level of ``sigma``, the likelihood becomes:

    .. math::
        L(\theta, \sigma|\boldsymbol{x})
            = p(\boldsymbol{x} | \theta, \sigma)
            = \prod_{j=1}^{n_t} \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(
                -\frac{(x_j - f_j(\theta))^2}{2\sigma^2}\right)

    leading to a log likelihood of:

    .. math::
        \log{L(\theta, \sigma|\boldsymbol{x})} =
            -\frac{n_t}{2} \log{2\pi}
            -n_t \log{\sigma}
            -\frac{1}{2\sigma^2}\sum_{j=1}^{n_t}{(x_j - f_j(\theta))^2}

    where ``n_t`` is the number of time points in the series, ``x_j`` is the
    sampled data at time ``j`` and ``f_j`` is the simulated data at time ``j``.

    For a system with ``n_o`` outputs, this becomes

    .. math::
        \log{L(\theta, \sigma|\boldsymbol{x})} =
            -\frac{n_t n_o}{2}\log{2\pi}
            -\sum_{i=1}^{n_o}{ {n_t}\log{\sigma_i} }
            -\sum_{i=1}^{n_o}{\left[
                \frac{1}{2\sigma_i^2}\sum_{j=1}^{n_t}{(x_j - f_j(\theta))^2}
             \right]}

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem a single parameter is added, for a multi-output
        problem ``n_outputs`` parameters are added.
    """

    def __init__(self, problem):
        super(GaussianLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._no

        # Pre-calculate parts
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        error = self._values - self._problem.evaluate(x[:-self._no])
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        sigma = np.asarray(x[-self._no:])

        # Evaluate, and get residuals
        y, dy = self._problem.evaluateS1(x[:-self._no])

        # Reshape dy, in case we're working with a single-output problem
        dy = dy.reshape(self._nt, self._no, self._n_parameters - self._no)

        # Note: Must be (data - simulation), sign now matters!
        r = self._values - y

        # Calculate log-likelihood
        L = self.__call__(x)

        # Calculate derivatives in the model parameters
        dL = np.sum(
            (sigma**(-2.0) * np.sum((r.T * dy.T).T, axis=0).T).T, axis=0)

        # Calculate derivative wrt sigma
        dsigma = -self._nt / sigma + sigma**(-3.0) * np.sum(r**2, axis=0)
        dL = np.concatenate((dL, np.array(list(dsigma))))

        # Return
        return L, dL


class KnownNoiseLogLikelihood(GaussianKnownSigmaLogLikelihood):
    """ Deprecated alias of :class:`GaussianKnownSigmaLogLikelihood`. """

    def __init__(self, problem, sigma):
        # Deprecated on 2019-02-06
        import logging
        logging.basicConfig()
        log = logging.getLogger(__name__)
        log.warning(
            'The class `pints.KnownNoiseLogLikelihood` is deprecated.'
            ' Please use `pints.GaussianKnownSigmaLogLikelihood` instead.')
        super(KnownNoiseLogLikelihood, self).__init__(problem, sigma)


class MultiplicativeGaussianLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming heteroscedastic Gaussian errors, with
    the magnitude of the error variance scaling with the problem output.

    For each output in the problem, this likelihood introduces two new scalar
    parameters: an exponential power ``eta`` and a scale ``sigma``.

    This likelihood is applicable to a model given by

    .. math::
        X(t) = f(t; \theta) + f(t; \theta)^\eta v(t)

    where v(t) is iid Gaussian:

    .. math::
        v(t) \sim \text{ iid } N(0, \sigma)

    Note that the scalar parameter ``eta`` controls the exponential dependence
    of the noise on the function output, while the scalar parameter ``sigma``
    provides a baseline level of the noise standard deviation. This model leads
    to a log likelihood of

    .. math::
        \log{L(\theta, \sigma, \eta | \boldsymbol{x})} =
            -\frac{n_t}{2} \log{2 \pi}
            -\sum_{i=1}^{n_t}{\log{f(t_i, \theta)^\eta \sigma}}
            -\frac{1}{2}\sum_{i=1}^{n_t}
                \frac{(X(t_i) - f(t_i, \theta))^2}
                {(f(t_i, \theta)^\eta \sigma)^2}

    where ``n_t`` is the number of time points in the series.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    ``problem``
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem two parameters are added (``eta``, ``sigma``),
        for a multi-output problem 2 times ``n_outputs`` parameters are added.
    """

    def __init__(self, problem):
        super(MultiplicativeGaussianLogLikelihood, self).__init__(problem)

        # Get number of times and number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + 2 * self._no

        # Pre-calculate the constant part of the likelihood
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)

    def __call__(self, x):
        m = 2 * self._no
        noise_parameters = x[-m:]
        eta = np.asarray(noise_parameters[0::2])
        sigma = np.asarray(noise_parameters[1::2])
        function_values = self._problem.evaluate(x[:-m])

        log_likelihood = \
            np.sum(-self._logn
                   - np.sum(np.log(function_values**eta * sigma), axis=0)
                   - 0.5 / sigma**2
                   * np.sum((self._values - function_values)**2
                            / function_values ** (2 * eta), axis=0))

        return log_likelihood


class ScaledLogLikelihood(pints.ProblemLogLikelihood):
    """
    Calculates a log-likelihood based on a (conditional)
    :class:`ProblemLogLikelihood` divided by the number of time samples.

    The returned value will be ``(1 / n) * log_likelihood(x|problem)``, where
    ``n`` is the number of time samples multiplied by the number of outputs.

    This log-likelihood operates on both single and multi-output problems.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    log_likelihood
        A :class:`ProblemLogLikelihood` to scale.
    """

    def __init__(self, log_likelihood):
        # Check arguments
        if not isinstance(log_likelihood, pints.ProblemLogLikelihood):
            raise ValueError(
                'Given log_likelihood must extend pints.ProblemLogLikelihood')

        # Call parent constructor
        super(ScaledLogLikelihood, self).__init__(log_likelihood._problem)

        # Store log-likelihood
        self._log_likelihood = log_likelihood

        # Pre-calculate parts
        self._f = 1.0 / np.product(self._values.shape)

    def __call__(self, x):
        return self._f * self._log_likelihood(x)

    def evaluateS1(self, x):
        """
        See :meth:`LogPDF.evaluateS1()`.

        *This method only works if the underlying :class:`LogPDF` object
        implements the optional method :meth:`LogPDF.evaluateS1()`!*
        """
        a, b = self._log_likelihood.evaluateS1(x)
        return self._f * a, self._f * np.asarray(b)


class StudentTLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming independent Student-t-distributed
    noise at each time point, and adds two parameters: one representing the
    degrees of freedom (``nu``), the other representing the scale (``sigma``).

    For a noise characterised by ``nu'' and ``sigma``, the log likelihood is of
    the form:

    .. math::
        \log{L(\theta, \nu, \sigma|\boldsymbol{x})} =
            N\frac{\nu}{2}\log(\nu) - N\log(\sigma) -
            N\log B(\nu/2, 1/2)
            -\frac{1+\nu}{2}\sum_{i=1}^N\log(\nu +
            \frac{x_i - f(\theta)}{\sigma}^2)

    where ``B(.,.)`` is a beta function.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem two parameters are added ``(nu, sigma)``, where
        ``nu`` is the degrees of freedom and ``sigma`` is scale, for a
        multi-output problem ``2 * n_outputs`` parameters are added.
    """

    def __init__(self, problem):
        super(StudentTLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem (two for each output)
        self._n_parameters = problem.n_parameters() + 2 * self._no

        # Pre-calculate
        self._n = len(self._times)

    def __call__(self, x):
        # For multiparameter problems the parameters are stored as
        # (model_params_1, model_params_2, ..., model_params_k,
        # nu_1, sigma_1, nu_2, sigma_2,...)
        n = self._n
        m = 2 * self._no

        # problem parameters
        problem_parameters = x[:-m]
        error = self._values - self._problem.evaluate(problem_parameters)

        # Distribution parameters
        parameters = x[-m:]
        nu = np.asarray(parameters[0::2])
        sigma = np.asarray(parameters[1::2])

        # Calculate
        return np.sum(
            + 0.5 * n * nu * np.log(nu)
            - n * np.log(sigma)
            - n * np.log(scipy.special.beta(0.5 * nu, 0.5))
            - 0.5 * (1 + nu) * np.sum(np.log(nu + (error / sigma)**2), axis=0)
        )


class UnknownNoiseLogLikelihood(GaussianLogLikelihood):
    """
    Deprecated alias of :class:`GaussianLogLikelihood`
    """

    def __init__(self, problem):
        # Deprecated on 2019-02-06
        import logging
        logging.basicConfig()
        log = logging.getLogger(__name__)
        log.warning(
            'The class `pints.KnownNoiseLogLikelihood` is deprecated.'
            ' Please use `pints.GaussianLogLikelihood` instead.')
        super(UnknownNoiseLogLikelihood, self).__init__(problem)
