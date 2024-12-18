#
# Log-likelihood functions
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np
import scipy.special
import scipy.stats


class AR1LogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming AR(1) (autoregressive order 1) errors.

    In this error model, the ith error term
    :math:`\epsilon_i = x_i - f_i(\theta)` is assumed to obey the following
    relationship.

    .. math::
        \epsilon_i = \rho \epsilon_{i-1} + \nu_i

    where :math:`\nu_i` is IID Gaussian white noise with variance
    :math:`\sigma^2 (1-\rho^2)`. Therefore, this likelihood is appropriate when
    error terms are autocorrelated, and the parameter :math:`\rho`
    determines the level of autocorrelation.

    This model is parameterised as such because it leads to a simple marginal
    distribution :math:`\epsilon_i \sim N(0, \sigma)`.

    This class treats the error at the first time point (i=1) as fixed, which
    simplifies the calculations. For sufficiently long time-series, this
    conditioning on the first observation has at most a small effect on the
    likelihood. Further details as well as the alternative unconditional
    likelihood are available in [1]_ , chapter 5.2.

    Noting that

    .. math::
        \nu_i = \epsilon_i - \rho \epsilon_{i-1} \sim N(0, \sigma^2 (1-\rho^2))

    we thus calculate the likelihood as the product of normal likelihoods from
    :math:`i=2,...,N`, for a time series with N time points.

    .. math::
        L(\theta, \sigma, \rho|\boldsymbol{x}) =
            -\frac{N-1}{2} \log(2\pi)
            - (N-1) \log(\sigma')
            - \frac{1}{2\sigma'^2} \sum_{i=2}^N (\epsilon_i
                                 - \rho \epsilon_{i-1})^2

    for :math:`\sigma' = \sigma \sqrt{1-\rho^2}`.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem two parameters are added (rho, sigma),
        for a multi-output problem 2 * ``n_outputs`` parameters are added.

    References
    ----------
    .. [1] Hamilton, James D. Time series analysis. Vol. 2. New Jersey:
           Princeton, 1994.
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
        if any(sigma <= 0):
            return -np.inf
        sigma = np.asarray(sigma) * np.sqrt(1 - rho**2)
        error = self._values - self._problem.evaluate(x[:-2 * self._no])
        autocorr_error = error[1:] - rho * error[:-1]
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(autocorr_error**2, axis=0) / (2 * sigma**2))


class ARMA11LogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming ARMA(1,1) errors.

    The ARMA(1,1) model has 1 autoregressive term and 1 moving average term. It
    assumes that the errors :math:`\epsilon_i = x_i - f_i(\theta)` obey

    .. math::
        \epsilon_i = \rho \epsilon_{i-1} + \nu_i + \phi \nu_{i-1}

    where :math:`\nu_i` is IID Gaussian white noise with standard deviation
    :math:`\sigma'`.

    .. math::
        \sigma' = \sigma \sqrt{\frac{1 - \rho^2}{1 + 2  \phi  \rho + \phi^2}}

    This model is parameterised as such because it leads to a simple marginal
    distribution :math:`\epsilon_i \sim N(0, \sigma)`.

    Due to the complexity of the exact ARMA(1,1) likelihood, this class
    calculates a likelihood conditioned on initial values. This topic is
    discussed further in [2]_ , chapter 5.6. Thus, for a time series defined at
    points :math:`i=1,...,N`, summation begins at :math:`i=3`, and the
    conditional log-likelihood is

    .. math::
        L(\theta, \sigma, \rho, \phi|\boldsymbol{x}) =
            -\frac{N-2}{2} \log(2\pi)
            - (N-2) \log(\sigma')
            - \frac{1}{2\sigma'^2} \sum_{i=3}^N (\nu_i)^2

    where the values of :math:`\nu_i` are calculated from the observations
    according to

    .. math::
        \nu_i = \epsilon_i - \rho \epsilon_{i-1}
        - \phi (\epsilon_{i-1} - \rho \epsilon_{i-2})

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem three parameters are added (rho, phi, sigma),
        for a multi-output problem 3 * ``n_outputs`` parameters are added.

    References
    ----------
    .. [2] Hamilton, James D. Time series analysis. Vol. 2. New Jersey:
           Princeton, 1994.
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
        if any(sigma <= 0):
            return -np.inf
        sigma = (
            sigma *
            np.sqrt((1.0 - rho**2) / (1.0 + 2.0 * phi * rho + phi**2))
        )
        error = self._values - self._problem.evaluate(x[:-m])
        v = error[1:] - rho * error[:-1]
        autocorr_error = v[1:] - phi * v[:-1]
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(autocorr_error**2, axis=0) / (2 * sigma**2))


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

        # Distribution parameters
        sigma = np.asarray(x[-m:])
        if any(sigma <= 0):
            return -np.inf

        # problem parameters
        problem_parameters = x[:-m]
        error = self._values - self._problem.evaluate(problem_parameters)

        # Calculate
        return np.sum(
            - self._n_log_pi
            - n * np.log(sigma)
            - np.sum(np.log(1 + (error / sigma)**2), axis=0)
        )


class CensoredGaussianLogLikelihood(pints.ProblemLogLikelihood):
    r""" Calculates a log-likelihood assuming independent Gaussian noise at
    each time point, and that data above and below certain limits are censored.
    In other words, any data values less than the lower limit are only known to
    be less than or equal to it; and any values greater than the upper limit
    are only known to be greater than or equal to it. This likelihood is useful
    for censored data - see for instance [3]_. The parameter sigma represents
    the standard deviation of the noise on each output.

    For a noise level of ``sigma``, and left and right censoring (below the
    ``lower`` limit cutoff and above the ``upper`` limit), the likelihood
    becomes:

    .. math::
        \begin{align}
                L(\theta, \sigma|\boldsymbol{x})
                    = p(\boldsymbol{x} | \theta, \sigma)
                    =& \prod_{\text{lower} < x_j < \text{upper}}
                    \phi \left(\frac{x_j - f_j(\theta)}{\sigma} \right)
                    \prod_{x_j = \text{lower}} \Phi \left(\frac{\text{lower} -
                    f_j(\theta)}{\sigma} \right)
                    \\ &\times   \prod_{x_j = \text{upper}} \left(1 -
                    \Phi \left(\frac{\text{upper} - f_j(\theta)}{\sigma}
                    \right) \right),
        \end{align}

    where :math:`x_j` is the sampled data at time :math:`j` and :math:`f_j` is
    the simulated data at time :math:`j`. The functions :math:`\phi` and
    :math:`\Phi` are the standard normal distribution probability density
    function and cumulative distribution function respectively.

    This leads to a log-likelihood of:

    .. math::
        \begin{align}
        \log{L(\theta, \sigma|\boldsymbol{x})} =
        &   \sum_{\text{lower} < x_j < \text{upper}}{ \left( -\frac{1}{2}
        \log{2\pi}
            -\log{\sigma} -\frac{1}{2\sigma^2}(x_j - f_j(\theta))^2 \right)} \\
            &  + \sum_{x_j = \text{lower}} \log \left( \Phi
            \left(\frac{\text{lower} - f_j(\theta)}{\sigma} \right) \right) \\
            &    + \sum_{x_j = \text{upper}}  \log\left(1 - \Phi
            \left(\frac{\text{upper} - f_j(\theta)}{\sigma} \right) \right)
        \end{align}

    For a system with :math:`n_o` outputs, this becomes

    .. math::
        \begin{align}
        \log{L(\theta, \sigma|\boldsymbol{x})} =
        &   \sum_{\substack{\text{lower}_i < x_{ij} < \text{upper}_i, \\ 1
        \leq i \leq n_t, \\ 1 \leq j \leq n_o}}{ \left( -\frac{1}{2} \log{2\pi}
            -\log{\sigma_{i}} -\frac{1}{2\sigma_{i}^2}(x_{ij} - f_j(t_i,
            \theta ))^2 \right)} \\
            & + \sum_{\substack{x_{ij} = \text{lower}_{i}, \\1 \leq i \leq n_t,
            \\ 1 \leq j \leq n_o}} \log \left( \Phi
            \left(\frac{\text{lower}_{i} - f_j(t_i, \theta)}{\sigma_{i}}
            \right) \right) \\
            & + \sum_{\substack{x_{ij} = \text{upper}_{i}, \\1 \leq i
            \leq n_t, \\ 1 \leq j \leq n_o}}  \log\left(1 -
            \Phi \left(\frac{\text{upper}_{i} - f_j(t_i, \theta)}{\sigma_{i}}
            \right) \right)
        \end{align}

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`.
    lower
        The lower limit for censoring.
    upper
        The upper limit for censoring.

    References
    ----------
    .. [3] Beal, S. L. (2001). Ways to fit a PK model with some data below the
           quantification limit. Journal of pharmacokinetics and
           pharmacodynamics 28, pp. 481-504.

    """

    def __init__(self, problem, lower=None, upper=None):
        super(CensoredGaussianLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._no

        # Convert the lower and upper limits to the correct type
        a = self._convert_type(lower, limit_type="lower")
        b = self._convert_type(upper, limit_type="upper")

        if len(a) != self._no:
            raise ValueError(
                'Lower limit must be a ' +
                ' scalar or a vector of length n_outputs.')

        if len(b) != self._no:
            raise ValueError(
                'Upper limit must be a ' +
                ' scalar or a vector of length n_outputs.')

        diff = b - a
        if np.any(diff <= 0):
            raise ValueError('Upper limit ' +
                             'must exceed lower limit.')
        self._a = a
        self._b = b

        # Define the conditions for whether a point is lower censored,
        # upper censored and not censored
        self._lower_condition = self._values <= self._a
        self._upper_condition = self._values >= self._b
        self._not_censored_condition = (self._a < self._values
                                        ) & (self._values < self._b)

        # Number of points that aren't censored (for each observation
        # in the multioutput case)
        self._n_not_censored = np.sum(self._not_censored_condition, axis=0)

    def _convert_type(self, limit, limit_type="lower"):

        # If no lower or upper limit is supplied set it equal to
        # -/+ infinity
        if limit is None:
            if limit_type == "lower":
                limit = - np.inf
            elif limit_type == "upper":
                limit = np.inf

        # Convert the limit to an object of the correct type
        if np.isscalar(limit):
            limit = np.ones(self._no) * float(limit)
        else:
            limit = pints.vector(limit)

        return limit

    def __call__(self, x):
        theta = np.asarray(x[:-self._no])
        sigma = np.asarray(x[-self._no:])
        if any(sigma <= 0):
            return -np.inf

        # Evaluate the problem output - do this only once as this is usually
        # the most expensive step in inference, especially for ODE models
        output = self._problem.evaluate(theta)

        squared_error = np.sum((self._values - output)**2,
                               axis=0, where=self._not_censored_condition)

        # Calculate part of the likelihood corresponding to the censored data
        lower_censored_sum = np.sum(np.log(
            scipy.stats.norm.cdf(x=self._a, loc=output, scale=sigma)),
            where=self._lower_condition)
        upper_censored_sum = np.sum(
            np.log(1 - scipy.stats.norm.
                   cdf(x=self._b, loc=output, scale=sigma)),
            where=self._upper_condition)

        # Calculate part of the likelihood corresponding to
        # the data that isn't censored
        non_censored_sum = np.sum(- 0.5 * self._n_not_censored *
                                  np.log(2 * np.pi) -
                                  self._n_not_censored * np.log(sigma)
                                  - squared_error / (2 * sigma**2))

        return lower_censored_sum + upper_censored_sum + non_censored_sum

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        theta = np.asarray(x[:-self._no])
        sigma = np.asarray(x[-self._no:])

        # Calculate log-likelihood
        L = self.__call__(x)
        if np.isneginf(L):
            return L, np.tile(np.nan, self._n_parameters)

        # Evaluate, and get residuals
        y, dy = self._problem.evaluateS1(theta)

        # Reshape dy, in case we're working with a single-output problem
        dy = dy.reshape(self._nt, self._no, self._n_parameters - self._no)

        # Note: Must be (data - simulation), sign now matters!
        r = self._values - y

        # Evaluate the problem output - do this only once as this is usually
        # the most expensive step in inference, especially for ODE models
        output = self._problem.evaluate(theta)

        # Make conditions for where data isn't censored and is lower/upper
        # censored into 3D arrays

        if self._values.ndim == 1:
            where_condition = np.reshape(
                self._not_censored_condition,
                newshape=(np.shape(self._not_censored_condition)[0], 1, 1))
            lower_where_condition = np.reshape(
                self._lower_condition,
                newshape=(np.shape(self._lower_condition)[0], 1, 1))
            upper_where_condition = np.reshape(
                self._upper_condition,
                newshape=(np.shape(self._upper_condition)[0], 1, 1))
        else:
            where_condition = np.repeat(
                self._not_censored_condition[:, :, np.newaxis],
                np.shape(self._not_censored_condition)[-1], axis=2)
            lower_where_condition = np.repeat(
                self._lower_condition[:, :, np.newaxis],
                np.shape(self._lower_condition)[-1], axis=2)
            upper_where_condition = np.repeat(
                self._upper_condition[:, :, np.newaxis],
                np.shape(self._upper_condition)[-1], axis=2)

        # 1. Parts of the derivative corresponding to the data that
        # isn't censored

        # Calculate derivatives in the model parameters
        inner_deriv = (r.T * dy.T).T
        inner_sum = np.sum(inner_deriv, where=where_condition, axis=0)
        not_censored_dL = np.sum((sigma**(-2.0) * inner_sum.T).T, axis=0)

        # Calculate derivative wrt sigma
        not_censored_dsigma = -self._n_not_censored / sigma + sigma**(-3.0) *\
            np.sum((self._values - y)**2, axis=0,
                   where=self._not_censored_condition)

        # 2. Parts of the derivative corresponding to the data that is
        # censored

        # Use pdf(x-loc/scale) rather than pdf(x, loc, scale) as
        # pdf(x-loc/scale)= pdf(x, loc, scale) * scale
        # (whereas cdf(x-loc/scale) = cdf(x, loc, scale))

        lower_pdf = scipy.stats.norm.pdf((self._a - output) / sigma)
        lower_cdf = scipy.stats.norm.cdf(x=self._a, loc=output,
                                         scale=sigma)

        upper_pdf = scipy.stats.norm.pdf((self._b - output) / sigma)
        upper_cdf = scipy.stats.norm.cdf(x=self._b, loc=output,
                                         scale=sigma)

        # Calculate derivatives in the model parameters
        lower_numerator = lower_pdf.T * dy.T
        lower_denominator = lower_cdf.T
        lower_inner_val = (lower_numerator / lower_denominator).T
        lower_inner_sum = np.sum(lower_inner_val, where=lower_where_condition,
                                 axis=0)
        lower_censored_dL = -np.sum((sigma**(-1) * lower_inner_sum).T, axis=0)

        upper_numerator = upper_pdf.T * dy.T
        upper_denominator = 1 - upper_cdf.T
        upper_inner_val = (upper_numerator / upper_denominator).T
        upper_inner_sum = np.sum(upper_inner_val, where=upper_where_condition,
                                 axis=0)
        upper_censored_dL = np.sum((sigma**(-1) * upper_inner_sum).T, axis=0)

        # Calculate derivative wrt sigma
        lower_dsigma_inner_val = (lower_pdf.T *
                                  (self._values - y).T) / (lower_cdf.T)
        lower_censored_dsigma = - sigma**(-2) *\
            np.sum(lower_dsigma_inner_val.T,
                   where=self._lower_condition, axis=0).T

        upper_dsigma_inner_val = (upper_pdf.T *
                                  (self._values - y).T) / (1 - upper_cdf.T)
        upper_censored_dsigma = sigma**(-2) *\
            np.sum(upper_dsigma_inner_val.T,
                   where=self._upper_condition, axis=0).T

        dL = not_censored_dL + lower_censored_dL + upper_censored_dL
        dsigma = not_censored_dsigma + lower_censored_dsigma + \
            upper_censored_dsigma

        dL = np.concatenate((dL, np.array(list(dsigma))))

        # Return
        return L, dL

    def print_censored_values(self):

        # Print data and values that are censored
        print("The data are {}. \n The lower censored values"
              " are {}. \n The upper censored values"
              " are {}.".format(self._values,
                                np.extract(self._lower_condition,
                                           self._values),
                                np.extract(self._upper_condition,
                                           self._values)))


class ConstantAndMultiplicativeGaussianLogLikelihood(
        pints.ProblemLogLikelihood):
    r"""
    Calculates the log-likelihood assuming a mixed error model of a
    Gaussian base-level noise and a Gaussian heteroscedastic noise.

    For a time series model :math:`f(t| \theta)` with parameters :math:`\theta`
    , the ConstantAndMultiplicativeGaussianLogLikelihood assumes that the
    model predictions :math:`X` are Gaussian distributed according to

    .. math::
        X(t| \theta , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
        f(t| \theta) + (\sigma _{\text{base}} + \sigma _{\text{rel}}
        f(t| \theta)^\eta ) \, \epsilon ,

    where :math:`\epsilon` is a i.i.d. standard Gaussian random variable

    .. math::
        \epsilon \sim \mathcal{N}(0, 1).

    For each output in the problem, this likelihood introduces three new scalar
    parameters: a base-level scale :math:`\sigma _{\text{base}}`; an
    exponential power :math:`\eta`; and a scale relative to the model output
    :math:`\sigma _{\text{rel}}`.

    The resulting log-likelihood of a constant and multiplicative Gaussian
    error model is

    .. math::
        \log L(\theta, \sigma _{\text{base}}, \eta ,
        \sigma _{\text{rel}} | X^{\text{obs}})
        = -\frac{n_t}{2} \log 2 \pi
        -\sum_{i=1}^{n_t}\log \sigma _{\text{tot}, i}
        - \sum_{i=1}^{n_t}
        \frac{(X^{\text{obs}}_i - f(t_i| \theta))^2}
        {2\sigma ^2_{\text{tot}, i}},

    where :math:`n_t` is the number of measured time points in the time series,
    :math:`X^{\text{obs}}_i` is the observation at time point :math:`t_i`, and
    :math:`\sigma _{\text{tot}, i}=\sigma _{\text{base}} +\sigma _{\text{rel}}
    f(t_i| \theta)^\eta` is the total standard deviation of the error at time
    :math:`t_i`.

    For a system with :math:`n_o` outputs, this becomes

    .. math::
        \log L(\theta, \sigma _{\text{base}}, \eta ,
        \sigma _{\text{rel}} | X^{\text{obs}})
        = -\frac{n_tn_o}{2} \log 2 \pi
        -\sum_{j=1}^{n_0}\sum_{i=1}^{n_t}\log \sigma _{\text{tot}, ij}
        - \sum_{j=1}^{n_0}\sum_{i=1}^{n_t}
        \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
        {2\sigma ^2_{\text{tot}, ij}},

    where :math:`n_o` is the number of outputs of the model,
    :math:`X^{\text{obs}}_{ij}` is the observation at time point :math:`t_i`
    of output :math:`j`, and
    :math:`\sigma _{\text{tot}, ij}=\sigma _{\text{base}, j} +
    \sigma _{\text{rel}, j}f_j(t_i| \theta)^{\eta _j}` is the total standard
    deviation of the error at time :math:`t_i` of output :math:`j`.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    ``problem``
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem three parameters are added
        (:math:`\sigma _{\text{base}}`, :math:`\eta`,
        :math:`\sigma _{\text{rel}}`),
        for a multi-output problem :math:`3n_o` parameters are added
        (:math:`\sigma _{\text{base},1},\ldots , \sigma _{\text{base},n_o},
        \eta _1,\ldots , \eta _{n_o}, \sigma _{\text{rel},1}, \ldots ,
        \sigma _{\text{rel},n_o})`.
    """

    def __init__(self, problem):
        super(ConstantAndMultiplicativeGaussianLogLikelihood, self).__init__(
            problem)

        # Get number of times and number of noise parameters
        self._nt = len(self._times)
        self._no = problem.n_outputs()
        self._np = 3 * self._no

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._np

        # Pre-calculate the constant part of the likelihood
        self._logn = -0.5 * self._nt * self._no * np.log(2 * np.pi)

    def __call__(self, parameters):
        # Get parameters from input
        noise_parameters = np.asarray(parameters[-self._np:])
        sigma_base = noise_parameters[:self._no]
        eta = noise_parameters[self._no:2 * self._no]
        sigma_rel = noise_parameters[2 * self._no:]

        # Evaluate noise-free model (n_times, n_outputs)
        function_values = self._problem.evaluate(parameters[:-self._np])

        # Compute error (n_times, n_outputs)
        error = self._values - function_values

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * function_values**eta
        if np.any(np.asarray(sigma_tot) <= 0):
            return -np.inf

        # Compute log-likelihood
        # (inner sums over time points, outer sum over parameters)
        log_likelihood = self._logn - np.sum(
            np.sum(np.log(sigma_tot), axis=0)
            + 0.5 * np.sum(error**2 / sigma_tot**2, axis=0))

        return log_likelihood

    def evaluateS1(self, parameters):
        r"""
        See :meth:`LogPDF.evaluateS1()`.

        The partial derivatives of the log-likelihood w.r.t. the model
        parameters are

        .. math::
            \frac{\partial \log L}{\partial \theta _k}
            =& -\sum_{i,j}\sigma _{\text{rel},j}\eta _j\frac{
            f_j(t_i| \theta)^{\eta _j-1}}
            {\sigma _{\text{tot}, ij}}
            \frac{\partial f_j(t_i| \theta)}{\partial \theta _k}
            + \sum_{i,j}
            \frac{X^{\text{obs}}_{ij} - f_j(t_i| \theta)}
            {\sigma ^2_{\text{tot}, ij}}
            \frac{\partial f_j(t_i| \theta)}{\partial \theta _k} \\
            &+\sum_{i,j}\sigma _{\text{rel},j}\eta _j
            \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
            {\sigma ^3_{\text{tot}, ij}}f_j(t_i| \theta)^{\eta _j-1}
            \frac{\partial f_j(t_i| \theta)}{\partial \theta _k} \\
            \frac{\partial \log L}{\partial \sigma _{\text{base}, j}}
            =& -\sum ^{n_t}_{i=1}\frac{1}{\sigma _{\text{tot}, ij}}
            +\sum ^{n_t}_{i=1}
            \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
            {\sigma ^3_{\text{tot}, ij}} \\
            \frac{\partial \log L}{\partial \eta _j}
            =& -\sigma _{\text{rel},j}\eta _j\sum ^{n_t}_{i=1}
            \frac{f_j(t_i| \theta)^{\eta _j}\log f_j(t_i| \theta)}
            {\sigma _{\text{tot}, ij}}
            + \sigma _{\text{rel},j}\eta _j \sum ^{n_t}_{i=1}
            \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
            {\sigma ^3_{\text{tot}, ij}}f_j(t_i| \theta)^{\eta _j}
            \log f_j(t_i| \theta) \\
            \frac{\partial \log L}{\partial \sigma _{\text{rel},j}}
            =& -\sum ^{n_t}_{i=1}
            \frac{f_j(t_i| \theta)^{\eta _j}}{\sigma _{\text{tot}, ij}}
            + \sum ^{n_t}_{i=1}
            \frac{(X^{\text{obs}}_{ij} - f_j(t_i| \theta))^2}
            {\sigma ^3_{\text{tot}, ij}}f_j(t_i| \theta)^{\eta _j},

        where :math:`i` sums over the measurement time points and :math:`j`
        over the outputs of the model.
        """
        L = self.__call__(parameters)
        if np.isneginf(L):
            return L, np.tile(np.nan, self._n_parameters)

        # Get parameters from input
        # Shape sigma_base, eta, sigma_rel = (n_outputs,)
        noise_parameters = np.asarray(parameters[-self._np:])
        sigma_base = noise_parameters[:self._no]
        eta = noise_parameters[self._no:2 * self._no]
        sigma_rel = noise_parameters[-self._no:]

        # Evaluate noise-free model, and get residuals
        # y shape = (n_times,) or (n_times, n_outputs)
        # dy shape = (n_times, n_model_parameters) or
        # (n_times, n_outputs, n_model_parameters)
        y, dy = self._problem.evaluateS1(parameters[:-self._np])

        # Reshape y and dy, in case we're working with a single-output problem
        # Shape y = (n_times, n_outputs)
        # Shape dy = (n_model_parameters, n_times, n_outputs)
        y = y.reshape(self._nt, self._no)
        dy = np.transpose(
            dy.reshape(self._nt, self._no, self._n_parameters - self._np),
            axes=(2, 0, 1))

        # Compute error
        # Note: Must be (data - simulation), sign now matters!
        # Shape: (n_times, output)
        error = self._values.reshape(self._nt, self._no) - y

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * y**eta

        # Compute derivative w.r.t. model parameters
        dtheta = -np.sum(sigma_rel * eta * np.sum(
            y**(eta - 1) * dy / sigma_tot, axis=1), axis=1) + \
            np.sum(error * dy / sigma_tot**2, axis=(1, 2)) + np.sum(
                sigma_rel * eta * np.sum(
                    error**2 * y**(eta - 1) * dy / sigma_tot**3, axis=1),
                axis=1)

        # Compute derivative w.r.t. sigma base
        dsigma_base = - np.sum(1 / sigma_tot, axis=0) + np.sum(
            error**2 / sigma_tot**3, axis=0)

        # Compute derivative w.r.t. eta
        deta = -sigma_rel * (
            np.sum(y**eta * np.log(y) / sigma_tot, axis=0) -
            np.sum(
                error**2 / sigma_tot**3 * y**eta * np.log(y),
                axis=0))

        # Compute derivative w.r.t. sigma rel
        dsigma_rel = -np.sum(y**eta / sigma_tot, axis=0) + np.sum(
            error**2 / sigma_tot**3 * y**eta, axis=0)

        # Collect partial derivatives
        dL = np.hstack((dtheta, dsigma_base, deta, dsigma_rel))

        # Return
        return L, dL


class GaussianIntegratedLogUniformLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming independent Gaussian-distributed noise
    at each time point where :math:`p(\sigma)\propto 1/\sigma` has been
    integrated out of the joint posterior of :math:`p(\theta,\sigma|X)`,

    .. math::
        \begin{align} p(\theta|X) &= \int_{0}^{\infty} p(\theta, \sigma|X)
        \mathrm{d}\sigma\\
        &\propto \int_{0}^{\infty} p(X|\theta, \sigma) p(\theta, \sigma)
        \mathrm{d}\sigma.\end{align}

    Note that this is exactly the same statistical model as
    :class:`pints.GaussianLogLikelihood` with a uniform prior on
    :math:`\sigma` in the logarithmic transformed space.
    This is also known as Jeffrey's prior, as the standard deviation is a scale
    parameter, as discussed here: https://doi.org/10.1093/gji/ggaa168.

    A possible advantage of this log-likelihood compared with using a
    :class:`pints.GaussianLogLikelihood`, is that it has one fewer parameters
    (:math:`sigma`) which may speed up convergence to the posterior
    distribution, especially for multi-output problems which will have
    ``n_outputs`` fewer parameter dimensions.

    The log-likelihood is given in terms of the sum of squared errors:

    .. math::
        SSE = \sum_{i=1}^n (f_i(\theta) - y_i)^2,

    and is given up to a normalisation constant by:

    .. math::
        \begin{align}
        \text{log} L =
            & - n / 2 \text{log}(SSE) \\
            & - (n - 2) / 2 \text{log}(2) \\
            & + \text{log}\left[\Gamma(n / 2)\right],
        \end{align}

    where :math:`\Gamma(a)` is the incomplete gamma function.

    This log-likelihood is inherently a Bayesian method since it assumes a
    uniform prior on :math:`\sigma` _in the logarithmic transformed space.
    However using this likelihood in optimisation routines should yield the
    same estimates as the full :class:`pints.GaussianLogLikelihood`.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`.
    """

    def __init__(self, problem):
        super(GaussianIntegratedLogUniformLogLikelihood,
              self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters()

        # Pre-calculate
        n = self._nt
        self._n_over_2 = n / 2
        self._log_gamma = scipy.special.gammaln(self._n_over_2)
        self._constant_1 = -np.log(2) * (n - 2) / 2
        self._constant = self._constant_1 + self._log_gamma

    def __call__(self, x):
        error = self._values - self._problem.evaluate(x)
        sse = np.sum(error**2, axis=0)

        # Calculate
        sse = pints.vector(sse)
        return np.sum(self._constant - self._n_over_2 * np.log(sse))


class GaussianIntegratedUniformLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming independent Gaussian-distributed noise
    at each time point where :math:`\sigma\sim U(a,b)` has been integrated out
    of the joint posterior of :math:`p(\theta,\sigma|X)` in the same way as in
    :class:`pints.GaussianIntegratedLogUniformLogLikelihood`,

    .. math::
        \begin{align} p(\theta|X) &= \int_{0}^{\infty} p(\theta, \sigma|X)
        \mathrm{d}\sigma\\
        &\propto \int_{0}^{\infty} p(X|\theta, \sigma) p(\theta, \sigma)
        \mathrm{d}\sigma.\end{align}

    Note that this is exactly the same statistical model as
    :class:`pints.GaussianLogLikelihood` with a uniform prior on
    :math:`\sigma`.

    Similar to :class:`pints.GaussianIntegratedLogUniformLogLikelihood`, a
    possible advantage of this log-likelihood compared with using a
    :class:`pints.GaussianLogLikelihood`, is that it has one fewer parameters
    (:math:`sigma`) which may speed up convergence to the posterior
    distribution, especially for multi-output problems which will have
    ``n_outputs`` fewer parameter dimensions.

    The log-likelihood is given in terms of the sum of squared errors:

    .. math::
        SSE = \sum_{i=1}^n (f_i(\theta) - y_i)^2,

    and is given up to a normalisation constant by:

    .. math::
        \begin{align}
        \text{log} L =
            & - n / 2 \text{log}(\pi) \\
            & - \text{log}(2 (b - a) \sqrt(2)) \\
            & + (1 / 2 - n / 2) \text{log}(SSE) \\
            & + \text{log}\left[\Gamma((n - 1) / 2, \frac{SSE}{2 b^2}) -
                \Gamma((n - 1) / 2, \frac{SSE}{2 a^2}) \right],
        \end{align}

    where :math:`\Gamma(u,v)` is the upper incomplete gamma function as defined
    here: https://en.wikipedia.org/wiki/Incomplete_gamma_function.

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
        The lower limit on the uniform prior on `sigma`. Must be
        non-negative.
    upper
        The upper limit on the uniform prior on `sigma`.
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
        if any(sigma <= 0):
            return -np.inf
        error = self._values - self._problem.evaluate(x[:-self._no])
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        sigma = np.asarray(x[-self._no:])

        # Calculate log-likelihood
        L = self.__call__(x)
        if np.isneginf(L):
            return L, np.tile(np.nan, self._n_parameters)

        # Evaluate, and get residuals
        y, dy = self._problem.evaluateS1(x[:-self._no])

        # Reshape dy, in case we're working with a single-output problem
        dy = dy.reshape(self._nt, self._no, self._n_parameters - self._no)

        # Note: Must be (data - simulation), sign now matters!
        r = self._values - y

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
        import warnings
        warnings.warn(
            'The class `pints.KnownNoiseLogLikelihood` is deprecated.'
            ' Please use `pints.GaussianKnownSigmaLogLikelihood` instead.')
        super(KnownNoiseLogLikelihood, self).__init__(problem, sigma)


class LogNormalLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming independent log-normal noise at each
    time point, and adds a parameter representing the standard deviation
    (sigma) of the noise on the log scale for each output.

    Specifically, the sampling distribution takes the form:

    .. math::
        y(t) \sim \text{log-Normal}(\log f(\theta, t), \sigma)

    which can alternatively be written:

    .. math::
        \log y(t) \sim \text{Normal}(\log f(\theta, t), \sigma)

    Important to note is that:

    .. math::
        \mathbb{E}(y(t)) = f(\theta, t) \exp(\sigma^2 / 2)

    As such, the optional parameter `mean_adjust` (if true) adjusts the mean of
    the distribution so that its expectation is :math:`f(\theta, t)`. This
    shifted distribution is of the form:

    .. math::
        y(t) \sim \text{log-Normal}(\log f(\theta, t) - \sigma^2/2, \sigma)

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    problem
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem a single parameter is added, for a multi-output
        problem ``n_outputs`` parameters are added.
    mean_adjust
        A Boolean. Adjusts the distribution so that its mean corresponds to the
        function value. By default, this parameter if False.
    """

    def __init__(self, problem, mean_adjust=False):
        super(LogNormalLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._no

        # Pre-calculate parts
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)

        # For a log-normal sampling distribution any data points being below
        # zero would mean that the log-likelihood is always -infinity
        vals = np.asarray(self._values)
        if np.any(vals <= 0):
            raise ValueError('All data points must exceed zero.')
        self._log_values = np.log(self._values)

        self._mean_adjust = mean_adjust

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        if any(sigma < 0):
            return -np.inf

        soln = self._problem.evaluate(x[:-self._no])
        if np.any(soln < 0):
            return -np.inf

        error = np.log(self._values) - np.log(soln)
        if self._mean_adjust:
            error += 0.5 * sigma**2
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(self._log_values, axis=0)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        # Calculate log-likelihood and when log-prob is infinite, gradients are
        # not defined
        L = self.__call__(x)
        if L == -np.inf:
            return L, np.tile(np.nan, self._n_parameters)

        sigma = np.asarray(x[-self._no:])

        # Evaluate, and get residuals
        y, dy = self._problem.evaluateS1(x[:-self._no])

        # Reshape dy, in case we're working with a single-output problem
        dy = dy.reshape(self._nt, self._no, self._n_parameters - self._no)

        # Note: Must be (np.log(data) - simulation), sign matters now since it
        # must match that of the partial derivative below
        error = np.log(self._values) - np.log(y)
        if self._mean_adjust:
            error += 0.5 * sigma**2

        # Calculate derivatives in the model parameters
        dL = np.sum(
            (sigma**(-2.0) * np.sum((error.T * dy.T / y.T).T, axis=0).T).T,
            axis=0)

        # Calculate derivative wrt sigma
        dsigma = -self._nt / sigma + sigma**(-3.0) * np.sum(error**2, axis=0)
        if self._mean_adjust:
            dsigma -= 1 / sigma * np.sum(error, axis=0)

        dL = np.concatenate((dL, np.array(list(dsigma))))

        # Return
        return L, dL


class MultiplicativeGaussianLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates the log-likelihood for a time-series model assuming a
    heteroscedastic Gaussian error of the model predictions
    :math:`f(t, \theta )`.

    This likelihood introduces two new scalar parameters for each dimension of
    the model output: an exponential power :math:`\eta` and a scale
    :math:`\sigma`.

    A heteroscedascic Gaussian noise model assumes that the observable
    :math:`X` is Gaussian distributed around the model predictions
    :math:`f(t, \theta )` with a standard deviation that scales with
    :math:`f(t, \theta )`

    .. math::
        X(t) = f(t, \theta) + \sigma f(t, \theta)^\eta v(t)

    where :math:`v(t)` is a standard i.i.d. Gaussian random variable

    .. math::
        v(t) \sim \mathcal{N}(0, 1).

    This model leads to a log likelihood of the model parameters of

    .. math::
        \log{L(\theta, \eta , \sigma | X^{\text{obs}})} =
            -\frac{n_t}{2} \log{2 \pi}
            -\sum_{i=1}^{n_t}{\log{f(t_i, \theta)^\eta \sigma}}
            -\frac{1}{2}\sum_{i=1}^{n_t}\left(
                \frac{X^{\text{obs}}_{i} - f(t_i, \theta)}
                {f(t_i, \theta)^\eta \sigma}\right) ^2,

    where :math:`n_t` is the number of time points in the series, and
    :math:`X^{\text{obs}}_{i}` the measurement at time :math:`t_i`.

    For a system with :math:`n_o` outputs, this becomes

    .. math::
        \log{L(\theta, \eta , \sigma | X^{\text{obs}})} =
            -\frac{n_t n_o}{2} \log{2 \pi}
            -\sum ^{n_o}_{j=1}\sum_{i=1}^{n_t}{\log{f_j(t_i, \theta)^\eta
            \sigma _j}}
            -\frac{1}{2}\sum ^{n_o}_{j=1}\sum_{i=1}^{n_t}\left(
                \frac{X^{\text{obs}}_{ij} - f_j(t_i, \theta)}
                {f_j(t_i, \theta)^\eta \sigma _j}\right) ^2,

    where :math:`n_o` is the number of outputs of the model, and
    :math:`X^{\text{obs}}_{ij}` the measurement of output :math:`j` at
    time point :math:`t_i`.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    ``problem``
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem two parameters are added (:math:`\eta`,
        :math:`\sigma`), for a multi-output problem 2 times :math:`n_o`
        parameters are added.
    """

    def __init__(self, problem):
        super(MultiplicativeGaussianLogLikelihood, self).__init__(problem)

        # Get number of times and number of outputs
        self._nt = len(self._times)
        no = problem.n_outputs()
        self._np = 2 * no  # 2 parameters added per output

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._np

        # Pre-calculate the constant part of the likelihood
        self._logn = 0.5 * self._nt * no * np.log(2 * np.pi)

    def __call__(self, x):
        # Get noise parameters
        noise_parameters = x[-self._np:]
        eta = np.asarray(noise_parameters[0::2])
        sigma = np.asarray(noise_parameters[1::2])

        if any(sigma <= 0):
            return -np.inf

        # Evaluate function (n_times, n_output)
        function_values = self._problem.evaluate(x[:-self._np])

        # Compute likelihood
        log_likelihood = \
            -self._logn - np.sum(
                np.sum(np.log(function_values**eta * sigma), axis=0)
                + 0.5 / sigma**2 * np.sum(
                    (self._values - function_values)**2
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
        self._f = 1.0 / np.prod(self._values.shape)

    def __call__(self, x):
        return self._f * self._log_likelihood(x)

    def evaluateS1(self, x):
        """
        See :meth:`LogPDF.evaluateS1()`.

        This method only works if the underlying :class:`LogPDF` object
        implements the optional method :meth:`LogPDF.evaluateS1()`!
        """
        a, b = self._log_likelihood.evaluateS1(x)
        return self._f * a, self._f * np.asarray(b)


class StudentTLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates a log-likelihood assuming independent Student-t-distributed
    noise at each time point, and adds two parameters: one representing the
    degrees of freedom (``nu``), the other representing the scale (``sigma``).

    For a noise characterised by ``nu`` and ``sigma``, the log likelihood is of
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
        if any(nu <= 0) or any(sigma <= 0):
            return -np.inf

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
        import warnings
        warnings.warn(
            'The class `pints.UnknownNoiseLogLikelihood` is deprecated.'
            ' Please use `pints.GaussianLogLikelihood` instead.')
        super(UnknownNoiseLogLikelihood, self).__init__(problem)
