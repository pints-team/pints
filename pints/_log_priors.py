#
# Defines different prior distributions
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
import scipy
import scipy.special
import scipy.stats


class BetaLogPrior(pints.LogPrior):
    """
    Defines a beta (log) prior with given shape parameters ``a`` and ``b``,
    with pdf

    .. math::
        f(x|a,b) = \\frac{x^{a-1} (1-x)^{b-1}}{\\mathrm{B}(a,b)}

    where :math:`\\mathrm{B}` is the Beta function. This pdf has expectation

    .. math::
        \\mathrm{E}(X)=\\frac{a}{a+b}.

    For example, to create a prior with shape parameters ``a=5`` and ``b=1``,
    use::

        p = pints.BetaLogPrior(5, 1)

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, a, b):
        # Parse input arguments
        self._a = float(a)
        self._b = float(b)

        # Validate inputs
        if self._a <= 0:
            raise ValueError('Shape parameter a must be positive')
        if self._b <= 0:
            raise ValueError('Shape parameter b must be positive')

        # Cache constant
        self._log_beta = scipy.special.betaln(self._a, self._b)

    def __call__(self, x):
        if x[0] < 0.0 or x[0] > 1.0:
            return -float('inf')
        else:
            return scipy.special.xlogy(self._a - 1.0,
                                       x[0]) + scipy.special.xlog1py(
                self._b - 1.0, -x[0]) - self._log_beta

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)
        _x = x[0]

        # Account for pathological edges
        if _x == 0.0:
            _x = np.nextafter(0.0, 1.0)
        elif _x == 1.0:
            _x = np.nextafter(1.0, 0.0)

        if _x < 0.0 or _x > 1.0:
            return value, np.asarray([0.])
        else:
            # Use np.divide here to better handle possible v small denominators
            return value, np.asarray([np.divide(self._a - 1., _x) - np.divide(
                self._b - 1., 1. - _x)])

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.beta(self._a, self._b, size=(n, 1))


class CauchyLogPrior(pints.LogPrior):
    """
    Defines a 1-d Cauchy (log) prior with a given ``location``, and ``scale``,
    with pdf

    .. math::
        f(x|\\text{location}, \\text{scale}) = \\frac{1}{\\pi\\;\\text{scale}
        \\left[1 + \\left(\\frac{x-\\text{location}}{\\text{scale}}\\right)^2
        \\right]}

    and undefined expectation.

    For example, to create a prior centered around 0 and a scale of 5, use::

        p = pints.CauchyLogPrior(0, 5)

    Arguments:

    ``location``
        The center of the distribution.
    ``scale``
        The scale of the distribution.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, location, scale):
        # Test inputs
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')

        self._student_t = pints.StudentTLogPrior(location=location, df=1,
                                                 scale=scale)

    def __call__(self, x):
        return self._student_t(x)

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return self._student_t.sample(n)


class ComposedLogPrior(pints.LogPrior):
    """
    N-dimensional LogPrior composed of one or more other Ni-dimensional
    LogPriors, such that ``sum(Ni) = N``. The evaluation of the composed
    log-prior assumes the input log-priors are all independent from each other.

    For example, a composed log prior::

        p = pints.ComposedLogPrior(log_prior1, log_prior2, log_prior3)

    where ``log_prior1``, 2, and 3 each have dimension 1 will have dimension 3
    itself.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, *priors):
        # Check if sub-priors given
        if len(priors) < 1:
            raise ValueError('Must have at least one sub-prior')

        # Check if proper priors, count dimension
        self._n_parameters = 0
        for prior in priors:
            if not isinstance(prior, pints.LogPrior):
                raise ValueError('All sub-priors must extend pints.LogPrior.')
            self._n_parameters += prior.n_parameters()

        # Store
        self._priors = priors

    def __call__(self, x):
        output = 0
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            output += prior(x[lo:hi])
        return output

    def evaluateS1(self, x):
        """
        See :meth:`LogPDF.evaluateS1()`.

        *This method only works if the underlying :class:`LogPrior` classes all
        implement the optional method :class:`LogPDF.evaluateS1().`.*
        """
        output = 0
        doutput = np.zeros(self._n_parameters)
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            p, dp = prior.evaluateS1(x[lo:hi])
            output += p
            doutput[lo:hi] = np.asarray(dp)
        return output, doutput

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._n_parameters

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        output = np.zeros((n, self._n_parameters))
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            output[:, lo:hi] = prior.sample(n)
        return output


class ExponentialLogPrior(pints.LogPrior):
    """
    Defines an exponential (log) prior with given rate parameter ``rate`` with
    pdf

    .. math::
        f(x|\\text{rate}) = \\text{rate} \\; e^{-\\text{rate}\;x}

    and expectation

    .. math::
        \\mathrm{E}(X)=\\frac{1}{\\text{rate}}.

    For example, to create a prior with ``rate=0.5`` use::

        p = pints.ExponentialLogPrior(0.5)

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, rate):
        # Parse input arguments
        self._rate = float(rate)

        # Validate inputs
        if self._rate <= 0:
            raise ValueError('Rate parameter "scale" must be positive')

        # Cache constant
        self._log_scale = np.log(self._rate)

    def __call__(self, x):
        if x[0] < 0.0:
            return -float('inf')
        else:
            return self._log_scale - self._rate * x[0]

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)

        if x[0] < 0.0:
            return value, np.asarray([0.])
        else:
            return value, np.asarray([-self._rate])

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.exponential(scale=1. / self._rate, size=(n, 1))


class GammaLogPrior(pints.LogPrior):
    """
    Defines a gamma (log) prior with given shape parameter ``a`` and rate
    parameter ``b``, with pdf

    .. math::
        f(x|a,b)=\\frac{b^a x^{a-1} e^{-bx}}{\\mathrm{\\Gamma}(a)}

    where :math:`\\Gamma` is the Gamma function.  This pdf has expectation

    .. math::
        \\mathrm{E}(X)=\\frac{a}{b}.

    For example, to create a prior with shape parameters ``a=5`` and ``b=1``,
    use::

        p = pints.GammaLogPrior(5, 1)

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, a, b):
        # Parse input arguments
        self._a = float(a)
        self._b = float(b)

        # Validate inputs
        if self._a <= 0:
            raise ValueError('Shape parameter a must be positive')
        if self._b <= 0:
            raise ValueError('Rate parameter b must be positive')

        # Cache constant
        self._constant = scipy.special.xlogy(self._a,
                                             self._b) - scipy.special.gammaln(
            self._a)

    def __call__(self, x):
        if x[0] < 0.0:
            return -float('inf')
        else:
            return self._constant + scipy.special.xlogy(self._a - 1.,
                                                        x[0]) - self._b * x[0]

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)

        _x = x[0]

        # Account for pathological edge
        if _x == 0.0:
            _x = np.nextafter(0.0, 1.0)

        if _x < 0.0:
            return value, np.asarray([0.])
        else:
            # Use np.divide here to better handle possible v small denominators
            return value, np.asarray([np.divide(self._a - 1., _x) - self._b])

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.gamma(self._a, 1. / self._b, size=(n, 1))


class GaussianLogPrior(pints.LogPrior):
    """
    Defines a 1-d Gaussian (log) prior with a given ``mean`` and
    standard deviation ``sd``, with pdf

    .. math::
        f(x|\\text{mean},\\text{sd}) = \\frac{1}{\\sqrt{2\\pi\\;\\text{sd}^2}}
        \\text{exp}\\left(-\\frac{(x-\\text{mean})^2}{2\\;\\text{sd}^2}\\right)

    and expectation

    .. math::
        \\mathrm{E}(X)=\\text{mean}.

    For example, to create a prior with mean of ``0`` and a standard deviation
    of ``1``, use::

        p = pints.GaussianLogPrior(0, 1)

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, mean, sd):
        # Parse input arguments
        self._mean = float(mean)
        self._sd = float(sd)

        # Cache constants
        self._offset = np.log(1 / np.sqrt(2 * np.pi * self._sd ** 2))
        self._factor = 1 / (2 * self._sd ** 2)
        self._factor2 = 1 / self._sd**2

    def __call__(self, x):
        return self._offset - self._factor * (x[0] - self._mean)**2

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        return self(x), self._factor2 * (self._mean - np.asarray(x))

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.normal(self._mean, self._sd, size=(n, 1))


class HalfCauchyLogPrior(pints.LogPrior):
    """
    Defines a 1-d half-Cauchy (log) prior with a given ``location`` and
    ``scale``. This is a Cauchy distribution that has been truncated to lie in
    between [0, inf], with pdf

    .. math::
        f(x|\\text{location},\\text{scale})=\\begin{cases}\\frac{1}{\\pi\\;
        \\text{scale}\\left(\\frac{1}{\\pi}\\text{arctan}\\left(\\frac{
        \\text{location}}{\\text{scale} }\\right)+\\frac{1}{2}\\right)\\left(
        \\frac{(x-\\text{location})^2}{\\text{scale}^2}+1\\right)},&x>0\\\\0,&
        \\text{Otherwise}\\end{cases}

    and undefined expectation.

    For example, to create a prior centered around 0 and a scale of 5, use::

        p = pints.HalfCauchyLogPrior(0, 5)

    Arguments:

    ``location``
        The center of the distribution.`
    ``scale``
        The scale of the distribution.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, location, scale):
        # Test inputs
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')

        # Parse input arguments
        # set df for Student-t for sampling only
        self._df = 1
        self._location = float(location)
        self._scale = float(scale)

        # Cache constants
        self._first = np.log(2)
        self._log_scale = np.log(self._scale)
        self._log_weird = np.log(
            np.pi + 2 * np.arctan(self._location / self._scale))
        self._scale_sq = self._scale**2

        # Store scipy distribution object
        self._t = scipy.stats.t(
            df=self._df, loc=self._location, scale=self._scale)

    def __call__(self, x):
        if x[0] > 0:
            return (
                self._first + self._log_scale - self._log_weird
                - np.log((x[0] - self._location)**2 + self._scale_sq))
        else:
            return -float('inf')

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        samples = self._t.rvs(size=n)
        resample = samples <= 0
        n_resample = np.sum(resample)
        while n_resample:
            samples[resample] = self._t.rvs(size=n_resample)
            resample = samples <= 0
            n_resample = np.sum(resample)
        return samples


class MultivariateGaussianLogPrior(pints.LogPrior):
    """
    Defines a multivariate Gaussian (log)prior with a given ``mean`` and
    covariance matrix ``cov``, with pdf

    .. math::
        f(x|\\text{mean},\\text{cov}) = \\frac{1}{(2\\pi)^{d/2}|
        \\text{cov}|^{1/2}} \\text{exp}\\left(-\\frac{1}{2}(x-\\text{mean})'
        \\text{cov}^{-1}(x-\\text{mean})\\right)

    and expectation

    .. math::
        \\mathrm{E}(X)=\\text{mean}.

    For example, to create a prior with zero mean and identity covariance,
    use::

        p = pints.MultivariateGaussianLogPrior(
                np.array([0, 0]), np.array([[1, 0],[0, 1]]))

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, mean, cov):
        # Check input
        mean = pints.vector(mean)
        cov = np.array(cov, copy=True)
        cov.setflags(write=False)
        if cov.ndim != 2:
            raise ValueError('Given covariance must be a matrix.')
        if not (mean.shape[0] == cov.shape[0] == cov.shape[1]):
            raise ValueError('Sizes of mean and covariance do not match.')

        # Store
        self._mean = mean
        self._cov = cov
        self._n_parameters = mean.shape[0]

    def __call__(self, x):
        return np.log(
            scipy.stats.multivariate_normal.pdf(
                x, mean=self._mean, cov=self._cov))

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._n_parameters

    def sample(self, n=1):
        """ See :meth:`LogPrior.call()`. """
        # Note: size=n returns shape (n, d)
        return np.random.multivariate_normal(
            self._mean, self._cov, size=n)


class NormalLogPrior(GaussianLogPrior):
    """ Deprecated alias of :class:`GaussianLogPrior`. """

    def __init__(self, mean, standard_deviation):
        # Deprecated on 2019-02-06
        import logging
        logging.basicConfig()
        log = logging.getLogger(__name__)
        log.warning(
            'The class `pints.NormalLogPrior` is deprecated.'
            ' Please use `pints.GaussianLogPrior` instead.')
        super(NormalLogPrior, self).__init__(mean, standard_deviation)


class StudentTLogPrior(pints.LogPrior):
    """
    Defines a 1-d Student-t (log) prior with a given ``location``,
    degrees of freedom ``df``,  and ``scale`` with pdf

    .. math::
        f(x|\\text{location},\\text{scale},\\text{df})=\\frac{\\left(\\frac{
        \\text{df}}{\\text{df}+\\frac{(x-\\text{location})^2}{\\text{scale}^2}}
        \\right)^{\\frac{\\text{df}+1}{2}}}{\\sqrt{\\text{df}}\\;\\text{scale}
        \\;\\mathrm{B}\\left(\\frac{\\text{df} }{2},\\frac{1}{2}\\right)}

    where :math:`\\mathrm{B}` is the Beta function. This pdf has expectation

    .. math::
        \\mathrm{E}(X)=\\begin{cases}\\text{location},&\\text{df}>1\\\\
        \\text{undefined},&\\text{otherwise.}\\end{cases}

    For example, to create a prior centered around 0 with 3 degrees of freedom
    and a scale of 1, use::

        p = pints.StudentTLogPrior(0, 3, 1)

    Arguments:

    ``location``
        The center of the distribution.
    ``df``
        The number of degrees of freedom of the distribution.
    ``scale``
        The scale of the distribution.

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, location, df, scale):
        # Test inputs
        if float(df) <= 0:
            raise ValueError('Degrees of freedom must be positive')
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')

        # Parse input arguments
        self._df = float(df)
        self._location = float(location)
        self._scale = float(scale)

        # Cache constants
        self._log_df = np.log(self._df)

        self._1_sig_sq = 1. / (self._scale * self._scale)

        self._first = 0.5 * (1.0 + self._df)

        self._samp_const = scipy.special.xlogy(-0.5, self._df) - np.log(
            self._scale) - scipy.special.betaln(0.5 * self._df, 0.5)

        self._deriv_const = (-1. - self._df) * self._1_sig_sq

    def __call__(self, x):
        return self._samp_const + self._first * (self._log_df - np.log(
            self._df + self._1_sig_sq * (x[0] - self._location) ** 2))

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        offset = x[0] - self._location
        return self(x), np.asarray([offset * self._deriv_const / (
            self._df + offset * offset * self._1_sig_sq)])

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return scipy.stats.t.rvs(df=self._df, loc=self._location,
                                 scale=self._scale, size=n)


class UniformLogPrior(pints.LogPrior):
    """
    Defines a uniform prior over a given range.

    The range includes the lower, but not the upper boundaries, so that any
    point ``x`` with a non-zero prior must have ``lower <= x < upper``.

    In 1D this has pdf

    .. math::
        f(x|\\text{lower},\\text{upper})=\\begin{cases}0,&\\text{if }x\\not\\in
        [\\text{lower},\\text{upper})\\\\\\frac{1}{\\text{upper}-\\text{lower}}
        ,&\\text{if }x\\in[\\text{lower},\\text{upper})\\end{cases}

    and expectation

    .. math::
        \\mathrm{E}(X)=\\frac{1}{2}(\\text{lower}+\\text{upper}).

    For example, to create a prior with :math:`x\\in[0,4]`, :math:`y\\in[1,5]`,
    and :math:`z\\in[2,6]` use either::

        p = pints.UniformLogPrior([0, 1, 2], [4, 5, 6])

    or::

        p = pints.UniformLogPrior(RectangularBoundaries([0, 1, 2], [4, 5, 6]))

    *Extends:* :class:`LogPrior`
    """
    def __init__(self, lower_or_boundaries, upper=None):
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries, pints.Boundaries):
                raise ValueError(
                    'UniformLogPrior requires a lower and an upper bound, or a'
                    ' single Boundaries object.')
            self._boundaries = lower_or_boundaries
        else:
            self._boundaries = pints.RectangularBoundaries(
                lower_or_boundaries, upper)

        # Cache dimension
        self._n_parameters = self._boundaries.n_parameters()

        # Minimum output value
        self._minf = -float('inf')

        # Maximum output value
        # Use normalised value (1/area) for rectangular boundaries,
        # otherwise just use 1.
        if isinstance(self._boundaries, pints.RectangularBoundaries):
            self._value = -np.log(np.product(self._boundaries.range()))
        else:
            self._value = 1

    def __call__(self, x):
        return self._value if self._boundaries.check(x) else self._minf

    def evaluateS1(self, x):
        """ See :meth:`LogPrior.evaluateS1()`. """
        # Ignoring points on the boundaries (i.e. on the surface of the
        # hypercube), because it's very unlikely and won't help the search
        # much...
        return self(x), np.zeros(self._n_parameters)

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._n_parameters

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return self._boundaries.sample(n)
