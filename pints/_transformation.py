#
# Transformation functions
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from scipy.special import logit, expit


class Transform(object):
    """
    Abstract base class for objects that provide some convenience parameter
    transformation from the model parameter space to a search space.

    If ``t`` is an instance of a ``Transform`` class, you can apply
    the transformation from the model parameter space ``p`` to the search
    space ``x`` by using ``x = t.to_search(p)`` and the inverse by using
    ``p = t.to_model(x)``.
    """
    def apply_log_pdf(self, log_pdf):
        """
        Returns a transformed log-PDF class.
        """
        return TransformedLogPDF(log_pdf, self)

    def apply_error_measure(self, error_measure):
        """
        Returns a transformed error measure class.
        """
        return TransformedErrorMeasure(error_measure, self)

    def apply_boundaries(self, boundaries):
        """
        Returns a transformed boundaries class.
        """
        return TransformedBoundaries(boundaries, self)

    def jacobian(self, x):
        """
        Returns the Jacobian for the parameter ``x`` in the search space.
        """
        raise NotImplementedError

    def log_jacobian_det(self, x):
        """
        Returns the logarithm of the absolute value of the Jacobian
        determinant for the parameter ``x`` in the search space.

        *This is an optional method; it is needed when transformation is
        performed on :class:`LogPDF`, but not necessary if it's used for
        :class:`ErrorMeasure`.*
        """
        return np.log(np.abs(np.linalg.det(self.jacobian(x))))

    def n_parameters(self):
        """
        Returns the dimension of the parameter space this transformation is
        defined over.
        """
        raise NotImplementedError

    def to_model(self, x):
        """
        Returns the inverse of transformation from the search space ``x`` to
        the model parameter space ``p``.
        """
        raise NotImplementedError

    def to_search(self, p):
        """
        Returns the forward transformation from the model parameter space
        ``p`` to the search space ``x``.
        """
        raise NotImplementedError


class TransformedLogPDF(pints.LogPDF):
    """
    A log-PDF that is transformed from the model space to the search space.
    """
    def __init__(self, log_pdf, transform):
        self._log_pdf = log_pdf
        self._transform = transform
        self._n_parameters = self._log_pdf.n_parameters()

    def __call__(self, x):
        logpdf_nojac = self.logpdf_nojac(x)
        log_jacobian_det = self._transform.log_jacobian_det(x)
        return logpdf_nojac + log_jacobian_det

    #TODO evaluateS1?

    def logpdf_nojac(self, x):
        """
        Returns log-PDF value of the transformed distribution evaluated at
        ``x`` without the Jacobian adjustment term.
        """
        return self._log_pdf(self._transform.to_model(x))

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters


class TransformedErrorMeasure(pints.ErrorMeasure):
    """
    An error measure that is transformed from the model space to the search
    space.
    """
    def __init__(self, function, transform):
        self._function = function
        self._transform = transform
        self._n_parameters = self._function.n_parameters()

    def __call__(self, x):
        # Get parameters at the model space
        p = self._transform.to_model(x)
        return self._function(p)

    #TODO evaluateS1?

    def n_parameters(self):
        """ See :meth:`ErrorMeasure.n_parameters()`. """
        return self._n_parameters


class TransformedBoundaries(pints.Boundaries):
    """
    Boundaries that are transformed from the model space to the search space.
    """
    def __init__(self, boundaries, transform):
        self._boundaries = boundaries
        self._transform = transform
        self._n_parameters = self._boundaries.n_parameters()

    def check(self, x):
        # Get parameters at the model space
        p = self._transform.to_model(x)
        return self._boundaries.check(p)

    def n_parameters(self):
        """ See :meth:`Boundaries.n_parameters()`. """
        return self._n_parameters

    def range(self):
        """
        Returns the size of the search space (i.e. ``upper - lower``).
        """
        upper = self._transform.to_search(self._boundaries.upper())
        lower = self._transform.to_search(self._boundaries.lower())
        return upper - lower


class LogTransform(Transform):
    r"""
    Logarithm transformation of the model parameters:

    .. math::
        x = \log(p),

    where :math:`p` is the model parameter vector and :math:`x` is the
    search space vector.

    The Jacobian adjustment of the log transformation is given by

    .. math::
        |\frac{d}{dx} \exp(x)| = \exp(x).

    Extends :class:`Transform`.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        return np.diag(np.exp(x))

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        return np.sum(x)

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        return np.exp(x)

    def to_search(self, p):
        """ See :meth:`Transform.to_search()`. """
        return np.log(p)


class LogitTransform(Transform):
    r"""
    Logit (or log-odds) transformation of the model parameters:

    .. math::
        x = \text{logit}(p) = \log(\frac{p}{1 - p}),

    where :math:`p` is the model parameter vector and :math:`x` is the
    search space vector.

    The Jacobian adjustment of the logit transformation is given by

    .. math::
        |\frac{d}{dx} \text{logit}^{-1}(x)| = \text{logit}^{-1}(x) \times
        (1 - \text{logit}^{-1}(x)).

    Extends :class:`Transform`.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        return np.diag(expit(x) * (1. - expit(x)))

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        return np.sum(np.log(expit(x)) + np.log(1. - expit(x)))

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        return expit(x)

    def to_search(self, p):
        """ See :meth:`Transform.to_search()`. """
        return logit(p)


class RectangularBoundariesTransform(Transform):
    r"""
    A generalised version of logit transformation for the model parameters,
    which transform an interval or ractangular boundaries :math:`[a, b)` to
    all real number:

    .. math::
        x = f(p) = \log(p - a) - \log(b - p),

    where :math:`p` is the model parameter vector and :math:`x` is the
    search space vector. The range includes the lower (:math:`a`), but not the
    upper (:math:`b`) boundaries. Note that :class:`LogitTransform` is a
    special case where :math:`a = 0` and :math:`b = 1`.

    The Jacobian adjustment of the transformation is given by

    .. math::
        |\frac{d}{dx} f^{-1}(x)| = \frac{b - a}{\exp(x) (1 + \exp(-x)) ^ 2}
        \log|\frac{d}{dx} f^{-1}(x)| = \log(b - a) - 2 \log(1 + \exp(-x)) - x

    For example, to create a transform with :math:`p_1 \in [0, 4)`,
    :math:`p_2 \in [1, 5)`, and :math:`p_3 \in [2, 6)` use either::

        transform = pints.IntervalTransform([0, 1, 2], [4, 5, 6])

    or::

        boundaries = pints.RectangularBoundaries([0, 1, 2], [4, 5, 6])
        transform = pints.IntervalTransform(boundaries)

    Extends :class:`Transform`.
    """
    def __init__(self, lower_or_boundaries, upper=None):
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries,
                              pints.RectangularBoundaries):
                raise ValueError(
                    'IntervalTransform requires a lower and an upper bound, '
                    'or a single RectangularBoundaries object.')
            boundaries = lower_or_boundaries
        else:
            # Create RectangularBoundaries for all the input checks
            boundaries = pints.RectangularBoundaries(lower_or_boundaries,
                                                     upper)

        self._a = boundaries.lower()
        self._b = boundaries.upper()

        # Cache dimension
        self._n_parameters = boundaries.n_parameters()
        del(boundaries)

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        x = np.asarray(x)
        diag = (self._b - self._a) / (np.exp(x) * (1. + np.exp(-x)) ** 2)
        return np.diag(diag)

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        x = np.asarray(x)
        s = self._softplus(-x)
        return np.sum(np.log(self._b - self._a) - 2. * s - x)

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        x = np.asarray(x)
        return (self._b - self._a) * expit(x) + self._a

    def to_search(self, p):
        p = np.asarray(p)
        """ See :meth:`Transform.to_search()`. """
        return np.log(p - self._a) - np.log(self._b - p)

    def _softplus(self, x):
        """ Returns the softplus function. """
        return np.log(1. + np.exp(x))


class IdentityTransform(Transform):
    """
    Identity transformation does nothing to the input parameters, i.e. the
    search space under this transformation is the same as the model space.
    And its Jacobian matrix is the identity matrix.

    Extends :class:`Transform`.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        return np.eye(self._n_parameters)

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters

    def to_model(self, x, *args, **kwargs):
        """ See :meth:`Transform.to_model()`. """
        return np.asarray(x, *args, **kwargs)

    def to_search(self, p, *args, **kwargs):
        """ See :meth:`Transform.to_search()`. """
        return np.asarray(p, *args, **kwargs)


class ComposedTransform(Transform):
    r"""
    N-dimensional :class:`Transform` composed of one or more other
    :math:`N_i`-dimensional ``Transform``, such that :math:`\sum _i N_i = N`.
    The evaluation and transformation of the composed transformations assume
    the input transformations are all independent from each other.

    For example, a composed transform

        ``t = pints.ComposedTransform(transform1, transform2, transform3)``,

    where ``transform1``, ``transform2``, and ``transform3`` each have
    dimension 1, 2 and 1, will have dimension 4.

    The dimensionality of the individual priors does not have to be the same,
    i.e. :math:`N_i\neq N_j` is allowed.

    The input parameters of the :class:`ComposedTransform` have to be ordered
    in the same way as the individual tranforms for the parameter vector. In
    the above example the transform may be performed by ``t.to_search(p)``,
    where:

        ``p = [parameter1_transform1, parameter1_transform2,
        parameter2_transform2, parameter1_transform3]``.

    Extends :class:`Transform`.
    """
    def __init__(self, *transforms):
        # Check if sub-transforms given
        if len(transforms) < 1:
            raise ValueError('Must have at least one sub-transform.')

        # Check if proper transform, count dimension
        self._n_parameters = 0
        for transform in transforms:
            if not isinstance(transform, pints.Transform):
                raise ValueError('All sub-transforms must extend '
                                 'pints.Transform.')
            self._n_parameters += transform.n_parameters()

        # Store
        self._transforms = transforms

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        x = np.asarray(x)
        lo, hi = 0, self._transforms[0].n_parameters()
        output = self._transforms[0].jacobian(x[lo:hi])
        for transform in self._transforms[1:]:
            lo = hi
            hi += transform.n_parameters()
            jaco = transform.jacobian(x[lo:hi])
            pack = np.zeros((output.shape[0], jaco.shape[1]))
            output = np.block([[output, pack], [pack.T, jaco]])
        return output

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        x = np.asarray(x)
        output = 0
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            output += transform.log_jacobian_det(x[lo:hi])
        return output

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        x = np.asarray(x)
        if np.product(x.shape) == self._n_parameters:
            x = x.reshape((self._n_parameters,))
            single = True
        else:
            single = False
        output = np.zeros(x.shape)
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            if single:
                output[lo:hi] = transform.to_model(x[lo:hi])
            else:
                output[:, lo:hi] = transform.to_model(x[:, lo:hi])
        return output

    def to_search(self, p):
        """ See :meth:`Transform.to_search()`. """
        p = np.asarray(p)
        if np.product(p.shape) == self._n_parameters:
            p = p.reshape((self._n_parameters,))
            single = True
        else:
            single = False
        output = np.zeros(p.shape)
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            if single:
                output[lo:hi] = np.asarray(transform.to_search(p[lo:hi]))
            else:
                output[:, lo:hi] = np.asarray(transform.to_search(p[:, lo:hi]))
        return output

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters
