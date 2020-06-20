#
# Transformation functions
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from scipy.special import logit, expit


class Transform(object):
    """
    Abstract base class for objects that provide some convenience parameter
    transformation from the model parameter space to a search space.

    If ``t`` is an instance of a :class:`Transform` class, you can apply
    the transformation from the model parameter space ``p`` to the search
    space ``x`` by using ``x = t.to_search(p)`` and the inverse by using
    ``p = t.to_model(x)``.
    """
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
    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        return np.diag(np.exp(x))

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        return np.sum(np.exp(x))

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
    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        return np.diag(expit(x) * (1. - expit(x)))

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        return np.log(np.sum(expit(x) * (1. - expit(x))))

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        return expit(x)

    def to_search(self, p):
        """ See :meth:`Transform.to_search()`. """
        return logit(p)
