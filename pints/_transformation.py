#
# Classes that automate parameter transformation
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


class Transformation(object):
    """
    Abstract base class for objects that provide transformations between two
    parameter spaces: the model parameter space and a search space.

    If ``trans`` is an instance of a ``Transformation`` class, you can apply
    the transformation of a parameter vector from the model space ``p`` to the
    search space ``q`` by using ``q = trans.to_search(p)`` and the inverse by
    using ``p = trans.to_model(q)``.
    """
    def convert_log_pdf(self, log_pdf):
        """
        Returns a transformed log-PDF class.
        """
        return TransformedLogPDF(log_pdf, self)

    def convert_log_prior(self, log_prior):
        """
        Returns a transformed log-prior class.
        """
        return TransformedLogPrior(log_prior, self)

    def convert_error_measure(self, error_measure):
        """
        Returns a transformed error measure class.
        """
        return TransformedErrorMeasure(error_measure, self)

    def convert_boundaries(self, boundaries):
        """
        Returns a transformed boundaries class.
        """
        return TransformedBoundaries(boundaries, self)

    def convert_covariance_matrix(self, C, q):
        r"""
        Converts a convariance matrix ``C`` from the model space to the search
        space around a parameter vector ``q`` provided in the search space.

        The transformation is performed using a linear approximation [1]_ with
        the Jacobian :math:`J`:

        .. math::

            C(q) &= \frac{dg(p)}{dp} C(p) (\frac{dg(p)}{dp})^T \\
                 &= J^{-1}(q) C(p) (J^{-1}(q))^T.

        Using the property that :math:`J^{-1} = \frac{dg}{dp}`, from the
        inverse function theorem, i.e. the matrix inverse of the Jacobian
        matrix of an invertible function is the Jacobian matrix of the inverse
        function.

        References
        ----------
        .. [1] How to Obtain Those Nasty Standard Errors From Transformed Data
               Erik Jorgensen and Asger Roer Pedersen,
               http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.47.9023
        """
        jac_inv = np.linalg.pinv(self.jacobian(q))
        return np.matmul(np.matmul(jac_inv, C), jac_inv.T)

    def convert_standard_deviation(self, s, q):
        r"""
        Converts standard deviation ``s``, either a scalar or a vector, from
        the model space to the search space around a parameter vector ``q``
        provided in the search space.

        The transformation is performed using a linear approximation [1]_ with
        the Jacobian :math:`J`:

        .. math::

            C(q) &= \frac{dg(p)}{dp} C(p) (\frac{dg(p)}{dp})^T \\
                 &= J^{-1}(q) C(p) (J^{-1}(q))^T.

        Using the property that :math:`J^{-1} = \frac{dg}{dp}`, from the
        inverse function theorem, i.e. the matrix inverse of the Jacobian
        matrix of an invertible function is the Jacobian matrix of the inverse
        function.

        If :math:`C(p) = var(p) \cdot I`, where :math:`var(.)` is the variance
        and :math:`I` is the identity matrix, then

        .. math::
            C(q) = var(q) \cdot I = diag(J^{-1}(q))^2 var(p) I

        Therefore the standard deviation ``s`` can be transformed using

        s(q) = diag(J^{-1}(q)) \times s(p)

        References
        ----------
        .. [1] How to Obtain Those Nasty Standard Errors From Transformed Data
               Erik Jorgensen and Asger Roer Pedersen,
               http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.47.9023
        """
        jac_inv = np.linalg.pinv(self.jacobian(q))
        return s * np.diagonal(jac_inv)

    def jacobian(self, q):
        r"""
        Returns the Jacobian matrix of the transformation calculated at the
        parameter vector ``q`` in the search space. For a transformation
        :math:`q = f(p)`, the Jacobian matrix is defined as

        .. math::
            J = [\frac{\partial f^{-1}}{\partial q_1} \quad
                 \frac{\partial f^{-1}}{\partial q_2} \quad \cdots].

        *This is an optional method. It is needed when transformation of
        standard deviation :meth:`Transformation.convert_standard_deviation` or
        covariance matrix :meth:`Transformation.convert_covariance_matrix` is
        needed, or when ``evaluateS1()`` is needed.*
        """
        raise NotImplementedError

    def log_jacobian_det(self, q):
        """
        Returns the logarithm of the absolute value of the determinant of the
        Jacobian matrix of the transformation :meth:`Transformation.jacobian`
        calculated at the parameter vector ``q`` in the search space.

        *This is an optional method. It is needed when transformation is
        performed on :class:`LogPDF` and/or that requires ``evaluateS1()``;
        e.g. not necessary if it's used for :class:`ErrorMeasure` without
        :meth:`ErrorMeasure.evaluateS1()`.*
        """
        return np.log(np.abs(np.linalg.det(self.jacobian(q))))

    def log_jacobian_det_S1(self, q):
        """
        Computes the logarithm of the absolute value of the determinant of the
        Jacobian, and returns the result plus the partial derivatives of the
        result with respect to the parameters.

        The returned data is a tuple ``(S, S')`` where ``S`` is a scalar value
        and ``S'`` is a sequence of length ``n_parameters``.

        Note that the derivative returned is of the log of the determinant of
        the Jacobian, so ``S' = d/dq log(|det(J(q))|)``, evaluated at input.

        *This is an optional method.*
        """
        raise NotImplementedError

    def multiple_to_model(self, qs):
        """
        Transforms a series of parameter vectors ``qs`` from the search space
        to the model space. ``qs`` must be provided in the shape
        (``n_vectors``, ``n_parameters``).
        """
        qs = np.asarray(qs)
        ps = np.zeros(qs.shape)
        for i, q in enumerate(qs):
            ps[i, :] = self.to_model(q)
        return ps

    def n_parameters(self):
        """
        Returns the dimension of the parameter space this transformation is
        defined over.
        """
        raise NotImplementedError

    def to_model(self, q):
        """
        Transforms a parameter vector ``q`` from the search space to the model
        space.
        """
        raise NotImplementedError

    def to_search(self, p):
        """
        Transforms a parameter vector ``p`` from the model space to the search
        space.
        """
        raise NotImplementedError


class ElementWiseTransformation(Transformation):
    r"""
    Abstract base class for objects that provide *element-wise* transformations
    between two parameter spaces: the model parameter space and a search space.
    That is, for a parameter vector :math:`\boldsymbol{p}` in the model space
    and a parameter vector :math:`\boldsymbol{q}` in the search space, the
    transformation :math:`\boldsymbol{f}` relating the two vectors takes the
    form

    .. math::
        q_i = f_i(p_i),

    where :math:`x_i` denotes the :math:`i^{\text{th}}` element of the vector
    :math:`\boldsymbol{x}`.
    """


class TransformedBoundaries(pints.Boundaries):
    """
    A :class:`pints.Boundaries` that accepts parameters in a transformed
    search space.

    Extends :class:`pints.Boundaries`.

    Parameters
    ----------
    boundaries
        A :class:`pints.Boundaries`.
    transform
        A :class:`pints.Transformation`.
    """
    def __init__(self, boundaries, transform):
        self._boundaries = boundaries
        self._transform = transform
        self._n_parameters = self._boundaries.n_parameters()

        if self._transform.n_parameters() != self._n_parameters:
            raise ValueError('Number of parameters for boundaries and '
                             'transform must match.')

    def check(self, q):
        """ See :meth:`Boundaries.check()`. """
        # Get parameters in the model space
        p = self._transform.to_model(q)
        # Check Boundaries in the model space
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


class TransformedErrorMeasure(pints.ErrorMeasure):
    r"""
    A :class:`pints.ErrorMeasure` that accepts parameters in a transformed
    search space.

    For the first order sensitivity of a :class:`pints.ErrorMeasure` :math:`E`
    and a :class:`pints.Transformation`
    :math:`\boldsymbol{q} = \boldsymbol{f}(\boldsymbol{p})`, the transformation
    is done using

    .. math::
        \frac{\partial E(\boldsymbol{q})}{\partial q_i} &=
        \frac{\partial E(\boldsymbol{f}^{-1}(\boldsymbol{q}))}{\partial q_i}\\
        &= \sum_l \frac{\partial E(\boldsymbol{p})}{\partial p_l}
        \frac{\partial p_l}{\partial q_i}.

    Extends :class:`pints.ErrorMeasure`.

    Parameters
    ----------
    error
        A :class:`pints.ErrorMeasure`.
    transform
        A :class:`pints.Transformation`.
    """
    def __init__(self, error, transform):
        self._error = error
        self._transform = transform
        self._n_parameters = self._error.n_parameters()
        if self._transform.n_parameters() != self._n_parameters:
            raise ValueError('Number of parameters for error and transform '
                             'must match.')

    def __call__(self, q):
        # Get parameters in the model space
        p = self._transform.to_model(q)
        # Compute ErrorMeasure in the model space
        return self._error(p)

    def evaluateS1(self, q):
        """ See :meth:`ErrorMeasure.evaluateS1()`. """

        # Get parameters in the model space
        p = self._transform.to_model(q)

        # Compute evaluateS1 of ErrorMeasure in the model space
        e, de_nojac = self._error.evaluateS1(p)

        # Calculate the S1 using change of variable
        # Wikipedia: https://w.wiki/Us8
        #
        # This can be done in matrix form, for Jacobian matrix J and
        # E = log(pi(p)):
        #
        # (\nabla_q E)^T = J_(E.g) = J_(E(p)) J_(g) = (\nabla_p E)^T J_(g)
        # (\nabla denotes the del operator)
        #
        jacobian = self._transform.jacobian(q)
        de = np.matmul(de_nojac, jacobian)  # Jacobian must be the second term

        return e, de

    def n_parameters(self):
        """ See :meth:`ErrorMeasure.n_parameters()`. """
        return self._n_parameters


class TransformedLogPDF(pints.LogPDF):
    r"""
    A :class:`pints.LogPDF` that accepts parameters in a transformed search
    space.

    When a :class:`TransformedLogPDF` object (initialised with a
    :class:`pints.LogPDF` of :math:`\pi(\boldsymbol{p})` and a
    :class:`Transformation` of
    :math:`\boldsymbol{q} = \boldsymbol{f}(\boldsymbol{p})`) is called with a
    vector argument :math:`\boldsymbol{q}` in the search space, it returns
    :math:`\log(\pi(\boldsymbol{q}))`` where :math:`\pi(\boldsymbol{q})` is the
    transformed unnormalised PDF of the input PDF, using

    .. math::
        \pi(\boldsymbol{q}) = \pi(\boldsymbol{f}^{-1}(\boldsymbol{q}))
            \,\, |det(\mathbf{J}(\boldsymbol{f}^{-1}(\boldsymbol{q})))|.

    :math:`\mathbf{J}` is the Jacobian matrix:

    .. math::
        \mathbf{J} =
            [\frac{\partial \boldsymbol{f}^{-1}}{\partial q_1} \quad
             \frac{\partial \boldsymbol{f}^{-1}}{\partial q_2} \quad \cdots].

    Hence

    .. math::
        \log(\pi(\boldsymbol{q})) =
            \log(\pi(\boldsymbol{f}^{-1}(\boldsymbol{q})))
            + \log(|det(\mathbf{J}(\boldsymbol{f}^{-1}(\boldsymbol{q})))|).

    For the first order sensitivity, the transformation is done using

    .. math::
        \frac{\partial \log(\pi(\boldsymbol{q}))}{\partial q_i} =
            \frac{\partial \log(|det(\mathbf{J})|)}{\partial q_i}
            + \frac{\partial
                \log(\pi(\boldsymbol{f}^{-1}(\boldsymbol{q})))}{\partial q_i}.

    The second term can be calculated using the calculated using the chain rule

    .. math::
        \frac{\partial
            \log(\pi(\boldsymbol{f}^{-1}(\boldsymbol{q})))}{\partial q_i} =
            \sum_l \frac{\partial \log(\pi(\boldsymbol{p}))}{\partial p_l}
            \frac{\partial p_l}{\partial q_i}.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    log_pdf
        A :class:`pints.LogPDF`.
    transform
        A :class:`pints.Transformation`.
    """
    def __init__(self, log_pdf, transform):
        self._log_pdf = log_pdf
        self._transform = transform
        self._n_parameters = self._log_pdf.n_parameters()
        if self._transform.n_parameters() != self._n_parameters:
            raise ValueError('Number of parameters for log_pdf and transform '
                             'must match.')

    def __call__(self, q):
        # Get parameters in the model space
        p = self._transform.to_model(q)

        # Compute LogPDF in the model space
        logpdf_nojac = self._log_pdf(p)

        # Calculate the PDF using change of variable
        # Wikipedia: https://w.wiki/UsJ
        log_jacobian_det = self._transform.log_jacobian_det(q)
        return logpdf_nojac + log_jacobian_det

    def evaluateS1(self, q):
        """ See :meth:`LogPDF.evaluateS1()`. """
        # Get parameters in the model space
        p = self._transform.to_model(q)

        # Compute evaluateS1 of LogPDF in the model space
        logpdf_nojac, dlogpdf_nojac = self._log_pdf.evaluateS1(p)

        # Compute log Jacobian and its derivatives
        logjacdet, dlogjacdet = self._transform.log_jacobian_det_S1(q)

        # Calculate the PDF change of variable, see self.__call__()
        logpdf = logpdf_nojac + logjacdet

        # Calculate the PDF S1 using change of variable
        # Wikipedia: https://w.wiki/Us8
        #
        # This can be done in matrix form, for Jacobian matrix J and
        # E = log(pi(p)):
        #
        # (\nabla_q E)^T = J_(E.g) = J_(E(p)) J_(g) = (\nabla_p E)^T J_(g)
        # (\nabla denotes the del operator)
        #
        jacobian = self._transform.jacobian(q)
        dlogpdf = np.matmul(dlogpdf_nojac, jacobian)  # Jacobian must be 2nd
        dlogpdf += pints.vector(dlogjacdet)

        return logpdf, dlogpdf

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters


class TransformedLogPrior(TransformedLogPDF, pints.LogPrior):
    """
    A :class:`pints.LogPrior` that accepts parameters in a transformed search
    space.

    Extends :class:`pints.LogPrior`, :class:`pints.TransformedLogPDF`.

    Parameters
    ----------
    log_prior
        A :class:`pints.LogPrior`.
    transform
        A :class:`pints.Transformation`.
    """
    def __init__(self, log_prior, transform):
        super(TransformedLogPrior, self).__init__(log_prior, transform)

    def sample(self, n):
        """
        See :meth:`pints.LogPrior.sample()`.

        *Note that this does not sample from the transformed log-prior but
        simply transform the samples from the original log-prior.*
        """
        ps = self._log_pdf.sample(n)
        qs = np.zeros(ps.shape)
        for i, p in enumerate(ps):
            qs[i, :] = self._transform.to_search(p)
        return qs


class ComposedTransformation(Transformation):
    r"""
    N-dimensional :class:`Transformation` composed of one or more other
    :math:`N_i`-dimensional ``Transformation``, such that
    :math:`\sum _i N_i = N`. The evaluation and transformation of the composed
    transformations assume the input transformations are all independent from
    each other.

    For example, a composed transform::

        t = pints.ComposedTransformation(transform1, transform2, transform3)

    where ``transform1``, ``transform2``, and ``transform3`` each have
    dimension 1, 2 and 1, will have dimension 4.

    The dimensionality of the individual priors does not have to be the same,
    i.e. :math:`N_i\neq N_j` is allowed.

    The input parameters of the :class:`ComposedTransformation` have to be
    ordered in the same way as the individual tranforms for the parameter
    vector. In the above example the transform may be performed by
    ``t.to_search(p)``, where::

        p = [parameter1_transform1,
             parameter1_transform2,
             parameter2_transform2,
             parameter1_transform3]

    Extends :class:`Transformation`.
    """
    def __init__(self, *transforms):
        # Check if sub-transforms given
        if len(transforms) < 1:
            raise ValueError('Must have at least one sub-transform.')

        # Check if proper transform, count dimension
        self._n_parameters = 0
        for transform in transforms:
            if not isinstance(transform, pints.Transformation):
                raise ValueError('All sub-transforms must extend '
                                 'pints.Transformation.')
            self._n_parameters += transform.n_parameters()

        # Store
        self._transforms = transforms

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        q = pints.vector(q)
        lo, hi = 0, self._transforms[0].n_parameters()
        output = self._transforms[0].jacobian(q[lo:hi])
        for transform in self._transforms[1:]:
            lo = hi
            hi += transform.n_parameters()
            jaco = transform.jacobian(q[lo:hi])
            pack = np.zeros((output.shape[0], jaco.shape[1]))
            output = np.block([[output, pack], [pack.T, jaco]])
        return output

    def log_jacobian_det(self, q):
        """ See :meth:`Transformation.log_jacobian_det()`. """
        q = pints.vector(q)
        output = 0
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            output += transform.log_jacobian_det(q[lo:hi])
        return output

    def log_jacobian_det_S1(self, q):
        """ See :meth:`Transformation.log_jacobian_det_S1()`. """
        q = pints.vector(q)
        output = 0
        output_s1 = np.zeros(q.shape)
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            j, js1 = transform.log_jacobian_det_S1(q[lo:hi])
            output += j
            output_s1[lo:hi] = np.asarray(js1)
        return output, output_s1

    def multiple_to_model(self, qs):
        """ See :meth:`Transformation.multiple_to_model()`. """
        qs = np.asarray(qs)
        output = np.zeros(qs.shape)
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            output[:, lo:hi] = np.asarray(
                transform.multiple_to_model(qs[:, lo:hi]))
        return output

    def to_model(self, q):
        """ See :meth:`Transformation.to_model()`. """
        q = pints.vector(q)
        output = np.zeros(q.shape)
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            output[lo:hi] = np.asarray(transform.to_model(q[lo:hi]))
        return output

    def to_search(self, p):
        """ See :meth:`Transformation.to_search()`. """
        p = pints.vector(p)
        output = np.zeros(p.shape)
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            output[lo:hi] = np.asarray(transform.to_search(p[lo:hi]))
        return output

    def n_parameters(self):
        """ See :meth:`Transformation.n_parameters()`. """
        return self._n_parameters


class IdentityTransformation(Transformation):
    """
    Identity transformation does nothing to the input parameters, i.e. the
    search space under this transformation is the same as the model space.
    And its Jacobian matrix is the identity matrix.

    Extends :class:`Transformation`.

    Parameters
    ----------
    n_parameters
        Number of model parameters this transformation is defined over.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        return np.eye(self._n_parameters)

    def log_jacobian_det(self, q):
        """ See :meth:`Transformation.log_jacobian_det()`. """
        return 0.

    def log_jacobian_det_S1(self, q):
        """ See :meth:`Transformation.log_jacobian_det_S1()`. """
        return self.log_jacobian_det(q), np.zeros(self._n_parameters)

    def n_parameters(self):
        """ See :meth:`Transformation.n_parameters()`. """
        return self._n_parameters

    def multiple_to_model(self, qs):
        """ See :meth:`Transformation.multiple_to_model()`. """
        return np.asarray(qs)

    def to_model(self, q):
        """ See :meth:`Transformation.to_model()`. """
        return pints.vector(q)

    def to_search(self, p):
        """ See :meth:`Transformation.to_search()`. """
        return pints.vector(p)


class LogitTransformation(Transformation):
    r"""
    Logit (or log-odds) transformation of the model parameters:

    .. math::
        q = \text{logit}(p) = \log(\frac{p}{1 - p}),

    where :math:`p` is the model parameter vector and :math:`q` is the
    search space vector.

    The Jacobian adjustment of the logit transformation is given by

    .. math::
        |\frac{d}{dq} \text{logit}^{-1}(q)| = \text{logit}^{-1}(q) \times
        (1 - \text{logit}^{-1}(q)).

    The first order derivative of the log determinant of the Jacobian is

    .. math::
        \frac{d}{dq} \log(|J(q)|) = 2 \times \exp(-q) \times
                                    \text{logit}^{-1}(q) - 1.

    Extends :class:`Transformation`.

    Parameters
    ----------
    n_parameters
        Number of model parameters this transformation is defined over.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        q = pints.vector(q)
        return np.diag(expit(q) * (1. - expit(q)))

    def log_jacobian_det(self, q):
        """ See :meth:`Transformation.log_jacobian_det()`. """
        q = pints.vector(q)
        return np.sum(np.log(expit(q)) + np.log(1. - expit(q)))

    def log_jacobian_det_S1(self, q):
        """ See :meth:`Transformation.log_jacobian_det_S1()`. """
        q = pints.vector(q)
        logjacdet = self.log_jacobian_det(q)
        dlogjacdet = 2. * np.exp(-q) * expit(q) - 1.
        return logjacdet, dlogjacdet

    def n_parameters(self):
        """ See :meth:`Transformation.n_parameters()`. """
        return self._n_parameters

    def multiple_to_model(self, qs):
        """ See :meth:`Transformation.multiple_to_model()`. """
        qs = np.asarray(qs)
        return expit(qs)

    def to_model(self, q):
        """ See :meth:`Transformation.to_model()`. """
        q = pints.vector(q)
        return expit(q)

    def to_search(self, p):
        """ See :meth:`Transformation.to_search()`. """
        p = pints.vector(p)
        return logit(p)


class LogTransformation(Transformation):
    r"""
    Logarithm transformation of the model parameters:

    .. math::
        q = \log(p),

    where :math:`p` is the model parameter vector and :math:`q` is the
    search space vector.

    The Jacobian adjustment of the log transformation is given by

    .. math::
        |\frac{d}{dq} \exp(q)| = \exp(q).

    The first order derivative of the log determinant of the Jacobian is

    .. math::
        \frac{d}{dq} \log(|J(q)|) = 1.

    Extends :class:`Transformation`.

    Parameters
    ----------
    n_parameters
        Number of model parameters this transformation is defined over.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        q = pints.vector(q)
        return np.diag(np.exp(q))

    def log_jacobian_det(self, q):
        """ See :meth:`Transformation.log_jacobian_det()`. """
        q = pints.vector(q)
        return np.sum(q)

    def log_jacobian_det_S1(self, q):
        """ See :meth:`Transformation.log_jacobian_det_S1()`. """
        q = pints.vector(q)
        logjacdet = self.log_jacobian_det(q)
        dlogjacdet = np.ones(self._n_parameters)
        return logjacdet, dlogjacdet

    def n_parameters(self):
        """ See :meth:`Transformation.n_parameters()`. """
        return self._n_parameters

    def multiple_to_model(self, qs):
        """ See :meth:`Transformation.multiple_to_model()`. """
        qs = np.asarray(qs)
        return np.exp(qs)

    def to_model(self, q):
        """ See :meth:`Transformation.to_model()`. """
        q = pints.vector(q)
        return np.exp(q)

    def to_search(self, p):
        """ See :meth:`Transformation.to_search()`. """
        p = pints.vector(p)
        return np.log(p)


class RectangularBoundariesTransformation(Transformation):
    r"""
    A generalised version of logit transformation for the model parameters,
    which transform an interval or ractangular boundaries :math:`[a, b)` to
    all real number:

    .. math::
        q = f(p) = \log(p - a) - \log(b - p),

    where :math:`p` is the model parameter vector and :math:`q` is the
    search space vector. The range includes the lower (:math:`a`), but not the
    upper (:math:`b`) boundaries. Note that :class:`LogitTransformation` is a
    special case where :math:`a = 0` and :math:`b = 1`.

    The Jacobian adjustment of the transformation is given by

    .. math::
        |\frac{d}{dq} f^{-1}(q)| = \frac{b - a}{\exp(q) (1 + \exp(-q)) ^ 2}.

    The log-determinant of the Jacobian matrix is given by

    .. math::
        \log|\frac{d}{dq} f^{-1}(q)| = \sum_i \left( \log(b_i - a_i) -
            2 \times \log(1 + \exp(-q_i)) - q_i \right)

    The first order derivative of the log determinant of the Jacobian is

    .. math::
        \frac{d}{dq} \log(|J(q)|) = 2 \times \exp(-q) \times
            \text{logit}^{-1}(q) - 1.

    For example, to create a transform with :math:`p_1 \in [0, 4)`,
    :math:`p_2 \in [1, 5)`, and :math:`p_3 \in [2, 6)` use either::

        transform = pints.RectangularBoundariesTransformation([0, 1, 2],
                                                              [4, 5, 6])

    or::

        boundaries = pints.RectangularBoundaries([0, 1, 2], [4, 5, 6])
        transform = pints.RectangularBoundariesTransformation(boundaries)

    Extends :class:`Transformation`.
    """
    def __init__(self, lower_or_boundaries, upper=None):
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries,
                              pints.RectangularBoundaries):
                raise ValueError(
                    'RectangularBoundariesTransformation requires a lower and '
                    'an upper bound, or a single RectangularBoundaries object.'
                )
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

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        q = pints.vector(q)
        diag = (self._b - self._a) / (np.exp(q) * (1. + np.exp(-q)) ** 2)
        return np.diag(diag)

    def log_jacobian_det(self, q):
        """ See :meth:`Transformation.log_jacobian_det()`. """
        q = pints.vector(q)
        s = self._softplus(-q)
        return np.sum(np.log(self._b - self._a) - 2. * s - q)

    def log_jacobian_det_S1(self, q):
        """ See :meth:`Transformation.log_jacobian_det_S1()`. """
        q = pints.vector(q)
        logjacdet = self.log_jacobian_det(q)
        dlogjacdet = 2. * np.exp(-q) * expit(q) - 1.
        return logjacdet, dlogjacdet

    def n_parameters(self):
        """ See :meth:`Transformation.n_parameters()`. """
        return self._n_parameters

    def multiple_to_model(self, qs):
        """ See :meth:`Transformation.multiple_to_model()`. """
        qs = np.asarray(qs)
        return (self._b - self._a) * expit(qs) + self._a

    def to_model(self, q):
        """ See :meth:`Transformation.to_model()`. """
        q = pints.vector(q)
        return (self._b - self._a) * expit(q) + self._a

    def to_search(self, p):
        p = pints.vector(p)
        """ See :meth:`Transformation.to_search()`. """
        return np.log(p - self._a) - np.log(self._b - p)

    def _softplus(self, q):
        """ Returns the softplus function. """
        return np.log(1. + np.exp(q))
