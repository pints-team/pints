#
# Classes that automate parameter transformation
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
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

    References
    ----------
    .. [1] How to Obtain Those Nasty Standard Errors From Transformed Data.
           Erik Jorgensen and Asger Roer Pedersen.
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.47.9023

    .. [2] The Matrix Cookbook.
           Kaare Brandt Petersen and Michael Syskind Pedersen.
           2012.
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
        if isinstance(boundaries, pints.RectangularBoundaries):
            if self.elementwise():
                return TransformedRectangularBoundaries(boundaries, self)
        return TransformedBoundaries(boundaries, self)

    def convert_covariance_matrix(self, C, q):
        r"""
        Converts a convariance matrix ``C`` from the model space to the search
        space around a parameter vector ``q`` provided in the search space.

        The transformation is performed using a first order linear
        approximation [1]_ with the Jacobian :math:`\mathbf{J}`:

        .. math::

            \mathbf{C}(\boldsymbol{q}) &=
                \frac{d\boldsymbol{g}(\boldsymbol{p})}{d\boldsymbol{p}}
                \mathbf{C}(\boldsymbol{p})
                \left(
                    \frac{d\boldsymbol{g}(\boldsymbol{p})}{d\boldsymbol{p}}
                \right)^T + \mathcal{O}(\mathbf{C}(\boldsymbol{p})^2) \\
                &= \mathbf{J}^{-1}(\boldsymbol{q})
                \mathbf{C}(\boldsymbol{p})
                (\mathbf{J}^{-1}(\boldsymbol{q}))^T
                + \mathcal{O}(\mathbf{C}(\boldsymbol{p})^2).

        Using the property that
        :math:`\mathbf{J}^{-1} = \frac{d\boldsymbol{g}}{d\boldsymbol{p}}`,
        from the inverse function theorem, i.e. the matrix inverse of the
        Jacobian matrix of an invertible function is the Jacobian matrix of the
        inverse function.
        """
        jac_inv = np.linalg.pinv(self.jacobian(q))
        return np.matmul(np.matmul(jac_inv, C), jac_inv.T)

    def convert_standard_deviation(self, s, q):
        r"""
        Converts standard deviation ``s``, either a scalar or a vector, from
        the model space to the search space around a parameter vector ``q``
        provided in the search space.

        The transformation is performed using a first order linear
        approximation [1]_ with the Jacobian :math:`\mathbf{J}`:

        .. math::

            \mathbf{C}(\boldsymbol{q}) &=
                \frac{d\boldsymbol{g}(\boldsymbol{p})}{d\boldsymbol{p}}
                \mathbf{C}(\boldsymbol{p})
                \left(
                    \frac{d\boldsymbol{g}(\boldsymbol{p})}{d\boldsymbol{p}}
                \right)^T + \mathcal{O}(\mathbf{C}(\boldsymbol{p})^2) \\
                &= \mathbf{J}^{-1}(\boldsymbol{q})
                \mathbf{C}(\boldsymbol{p})
                (\mathbf{J}^{-1}(\boldsymbol{q}))^T
                + \mathcal{O}(\mathbf{C}(\boldsymbol{p})^2).

        Using the property that
        :math:`\mathbf{J}^{-1} = \frac{d\boldsymbol{g}}{d\boldsymbol{p}}`,
        from the inverse function theorem, i.e. the matrix inverse of the
        Jacobian matrix of an invertible function is the Jacobian matrix of the
        inverse function.

        To transform the provided standard deviation :math:`\boldsymbol{s}`, we
        assume the covariance matrix :math:`\mathbf{C}(\boldsymbol{p})` above
        is a diagonal matrix with :math:`\boldsymbol{s}^2` on the diagonal,
        such that

        .. math::
            s_i(\boldsymbol{q}) =
                \left(
                    \mathbf{J}^{-1} (\mathbf{J}^{-1})^T
                \right)^{1/2}_{i, i}
                s_i(\boldsymbol{p}).
        """
        jac_inv = np.linalg.pinv(self.jacobian(q))
        return s * np.sqrt(np.diagonal(np.matmul(jac_inv, jac_inv.T)))

    def jacobian(self, q):
        r"""
        Returns the Jacobian matrix of the transformation calculated at the
        parameter vector ``q`` in the search space. For a transformation
        :math:`\boldsymbol{q} = \boldsymbol{f}(\boldsymbol{p})`, the Jacobian
        matrix is defined as

        .. math::
            \mathbf{J} =
                \left[\frac{\partial \boldsymbol{f}^{-1}}{\partial q_1} \quad
                 \frac{\partial \boldsymbol{f}^{-1}}{\partial q_2} \quad
                 \cdots \right].

        *This is an optional method.* It is needed when transformation of
        standard deviation :meth:`Transformation.convert_standard_deviation` or
        covariance matrix :meth:`Transformation.convert_covariance_matrix` is
        needed, or when ``evaluateS1()`` is needed.
        """
        raise NotImplementedError

    def jacobian_S1(self, q):
        r"""
        Computes the Jacobian matrix of the transformation calculated at the
        parameter vector ``q`` in the search space, and returns the result
        along with the partial derivatives of the result with respect to the
        parameters.

        The returned data is a tuple ``(S, S')`` where ``S`` is a
        ``n_parameters`` by ``n_parameters`` matrix and ``S'`` is a sequence of
        ``n_parameters`` matrices.

        *This is an optional method.* It is needed when the transformation is
        used along with a non-element-wise transformation in
        :class:`ComposedTransformation`.
        """
        raise NotImplementedError

    def log_jacobian_det(self, q):
        """
        Returns the logarithm of the absolute value of the determinant of the
        Jacobian matrix of the transformation :meth:`Transformation.jacobian`
        calculated at the parameter vector ``q`` in the search space.

        The default implementation numerically calculates the determinant of
        the full matrix which only works if the optional method
        :meth:`Transformation.jacobian` is implemented. If there is an analytic
        expression for the specific transformation, a reimplementation of this
        method may be preferred.

        *This is an optional method.* It is needed when transformation is
        performed on :class:`LogPDF` and/or that requires ``evaluateS1()``;
        e.g. not necessary if it's used for :class:`ErrorMeasure` without
        :meth:`ErrorMeasure.evaluateS1()`.
        """
        return np.log(np.abs(np.linalg.det(self.jacobian(q))))

    def log_jacobian_det_S1(self, q):
        r"""
        Computes the logarithm of the absolute value of the determinant of the
        Jacobian, and returns the result plus the partial derivatives of the
        result with respect to the parameters.

        The returned data is a tuple ``(S, S')`` where ``S`` is a scalar value
        and ``S'`` is a sequence of length ``n_parameters``.

        Note that the derivative returned is of the log of the determinant of
        the Jacobian, so ``S' = d/dq log(|det(J(q))|)``, evaluated at input.

        The absolute value of the determinant of the Jacobian is provided by
        :meth:`Transformation.log_jacobian_det`. The default implementation
        calculates the derivatives of the log-determinant using [2]_

        .. math::
            \frac{d}{dq} \log(|det(\mathbf{J})|) =
                trace(\mathbf{J}^{-1} \frac{d}{dq} \mathbf{J}),

        where the derivative of the Jacobian matrix is provided by
        :meth:`Transformation.jacobian_S1` and the matrix inversion is
        numerically calculated. If there is an analytic expression for the
        specific transformation, a reimplementation of this method may be
        preferred.

        *This is an optional method.* It is needed when transformation is
        performed on :class:`LogPDF` and that requires ``evaluateS1()``.
        """
        # Directly calculate the derivative using jacobian_S1(), we have
        #
        # d/dq log(|det(J(q))|) = 1/|det(J(q))| * sign(det(J(q)))
        #                         * d/dq det(J(q))
        #
        # The last term is given by Eq. 46 in the Matrix Cookbook (2012)
        # http://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
        #
        # d/dq det(J(q)) = det(J) Tr(J^{-1} d/dq J)
        #
        # Therefore
        #
        # d/dq log(|det(J(q))|) = Tr(J^{-1} d/dq J)
        #
        q = pints.vector(q)
        jac, jac_S1 = self.jacobian_S1(q)
        out_S1 = np.zeros(q.shape)
        for i, jac_S1_i in enumerate(jac_S1):
            out_S1[i] = np.trace(np.matmul(np.linalg.pinv(jac), jac_S1_i))
        return self.log_jacobian_det(q), out_S1

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

    def elementwise(self):
        r"""
        Returns True if the transformation is element-wise.

        An element-wise transformation is a transformation
        :math:`\boldsymbol{f}` that can be carried out element by element:
        for a parameter vector :math:`\boldsymbol{p}` in the model space and a
        parameter vector :math:`\boldsymbol{q}` in the search space, then it
        has

        .. math::
            q_i = f(p_i),

        where :math:`x_i` denotes the :math:`i^{\text{th}}` element of the
        vector :math:`\boldsymbol{x}`, as opposed to a transformation in which
        multiple elements are combined to create the transformed elements.
        """
        raise NotImplementedError


class ComposedTransformation(Transformation):
    r"""
    N-dimensional :class:`Transformation` composed of one or more other
    :math:`N_i`-dimensional sub-transformations, so that
    :math:`\sum _i N_i = N`.

    The dimensionality of the individual transformations does not have to be
    the same, i.e. :math:`N_i\neq N_j` is allowed.

    For example, a composed transformation::

        t = pints.ComposedTransformation(
            transformation_1, transformation_2, transformation_3)

    where ``transformation_1``, ``transformation_2``, and ``transformation_3``
    have dimension 1, 2 and 1 respectively, will have dimension N=4.

    The evaluation and transformation of the composed transformations assume
    that the input transformations are all independent from each other.

    The input parameters of the :class:`ComposedTransformation` are ordered in
    the same way as the individual tranforms for the parameter vector. In the
    above example the transformation may be performed by ``t.to_search(p)``,
    where::

        p = [parameter_1_for_transformation_1,
             parameter_1_for_transformation_2,
             parameter_2_for_transformation_2,
             parameter_1_for_transformation_3]

    Extends :class:`Transformation`.
    """
    def __init__(self, *transformations):
        # Check if sub-transformations given
        if len(transformations) < 1:
            raise ValueError('Must have at least one sub-transformation.')

        # Check if proper Transformation, count dimension
        self._n_parameters = 0
        self._elementwise = True
        for transformation in transformations:
            if not isinstance(transformation, pints.Transformation):
                raise ValueError('All entries in `transformations` must extend'
                                 ' pints.Transformation.')
            self._n_parameters += transformation.n_parameters()
            # If any transformation is not elementwise, then it is not
            # elementwise
            self._elementwise &= transformation.elementwise()

        # Store
        self._transformations = transformations

        # Use elementwise or not
        if self._elementwise:
            self._jacobian = self._elementwise_jacobian
            self._log_jacobian_det = self._elementwise_log_jacobian_det
            self._log_jacobian_det_S1 = self._elementwise_log_jacobian_det_S1
        else:
            self._jacobian = self._general_jacobian
            self._log_jacobian_det = self._general_log_jacobian_det
            self._log_jacobian_det_S1 = self._general_log_jacobian_det_S1

    def elementwise(self):
        """ See :meth:`Transformation.elementwise()`. """
        return self._elementwise

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        return self._jacobian(q)

    def jacobian_S1(self, q):
        """ See :meth:`Transformation.jacobian_S1()`. """
        q = pints.vector(q)
        matrix_shape = (self.n_parameters(), self.n_parameters())
        lo = hi = 0
        output_S1 = np.zeros((self.n_parameters(),) + matrix_shape)
        for transformation in self._transformations:
            lo = hi
            hi += transformation.n_parameters()
            _, jac_S1 = transformation.jacobian_S1(q[lo:hi])
            for i, jac_S1_i in enumerate(jac_S1):
                # Due to the composed transformation are independent, we can
                # pack the derivative of the Jacobian matrices J_i with zeros:
                #
                #     [ 0   0   0 ]
                # J = [ 0  J_i  0 ]
                #     [ 0   0   0 ]
                #
                o = np.zeros(matrix_shape)
                o[lo:hi, lo:hi] = jac_S1_i[:, :]
                output_S1[lo + i, :, :] = o
        return self.jacobian(q), output_S1

    def log_jacobian_det(self, q):
        """ See :meth:`Transformation.log_jacobian_det()`. """
        return self._log_jacobian_det(q)

    def log_jacobian_det_S1(self, q):
        """ See :meth:`Transformation.log_jacobian_det_S1()`. """
        return self._log_jacobian_det_S1(q)

    def to_model(self, q):
        """ See :meth:`Transformation.to_model()`. """
        q = pints.vector(q)
        output = np.zeros(q.shape)
        lo = hi = 0
        for transformation in self._transformations:
            lo = hi
            hi += transformation.n_parameters()
            output[lo:hi] = np.asarray(transformation.to_model(q[lo:hi]))
        return output

    def to_search(self, p):
        """ See :meth:`Transformation.to_search()`. """
        p = pints.vector(p)
        output = np.zeros(p.shape)
        lo = hi = 0
        for transformation in self._transformations:
            lo = hi
            hi += transformation.n_parameters()
            output[lo:hi] = np.asarray(transformation.to_search(p[lo:hi]))
        return output

    def n_parameters(self):
        """ See :meth:`Transformation.n_parameters()`. """
        return self._n_parameters

    def _elementwise_jacobian(self, q):
        """
        Element-wise implementation of :meth:`Transformation.jacobian()`.
        """
        q = pints.vector(q)
        diag = np.zeros(q.shape)
        lo = hi = 0
        for transformation in self._transformations:
            lo = hi
            hi += transformation.n_parameters()
            diag[lo:hi] = np.diagonal(transformation.jacobian(q[lo:hi]))
        output = np.diag(diag)
        return output

    def _elementwise_log_jacobian_det(self, q):
        """
        Element-wise implementation of
        :meth:`Transformation.log_jacobian_det()`.
        """
        q = pints.vector(q)
        output = 0
        lo = hi = 0
        for transformation in self._transformations:
            lo = hi
            hi += transformation.n_parameters()
            output += transformation.log_jacobian_det(q[lo:hi])
        return output

    def _elementwise_log_jacobian_det_S1(self, q):
        """
        Element-wise implementation of
        :meth:`Transformation.log_jacobian_det_S1()`.
        """
        q = pints.vector(q)
        output = 0
        output_S1 = np.zeros(q.shape)
        lo = hi = 0
        for transformation in self._transformations:
            lo = hi
            hi += transformation.n_parameters()
            j, j_S1 = transformation.log_jacobian_det_S1(q[lo:hi])
            output += j
            output_S1[lo:hi] = np.asarray(j_S1)
        return output, output_S1

    def _general_jacobian(self, q):
        """
        General implementation of :meth:`Transformation.jacobian()`.
        """
        q = pints.vector(q)
        lo, hi = 0, self._transformations[0].n_parameters()
        output = self._transformations[0].jacobian(q[lo:hi])
        for transformation in self._transformations[1:]:
            lo = hi
            hi += transformation.n_parameters()
            jaco = transformation.jacobian(q[lo:hi])
            # Due to the composed transformation are independent, we can stack
            # the Jacobian matrices J_i and J_{i+1} as
            #
            # J = [ J_i     0    ]
            #     [  0   J_{i+1} ]
            #
            pack = np.zeros((output.shape[0], jaco.shape[1]))
            output = np.block([[output, pack], [pack.T, jaco]])
        return output

    def _general_log_jacobian_det(self, q):
        """
        General implementation of :meth:`Transformation.log_jacobian_det()`.
        """
        # The same as in Transformation.log_jacobian_det()
        return np.log(np.abs(np.linalg.det(self.jacobian(q))))

    def _general_log_jacobian_det_S1(self, q):
        """
        General implementation of :meth:`Transformation.log_jacobian_det_S1()`.
        """
        # The same as in Transformation.log_jacobian_det_S1()
        q = pints.vector(q)
        jac, jac_S1 = self.jacobian_S1(q)
        out_S1 = np.zeros(q.shape)
        for i, jac_S1_i in enumerate(jac_S1):
            out_S1[i] = np.trace(np.matmul(np.linalg.pinv(jac), jac_S1_i))
        return self.log_jacobian_det(q), out_S1


class IdentityTransformation(Transformation):
    """
    :class`Transformation` that returns the input (untransformed) parameters,
    i.e. the search space under this transformation is the same as the model
    space. And its Jacobian matrix is the identity matrix.

    Extends :class:`Transformation`.

    Parameters
    ----------
    n_parameters
        Number of model parameters this transformation is defined over.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def elementwise(self):
        """ See :meth:`Transformation.elementwise()`. """
        return True

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        return np.eye(self._n_parameters)

    def jacobian_S1(self, q):
        """ See :meth:`Transformation.jacobian_S1()`. """
        n = self._n_parameters
        return self.jacobian(q), np.zeros((n, n, n))

    def log_jacobian_det(self, q):
        """ See :meth:`Transformation.log_jacobian_det()`. """
        return 0.

    def log_jacobian_det_S1(self, q):
        """ See :meth:`Transformation.log_jacobian_det_S1()`. """
        return self.log_jacobian_det(q), np.zeros(self._n_parameters)

    def n_parameters(self):
        """ See :meth:`Transformation.n_parameters()`. """
        return self._n_parameters

    def to_model(self, q):
        """ See :meth:`Transformation.to_model()`. """
        return pints.vector(q)

    def to_search(self, p):
        """ See :meth:`Transformation.to_search()`. """
        return pints.vector(p)


class LogitTransformation(Transformation):
    r"""
    Logit (or log-odds) transformation of the model parameters.

    The transformation is given by

    .. math::
        q = \text{logit}(p) = \log(\frac{p}{1 - p}),

    where :math:`p` is the model parameter vector and :math:`q` is the
    search space vector.

    The Jacobian adjustment of the logit transformation is given by

    .. math::
        |\frac{d}{dq} \text{logit}^{-1}(q)| = \text{logit}^{-1}(q) \times
        (1 - \text{logit}^{-1}(q)).

    And its derivative is given by

    .. math::
        \frac{d^2}{dq^2} \text{logit}^{-1}(q) = \frac{d f^{-1}(q)}{dq} \times
            \left( \frac{\exp(-q) - 1}{exp(-q) + 1} \right).

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

    def elementwise(self):
        """ See :meth:`Transformation.elementwise()`. """
        return True

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        q = pints.vector(q)
        return np.diag(expit(q) * (1. - expit(q)))

    def jacobian_S1(self, q):
        """ See :meth:`Transformation.jacobian_S1()`. """
        q = pints.vector(q)
        n = self._n_parameters
        jac = self.jacobian(q)
        jac_S1 = np.zeros((n, n, n))
        rn = np.arange(n)
        jac_S1[rn, rn, rn] = np.diagonal(jac) * (np.exp(-q) - 1.) * expit(q)
        return jac, jac_S1

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

    The transformation is given by

    .. math::
        q = \log(p),

    where :math:`p` is the model parameter vector and :math:`q` is the
    search space vector.

    The Jacobian adjustment of the log transformation is given by

    .. math::
        |\frac{d}{dq} \exp(q)| = \exp(q).

    And its derivative is given by

    .. math::
        \frac{d^2}{dq^2} \exp(q) = \exp(q).

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

    def elementwise(self):
        """ See :meth:`Transformation.elementwise()`. """
        return True

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        q = pints.vector(q)
        return np.diag(np.exp(q))

    def jacobian_S1(self, q):
        """ See :meth:`Transformation.jacobian_S1()`. """
        q = pints.vector(q)
        n = self._n_parameters
        jac = self.jacobian(q)
        jac_S1 = np.zeros((n, n, n))
        rn = np.arange(n)
        jac_S1[rn, rn, rn] = np.diagonal(jac)
        return jac, jac_S1

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
    A generalised version of the logit transformation for the model parameters,
    which transforms an interval or rectangular boundaries :math:`[a, b)` to
    all real number.

    The transformation is given by

    .. math::
        q = f(p) = \text{logit}\left(\frac{p - a}{b - p}\right)
                 = \log(p - a) - \log(b - p),

    where :math:`p` is the model parameter vector and :math:`q` is the
    search space vector. Note that :class:`LogitTransformation` is a special
    case where :math:`a = 0` and :math:`b = 1`.

    The Jacobian adjustment of the transformation is given by

    .. math::
        |\frac{d}{dq} f^{-1}(q)| = \frac{b - a}{\exp(q) (1 + \exp(-q)) ^ 2}.

    And its derivative is given by

    .. math::
        \frac{d^2}{dq^2} f^{-1}(q) = \frac{d f^{-1}(q)}{dq} \times
            \left( \frac{\exp(-q) - 1}{exp(-q) + 1} \right).

    The log-determinant of the Jacobian matrix is given by

    .. math::
        \log|\frac{d}{dq} f^{-1}(q)| = \sum_i \left( \log(b_i - a_i) -
            2 \times \log(1 + \exp(-q_i)) - q_i \right)

    The first order derivative of the log determinant of the Jacobian is

    .. math::
        \frac{d}{dq} \log(|J(q)|) = 2 \times \exp(-q) \times
            \text{logit}^{-1}(q) - 1.

    For example, to create a transformation with :math:`p_1 \in [0, 4)`,
    :math:`p_2 \in [1, 5)`, and :math:`p_3 \in [2, 6)` use either::

        transformation = pints.RectangularBoundariesTransformation([0, 1, 2],
                                                                   [4, 5, 6])

    or::

        boundaries = pints.RectangularBoundaries([0, 1, 2], [4, 5, 6])
        transformation = pints.RectangularBoundariesTransformation(boundaries)

    Not to be confused with :class:`pints.TransformedRectangularBoundaries`.

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
        del boundaries

    def elementwise(self):
        """ See :meth:`Transformation.elementwise()`. """
        return True

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        q = pints.vector(q)
        diag = (self._b - self._a) / (np.exp(q) * (1. + np.exp(-q)) ** 2)
        return np.diag(diag)

    def jacobian_S1(self, q):
        """ See :meth:`Transformation.jacobian_S1()`. """
        q = pints.vector(q)
        n = self._n_parameters
        jac = self.jacobian(q)
        jac_S1 = np.zeros((n, n, n))
        rn = np.arange(n)
        jac_S1[rn, rn, rn] = np.diagonal(jac) * (np.exp(-q) - 1.) * expit(q)
        return jac, jac_S1

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


class ScalingTransformation(Transformation):
    """
    Scales the input parameters by multiplying with an array ``scalings`` and
    adding an optional array ``translation``.

    The transformation from model parameters ``p`` to search parameters ``q``
    is performed as::

        q = (p + translation) * scalings

    Its Jacobian matrix is a diagonal matrix with ``1 / scalings`` on the
    diagonal.

    Extends :class:`Transformation`.
    """
    def __init__(self, scalings, translation=None):
        self._s = pints.vector(scalings)
        self._inv_s = 1. / self._s
        self._n_parameters = len(self._s)

        self._translation = None
        if translation is not None:
            self._translation = pints.vector(translation)
            if len(self._translation) != self._n_parameters:
                raise ValueError(
                    'Translation must be None or be a vector of the same'
                    ' length as the scalings.')

    def elementwise(self):
        """ See :meth:`Transformation.elementwise()`. """
        return True

    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        return np.diag(self._inv_s)

    def jacobian_S1(self, q):
        """ See :meth:`Transformation.jacobian_S1()`. """
        n = self._n_parameters
        return self.jacobian(q), np.zeros((n, n, n))

    def log_jacobian_det(self, q):
        """ See :meth:`Transformation.log_jacobian_det()`. """
        return np.sum(np.log(np.abs(self._inv_s)))

    def log_jacobian_det_S1(self, q):
        """ See :meth:`Transformation.log_jacobian_det_S1()`. """
        return self.log_jacobian_det(q), np.zeros(self._n_parameters)

    def n_parameters(self):
        """ See :meth:`Transformation.n_parameters()`. """
        return self._n_parameters

    def to_model(self, q):
        """ See :meth:`Transformation.to_model()`. """
        p = self._inv_s * pints.vector(q)
        if self._translation is not None:
            p -= self._translation
        return p

    def to_search(self, p):
        """ See :meth:`Transformation.to_search()`. """
        p = pints.vector(p)
        if self._translation is not None:
            p = p + self._translation
        return self._s * p


class TransformedBoundaries(pints.Boundaries):
    """
    A :class:`pints.Boundaries` that accepts parameters in a transformed
    search space.

    Extends :class:`pints.Boundaries`.

    Parameters
    ----------
    boundaries
        A :class:`pints.Boundaries`.
    transformation
        A :class:`pints.Transformation`.
    """
    def __init__(self, boundaries, transformation):
        self._boundaries = boundaries
        self._transform = transformation
        self._n_parameters = self._boundaries.n_parameters()

        if self._transform.n_parameters() != self._n_parameters:
            raise ValueError('Number of parameters for boundaries and '
                             'transformation must match.')

    def check(self, q):
        """ See :meth:`Boundaries.check()`. """
        # Get parameters in the model space
        p = self._transform.to_model(q)
        # Check Boundaries in the model space
        return self._boundaries.check(p)

    def n_parameters(self):
        """ See :meth:`Boundaries.n_parameters()`. """
        return self._n_parameters

    def sample(self, n=1):
        """ See :meth:`Boundaries.sample()`. """
        return [
            self._transform.to_search(p) for p in self._boundaries.sample(n)]


class TransformedRectangularBoundaries(pints.RectangularBoundaries):
    """
    A :class:`pints.RectangularBoundaries` that accepts parameters in a
    transformed search space.

    This transformation handles the special case where:

    - the boundaries being transformed are :class:`pints.RectangularBoundaries`
    - the transformation is element-wise.

    When these conditions are met, the lower and upper boundaries can simply be
    transformed and methods like :meth:`lower()` and :meth:`range()` can be
    provided.

    Not to be confused with :class:`pints.RectangularBoundariesTransformation`.

    Extends :class:`pints.RectangularBoundaries`.

    Parameters
    ----------
    boundaries
        A :class:`pints.RectangularBoundaries` object.
    transformation
        An element-wise transformation.
    """
    def __init__(self, boundaries, transformation):

        # Check input
        if not isinstance(boundaries, pints.RectangularBoundaries):
            raise ValueError('A TransformedRectangularBoundaries can only be'
                             ' created from a RectangularBoundaries object.')
        if not transformation.elementwise():
            raise ValueError('A TransformedRectangularBoundaries can only be'
                             ' created from an element-wise transformation.')
        if transformation.n_parameters() != boundaries.n_parameters():
            raise ValueError('Number of parameters for boundaries and '
                             'transformation must match.')

        # Transform upper and lower boundaries
        a = transformation.to_search(boundaries.lower())
        b = transformation.to_search(boundaries.upper())
        lower = np.minimum(a, b)
        upper = np.maximum(a, b)

        # Pass transformed boundaries to RectangularBoundaries
        super().__init__(lower, upper)

        # Store input
        self._boundaries = boundaries
        self._transformation = transformation

    def sample(self, n=1):
        """ See :meth:`Boundaries.sample()`. """
        # Sample from the original boundaries, but transform to new space
        return [self._transformation.to_search(p)
                for p in self._boundaries.sample(n)]


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
    transformation
        A :class:`pints.Transformation`.
    """
    def __init__(self, error, transformation):
        self._error = error
        self._transform = transformation
        self._n_parameters = self._error.n_parameters()
        if self._transform.n_parameters() != self._n_parameters:
            raise ValueError('Number of parameters for error and '
                             'transformation must match.')

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
            \left[\frac{\partial \boldsymbol{f}^{-1}}{\partial q_1} \quad
             \frac{\partial \boldsymbol{f}^{-1}}{\partial q_2} \quad
             \cdots \right].

    Hence

    .. math::
        \log(\pi(\boldsymbol{q})) =
            \log(\pi(\boldsymbol{f}^{-1}(\boldsymbol{q})))
            + \log(|det(\mathbf{J}(\boldsymbol{f}^{-1}(\boldsymbol{q})))|).

    For the first order sensitivity, the transformation is done using

    .. math::
        \frac{\partial \log(\pi(\boldsymbol{q}))}{\partial q_i} =
            \frac{\partial
                \log(\pi(\boldsymbol{f}^{-1}(\boldsymbol{q})))}{\partial q_i}
            + \frac{\partial \log(|det(\mathbf{J})|)}{\partial q_i}.

    The first term can be calculated using the chain rule

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
    transformation
        A :class:`pints.Transformation`.
    """
    def __init__(self, log_pdf, transformation):
        self._log_pdf = log_pdf
        self._transform = transformation
        self._n_parameters = self._log_pdf.n_parameters()
        if self._transform.n_parameters() != self._n_parameters:
            raise ValueError('Number of parameters for log_pdf and '
                             'transformation must match.')

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
    transformation
        A :class:`pints.Transformation`.
    """
    def __init__(self, log_prior, transformation):
        super(TransformedLogPrior, self).__init__(log_prior, transformation)

    def sample(self, n):
        """
        See :meth:`pints.LogPrior.sample()`.

        *Note that this does not sample from the transformed log-prior but
        simply transforms the samples from the original log-prior.*
        """
        ps = self._log_pdf.sample(n)
        qs = np.zeros(ps.shape)
        for i, p in enumerate(ps):
            qs[i, :] = self._transform.to_search(p)
        return qs


class UnitCubeTransformation(ScalingTransformation):
    """
    Maps a parameter space onto the unit (hyper)cube.

    Transformations from model parameters ``p`` to search parameters ``q`` are
    made as::

        q = (p - lower) / (upper - lower)

    Extends :class:`ScalingTransformation`.
    """
    def __init__(self, lower, upper):

        # Check input
        self._lower = pints.vector(lower)
        self._upper = pints.vector(upper)
        self._n_parameters = len(lower)
        del lower, upper

        if len(self._upper) != self._n_parameters:
            raise ValueError(
                'Lower and upper bounds must have the same length.')
        if not np.all(self._upper > self._lower):
            raise ValueError('Upper bounds must exceed lower bounds.')

        super().__init__(1 / (self._upper - self._lower), -self._lower)

