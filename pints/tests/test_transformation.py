#!/usr/bin/env python3
#
# Tests Transformation functions in Pints
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import warnings

import numpy as np

import pints
import pints.toy

from shared import CircularBoundaries, SwappingTransformation


class TestTransformation(pints.Transformation):
    """A testing log-transformation class"""
    def jacobian(self, q):
        """ See :meth:`Transformation.jacobian()`. """
        q = pints.vector(q)
        return np.diag(np.exp(q))

    def jacobian_S1(self, q):
        """ See :meth:`Transformation.jacobian_S1()`. """
        q = pints.vector(q)
        n = len(q)
        jac = self.jacobian(q)
        jac_S1 = np.zeros((n, n, n))
        rn = np.arange(n)
        jac_S1[rn, rn, rn] = np.diagonal(jac)
        return jac, jac_S1

    def n_parameters(self):
        return 4

    def to_model(self, q):
        """ See :meth:`Transformation.to_model()`. """
        q = pints.vector(q)
        return np.exp(q)


class TestNonElementWiseIdentityTransformation(pints.IdentityTransformation):
    """A testing non-element-wise transformation class"""
    def elementwise(self):
        return False


class TestAbstractClassTransformation(unittest.TestCase):
    # Test methods defined in the abstract class

    @classmethod
    def setUpClass(cls):
        # Create Transformation class
        cls.t = TestTransformation()

        cls.p = [0.1, 1., 10., 999.]
        cls.x = [-2.3025850929940455, 0., 2.3025850929940459,
                 6.9067547786485539]
        cls.j = np.diag(cls.p)
        cls.j_s1 = np.zeros((4, 4, 4))
        for i in range(4):
            cls.j_s1[i, i, i] = cls.p[i]
        cls.log_j_det = np.sum(cls.x)
        cls.log_j_det_s1 = np.ones(4)

    def test_to_model(self):
        # Test inverse transform
        self.assertTrue(np.allclose(self.t.to_model(self.x), self.p))

    def test_n_parameters(self):
        # Test n_parameters
        self.assertEqual(self.t.n_parameters(), 4)

    def test_jacobian(self):
        # Test Jacobian
        self.assertTrue(np.allclose(self.t.jacobian(self.x), self.j))

    def test_jacobian_S1(self):
        # Test Jacobian derivatives
        calc_mat, calc_deriv = self.t.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))

    def test_log_jacobian_det(self):
        # Test log-Jacobian determinant
        self.assertAlmostEqual(self.t.log_jacobian_det(self.x), self.log_j_det)

    def test_log_jacobian_det_S1(self):
        # Test log-Jacobian determinant derivatives
        calc_val, calc_deriv = self.t.log_jacobian_det_S1(self.x)
        self.assertAlmostEqual(calc_val, self.log_j_det)
        self.assertTrue(np.allclose(calc_deriv, self.log_j_det_s1))

    def test_convert_std_and_cov(self):
        # Test standard deviation and covariance matrix transformations
        sd = np.array([0.01, 0.1, 1., 99.9])
        cov = np.diag(sd ** 2)
        tsd = np.ones(4) * 0.1
        tcov = np.eye(4) * 0.1 ** 2
        self.assertTrue(
            np.allclose(self.t.convert_standard_deviation(sd, self.x), tsd))
        self.assertTrue(
            np.allclose(self.t.convert_covariance_matrix(cov, self.x), tcov))


class TestComposedTransformationElementWise(unittest.TestCase):
    # Test ComposedTransformation class for element-wise case

    @classmethod
    def setUpClass(cls):
        # Create Transformation class
        t1 = pints.IdentityTransformation(1)
        lower2 = np.array([1, 2])
        upper2 = np.array([10, 20])
        t2 = pints.RectangularBoundariesTransformation(lower2, upper2)
        t3 = pints.LogTransformation(1)

        cls.t = pints.ComposedTransformation(t1, t2, t3)

        cls.p = [0.1, 1.5, 15., 999.]
        cls.x = [0.1, -2.8332133440562162, 0.9555114450274365,
                 6.9067547786485539]
        cls.j = np.diag([1., 0.4722222222222225, 3.6111111111111098, 999.])
        cls.j_s1_diag = [0., 0.4197530864197533, -1.6049382716049378, 999.]
        cls.j_s1 = np.zeros((4, 4, 4))
        for i in range(4):
            cls.j_s1[i, i, i] = cls.j_s1_diag[i]
        cls.log_j_det = 7.4404646962481324
        cls.log_j_det_s1 = [0., 0.8888888888888888, -0.4444444444444445, 1.]

    def test_bad_constructor(self):
        # Test invalid constructors
        self.assertRaises(ValueError, pints.ComposedTransformation)
        self.assertRaises(ValueError, pints.ComposedTransformation, np.log)

    def test_to_search(self):
        # Test forward transform
        self.assertTrue(np.allclose(self.t.to_search(self.p), self.x))

    def test_to_model(self):
        # Test inverse transform
        self.assertTrue(np.allclose(self.t.to_model(self.x), self.p))

    def test_n_parameters(self):
        # Test n_parameters
        self.assertEqual(self.t.n_parameters(), 4)

    def test_jacobian(self):
        # Test Jacobian
        self.assertTrue(np.allclose(self.t.jacobian(self.x), self.j))

    def test_jacobian_S1(self):
        # Test Jacobian derivatives
        calc_mat, calc_deriv = self.t.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))

    def test_log_jacobian_det(self):
        # Test log-Jacobian determinant
        self.assertAlmostEqual(self.t.log_jacobian_det(self.x), self.log_j_det)

    def test_log_jacobian_det_S1(self):
        # Test log-Jacobian determinant derivatives
        calc_val, calc_deriv = self.t.log_jacobian_det_S1(self.x)
        self.assertAlmostEqual(calc_val, self.log_j_det)
        self.assertTrue(np.allclose(calc_deriv, self.log_j_det_s1))

    def test_retransform(self):
        # Test forward transform the inverse transform
        self.assertTrue(
            np.allclose(self.p, self.t.to_model(self.t.to_search(self.p))))
        self.assertTrue(
            np.allclose(self.x, self.t.to_search(self.t.to_model(self.x))))

    def test_elementwise(self):
        # Test is elementwise
        self.assertTrue(self.t.elementwise())


class TestComposedTransformation(unittest.TestCase):
    # Test ComposedTransformation class for non-element-wise case

    @classmethod
    def setUpClass(cls):
        # Create Transformation class
        cls.t1 = TestNonElementWiseIdentityTransformation(1)
        lower2 = np.array([1, 2])
        upper2 = np.array([10, 20])
        cls.t2 = pints.RectangularBoundariesTransformation(lower2, upper2)
        cls.t3 = pints.LogTransformation(1)

        cls.t = pints.ComposedTransformation(cls.t1, cls.t2, cls.t3)

        cls.p = [0.1, 1.5, 15., 999.]
        cls.x = [0.1, -2.8332133440562162, 0.9555114450274365,
                 6.9067547786485539]
        cls.j = np.diag([1., 0.4722222222222225, 3.6111111111111098, 999.])
        cls.j_s1_diag = [0., 0.4197530864197533, -1.6049382716049378, 999.]
        cls.j_s1 = np.zeros((4, 4, 4))
        for i in range(4):
            cls.j_s1[i, i, i] = cls.j_s1_diag[i]
        cls.log_j_det = 7.4404646962481324
        cls.log_j_det_s1 = [0., 0.8888888888888888, -0.4444444444444445, 1.]

    def test_bad_constructor(self):
        # Test invalid constructors
        self.assertRaises(ValueError, pints.ComposedTransformation)
        self.assertRaises(ValueError, pints.ComposedTransformation, np.log)

    def test_to_search(self):
        # Test forward transform
        self.assertTrue(np.allclose(self.t.to_search(self.p), self.x))

    def test_to_model(self):
        # Test inverse transform
        self.assertTrue(np.allclose(self.t.to_model(self.x), self.p))

    def test_n_parameters(self):
        # Test n_parameters
        self.assertEqual(self.t.n_parameters(), 4)

    def test_jacobian(self):
        # Test Jacobian
        self.assertTrue(np.allclose(self.t.jacobian(self.x), self.j))

    def test_jacobian_S1(self):
        # Test Jacobian derivatives
        calc_mat, calc_deriv = self.t.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))

    def test_log_jacobian_det(self):
        # Test log-Jacobian determinant
        self.assertEqual(self.t.log_jacobian_det(self.x), self.log_j_det)

    def test_log_jacobian_det_S1(self):
        # Test log-Jacobian determinant derivatives
        calc_val, calc_deriv = self.t.log_jacobian_det_S1(self.x)
        self.assertAlmostEqual(calc_val, self.log_j_det)
        self.assertTrue(np.allclose(calc_deriv, self.log_j_det_s1))

    def test_against_elementwise_transformation(self):
        # Test general case gives the same result as the elementwise case
        t1 = pints.IdentityTransformation(1)  # This is element-wise
        t_elem = pints.ComposedTransformation(t1, self.t2, self.t3)
        self.assertTrue(t_elem.elementwise())  # This is element-wise

        # Test log-Jacobian determinant
        self.assertAlmostEqual(self.t.log_jacobian_det(self.x),
                               t_elem.log_jacobian_det(self.x))

        # Test log-Jacobian determinant derivatives
        _, t_deriv = self.t.log_jacobian_det_S1(self.x)
        _, t_elem_deriv = t_elem.log_jacobian_det_S1(self.x)
        self.assertTrue(np.allclose(t_deriv, t_elem_deriv))

    def test_retransform(self):
        # Test forward transform the inverse transform
        self.assertTrue(
            np.allclose(self.p, self.t.to_model(self.t.to_search(self.p))))
        self.assertTrue(
            np.allclose(self.x, self.t.to_search(self.t.to_model(self.x))))

    def test_elementwise(self):
        # Test is elementwise
        self.assertFalse(self.t.elementwise())


class TestIdentityTransformation(unittest.TestCase):
    # Test IdentityTransformation class

    @classmethod
    def setUpClass(cls):
        # Create Transformation class
        cls.t1 = pints.IdentityTransformation(1)
        cls.t4 = pints.IdentityTransformation(4)

        cls.p = [-177., 0.333, 10., 99.99]
        cls.x = [-177., 0.333, 10., 99.99]
        cls.j = np.eye(4)
        cls.j_s1 = np.zeros((4, 4, 4))
        cls.log_j_det = 0.
        cls.log_j_det_s1 = np.zeros(4)

    def test_to_search(self):
        # Test forward transform
        for xi, pi in zip(self.x, self.p):
            calc_xi = self.t1.to_search(pi)
            self.assertAlmostEqual(calc_xi[0], xi)
        self.assertTrue(np.allclose(self.t4.to_search(self.p), self.x))

    def test_to_model(self):
        # Test inverse transform
        for xi, pi in zip(self.x, self.p):
            calc_pi = self.t1.to_model(xi)
            self.assertAlmostEqual(calc_pi[0], pi)
        self.assertTrue(np.allclose(self.t4.to_model(self.x), self.p))

    def test_n_parameters(self):
        # Test n_parameters
        self.assertEqual(self.t1.n_parameters(), 1)
        self.assertEqual(self.t4.n_parameters(), 4)

    def test_jacobian(self):
        # Test Jacobian
        self.assertTrue(np.allclose(self.t4.jacobian(self.x), self.j))

    def test_jacobian_S1(self):
        # Test Jacobian derivatives
        calc_mat, calc_deriv = self.t4.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))

    def test_log_jacobian_det(self):
        # Test log-Jacobian determinant
        self.assertEqual(self.t4.log_jacobian_det(self.x), self.log_j_det)

    def test_log_jacobian_det_S1(self):
        # Test log-Jacobian determinant derivatives
        calc_val, calc_deriv = self.t4.log_jacobian_det_S1(self.x)
        self.assertEqual(calc_val, self.log_j_det)
        self.assertTrue(np.all(np.equal(calc_deriv, self.log_j_det_s1)))

    def test_retransform(self):
        # Test forward transform the inverse transform
        self.assertTrue(
            np.allclose(self.p, self.t4.to_model(self.t4.to_search(self.p))))
        self.assertTrue(
            np.allclose(self.x, self.t4.to_search(self.t4.to_model(self.x))))

    def test_elementwise(self):
        # Test is elementwise
        self.assertTrue(self.t1.elementwise())
        self.assertTrue(self.t4.elementwise())


class TestLogitTransformation(unittest.TestCase):
    # Test LogitTransformation class

    @classmethod
    def setUpClass(cls):
        # Create Transformation class
        cls.t1 = pints.LogitTransformation(1)
        cls.t4 = pints.LogitTransformation(4)

        cls.p = [0.1, 0.333, 0.5, 0.9]
        cls.x = [-2.1972245773362191, -0.6946475559351799, 0.,
                 2.1972245773362196]
        cls.j = np.diag([0.09, 0.222111, 0.25, 0.09])
        cls.j_s1_diag = [0.072, 0.074185074, 0., -0.072]
        cls.j_s1 = np.zeros((4, 4, 4))
        for i in range(4):
            cls.j_s1[i, i, i] = cls.j_s1_diag[i]
        cls.log_j_det = -7.7067636004918398
        cls.log_j_det_s1 = [0.8, 0.334, 0., -0.8]

    def test_to_search(self):
        # Test forward transform
        for xi, pi in zip(self.x, self.p):
            calc_xi = self.t1.to_search(pi)
            self.assertAlmostEqual(calc_xi[0], xi)
        self.assertTrue(np.allclose(self.t4.to_search(self.p), self.x))

    def test_to_model(self):
        # Test inverse transform
        for xi, pi in zip(self.x, self.p):
            calc_pi = self.t1.to_model(xi)
            self.assertAlmostEqual(calc_pi[0], pi)
        self.assertTrue(np.allclose(self.t4.to_model(self.x), self.p))

    def test_n_parameters(self):
        # Test n_parameters
        self.assertEqual(self.t1.n_parameters(), 1)
        self.assertEqual(self.t4.n_parameters(), 4)

    def test_jacobian(self):
        # Test Jacobian
        self.assertTrue(np.allclose(self.t4.jacobian(self.x), self.j))

    def test_jacobian_S1(self):
        # Test Jacobian derivatives
        calc_mat, calc_deriv = self.t4.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))

    def test_log_jacobian_det(self):
        # Test log-Jacobian determinant
        self.assertAlmostEqual(self.t4.log_jacobian_det(self.x),
                               self.log_j_det)

    def test_log_jacobian_det_S1(self):
        # Test log-Jacobian determinant derivatives
        calc_val, calc_deriv = self.t4.log_jacobian_det_S1(self.x)
        self.assertAlmostEqual(calc_val, self.log_j_det)
        self.assertTrue(np.allclose(calc_deriv, self.log_j_det_s1))

    def test_invalid_inputs(self):
        # Test invalid inputs
        self.assertTrue(np.isnan(self.t1.to_search(-1.)))
        self.assertTrue(np.isnan(self.t1.to_search(2.)))
        self.assertTrue(np.isinf(self.t1.to_search(1.)))

    def test_retransform(self):
        # Test forward transform the inverse transform
        self.assertTrue(
            np.allclose(self.p, self.t4.to_model(self.t4.to_search(self.p))))
        self.assertTrue(
            np.allclose(self.x, self.t4.to_search(self.t4.to_model(self.x))))

    def test_elementwise(self):
        # Test is elementwise
        self.assertTrue(self.t1.elementwise())
        self.assertTrue(self.t4.elementwise())


class TestLogTransformation(unittest.TestCase):
    # Test LogTransformation class

    @classmethod
    def setUpClass(cls):
        # Create Transformation class
        cls.t1 = pints.LogTransformation(1)
        cls.t4 = pints.LogTransformation(4)

        cls.p = [0.1, 1., 10., 999.]
        cls.x = [-2.3025850929940455, 0., 2.3025850929940459,
                 6.9067547786485539]
        cls.j = np.diag(cls.p)
        cls.j_s1 = np.zeros((4, 4, 4))
        for i in range(4):
            cls.j_s1[i, i, i] = cls.p[i]
        cls.log_j_det = np.sum(cls.x)
        cls.log_j_det_s1 = np.ones(4)

    def test_to_search(self):
        # Test forward transform
        for xi, pi in zip(self.x, self.p):
            calc_xi = self.t1.to_search(pi)
            self.assertAlmostEqual(calc_xi[0], xi)
        self.assertTrue(np.allclose(self.t4.to_search(self.p), self.x))

    def test_to_model(self):
        # Test inverse transform
        for xi, pi in zip(self.x, self.p):
            calc_pi = self.t1.to_model(xi)
            self.assertAlmostEqual(calc_pi[0], pi)
        self.assertTrue(np.allclose(self.t4.to_model(self.x), self.p))

    def test_n_parameters(self):
        # Test n_parameters
        self.assertEqual(self.t1.n_parameters(), 1)
        self.assertEqual(self.t4.n_parameters(), 4)

    def test_jacobian(self):
        # Test Jacobian
        self.assertTrue(np.allclose(self.t4.jacobian(self.x), self.j))

    def test_jacobian_S1(self):
        # Test Jacobian derivatives
        calc_mat, calc_deriv = self.t4.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))

    def test_log_jacobian_det(self):
        # Test log-Jacobian determinant
        self.assertAlmostEqual(self.t4.log_jacobian_det(self.x),
                               self.log_j_det)

    def test_log_jacobian_det_S1(self):
        # Test log-Jacobian determinant derivatives
        calc_val, calc_deriv = self.t4.log_jacobian_det_S1(self.x)
        self.assertAlmostEqual(calc_val, self.log_j_det)
        self.assertTrue(np.all(np.equal(calc_deriv, self.log_j_det_s1)))

    def test_invalid_inputs(self):
        # Test invalid inputs
        with warnings.catch_warnings(record=True):
            self.assertTrue(np.isnan(self.t1.to_search(-1.)))
            self.assertTrue(np.isinf(self.t1.to_search(0)))

    def test_retransform(self):
        # Test forward transform the inverse transform
        self.assertTrue(
            np.allclose(self.p, self.t4.to_model(self.t4.to_search(self.p))))
        self.assertTrue(
            np.allclose(self.x, self.t4.to_search(self.t4.to_model(self.x))))

    def test_elementwise(self):
        # Test is elementwise
        self.assertTrue(self.t1.elementwise())
        self.assertTrue(self.t4.elementwise())


class TestRectangularBoundariesTransformation(unittest.TestCase):
    # Test RectangularBoundariesTransformation class

    @classmethod
    def setUpClass(cls):
        # Create Transformation class

        lower1 = np.array([1])
        upper1 = np.array([10])

        lower2 = np.array([1, 2])
        upper2 = np.array([10, 20])

        # Test normal construction with lower and upper
        cls.t1 = pints.RectangularBoundariesTransformation(lower1, upper1)
        cls.t2 = pints.RectangularBoundariesTransformation(lower2, upper2)

        # Test construction with rectangular boundaries object
        b2 = pints.RectangularBoundaries(lower2, upper2)
        cls.t2b = pints.RectangularBoundariesTransformation(b2)

        cls.p = [1.5, 15.]
        cls.x = [-2.8332133440562162, 0.9555114450274365]
        cls.j = np.diag([0.4722222222222225, 3.6111111111111098])
        cls.j_s1_diag = [0.4197530864197533, -1.6049382716049378]
        cls.j_s1 = np.zeros((2, 2, 2))
        for i in range(2):
            cls.j_s1[i, i, i] = cls.j_s1_diag[i]
        cls.log_j_det = 0.5337099175995788
        cls.log_j_det_s1 = [0.8888888888888888, -0.4444444444444445]

    def test_bad_constructor(self):
        # Test bad constructor
        self.assertRaises(ValueError,
                          pints.RectangularBoundariesTransformation,
                          [1, 2])

    def test_to_search(self):
        # Test forward transform
        self.assertTrue(np.allclose(self.t1.to_search([self.p[0]]),
                                    [self.x[0]]))
        self.assertTrue(np.allclose(self.t2.to_search(self.p), self.x))
        self.assertTrue(np.allclose(self.t2b.to_search(self.p), self.x))

    def test_to_model(self):
        # Test inverse transform
        self.assertTrue(np.allclose(self.t1.to_model([self.x[0]]),
                                    [self.p[0]]))
        self.assertTrue(np.allclose(self.t2.to_model(self.x), self.p))
        self.assertTrue(np.allclose(self.t2b.to_model(self.x), self.p))

    def test_n_parameters(self):
        # Test n_parameters
        self.assertEqual(self.t1.n_parameters(), 1)
        self.assertEqual(self.t2.n_parameters(), 2)
        self.assertEqual(self.t2b.n_parameters(), 2)

    def test_jacobian(self):
        # Test Jacobian
        self.assertTrue(np.allclose(self.t2.jacobian(self.x), self.j))
        self.assertTrue(np.allclose(self.t2b.jacobian(self.x), self.j))

    def test_jacobian_S1(self):
        # Test Jacobian derivatives
        calc_mat, calc_deriv = self.t2.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))
        calc_mat, calc_deriv = self.t2b.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))

    def test_log_jacobian_det(self):
        # Test log-Jacobian determinant
        self.assertAlmostEqual(self.t2.log_jacobian_det(self.x),
                               self.log_j_det)
        self.assertAlmostEqual(self.t2b.log_jacobian_det(self.x),
                               self.log_j_det)

    def test_log_jacobian_det_S1(self):
        # Test log-Jacobian determinant derivatives
        calc_val, calc_deriv = self.t2.log_jacobian_det_S1(self.x)
        self.assertAlmostEqual(calc_val, self.log_j_det)
        self.assertTrue(np.allclose(calc_deriv, self.log_j_det_s1))
        calc_val, calc_deriv = self.t2b.log_jacobian_det_S1(self.x)
        self.assertAlmostEqual(calc_val, self.log_j_det)
        self.assertTrue(np.allclose(calc_deriv, self.log_j_det_s1))

    def test_retransform(self):
        # Test forward transform the inverse transform
        self.assertTrue(
            np.allclose(self.p, self.t2.to_model(self.t2.to_search(self.p))))
        self.assertTrue(
            np.allclose(self.x, self.t2.to_search(self.t2.to_model(self.x))))
        self.assertTrue(
            np.allclose(self.p, self.t2b.to_model(self.t2b.to_search(self.p))))
        self.assertTrue(
            np.allclose(self.x, self.t2b.to_search(self.t2b.to_model(self.x))))

    def test_elementwise(self):
        # Test is elementwise
        self.assertTrue(self.t2.elementwise())
        self.assertTrue(self.t2b.elementwise())


class TestScalingTransformation(unittest.TestCase):
    """ Tests the ScalingTransformation class, without a translation. """

    @classmethod
    def setUpClass(cls):
        # Create Transformation class
        cls.t = pints.ScalingTransformation(
            1. / np.array([-177., 0.333, 10., 99.99]))

        cls.p = [-177., 0.333, 10., 99.99]
        cls.x = [1., 1., 1., 1.]
        cls.j = np.diag([-177., 0.333, 10., 99.99])
        cls.j_s1 = np.zeros((4, 4, 4))
        cls.log_j_det = 10.9841922175539395
        cls.log_j_det_s1 = np.zeros(4)

    def test_to_search(self):
        # Test forward transform
        self.assertTrue(np.allclose(self.t.to_search(self.p), self.x))

    def test_to_model(self):
        # Test inverse transform
        self.assertTrue(np.allclose(self.t.to_model(self.x), self.p))

    def test_n_parameters(self):
        # Test n_parameters
        self.assertEqual(self.t.n_parameters(), 4)

    def test_jacobian(self):
        # Test Jacobian
        self.assertTrue(np.allclose(self.t.jacobian(self.x), self.j))

    def test_jacobian_S1(self):
        # Test Jacobian derivatives
        calc_mat, calc_deriv = self.t.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))

    def test_log_jacobian_det(self):
        # Test log-Jacobian determinant
        self.assertEqual(self.t.log_jacobian_det(self.x), self.log_j_det)

    def test_log_jacobian_det_S1(self):
        # Test log-Jacobian determinant derivatives
        calc_val, calc_deriv = self.t.log_jacobian_det_S1(self.x)
        self.assertEqual(calc_val, self.log_j_det)
        self.assertTrue(np.all(np.equal(calc_deriv, self.log_j_det_s1)))

    def test_retransform(self):
        # Test forward transform the inverse transform
        self.assertTrue(
            np.allclose(self.p, self.t.to_model(self.t.to_search(self.p))))
        self.assertTrue(
            np.allclose(self.x, self.t.to_search(self.t.to_model(self.x))))

    def test_elementwise(self):
        # Test is elementwise
        self.assertTrue(self.t.elementwise())


class TestScalingTransformationWithTranslation(unittest.TestCase):
    """ Tests the ScalingTransformation class, with a translation. """

    @classmethod
    def setUpClass(cls):
        # Create Transformation class
        cls.p = np.array([-77, 0.333, 5, 66.66])
        cls.o = np.array([-100, 0, 5, 33.33])
        cls.s = np.array([-177, 0.333, 10., 99.99])
        cls.t = pints.ScalingTransformation(1 / cls.s, cls.o)
        cls.x = [1., 1., 1., 1.]
        cls.j = np.diag(cls.s)
        cls.j_s1 = np.zeros((4, 4, 4))
        cls.log_j_det = 10.9841922175539395
        cls.log_j_det_s1 = np.zeros(4)

    def test_creation(self):
        # Tests creation options (at the moment just errors)
        pints.ScalingTransformation(1 / self.s, None)
        pints.ScalingTransformation(1 / self.s, list(self.o))
        self.assertRaisesRegex(
            ValueError, 'same length',
            pints.ScalingTransformation, self.s, self.o[:-1])
        self.assertRaisesRegex(
            ValueError, 'same length',
            pints.ScalingTransformation, self.s, [1, 2, 3, 4, 5])

    def test_to_search(self):
        # Test forward transform
        self.assertTrue(np.allclose(self.t.to_search(self.p), self.x))

    def test_to_model(self):
        # Test inverse transform
        self.assertTrue(np.allclose(self.t.to_model(self.x), self.p))

    def test_n_parameters(self):
        # Test n_parameters
        self.assertEqual(self.t.n_parameters(), 4)

    def test_jacobian(self):
        # Test Jacobian
        self.assertTrue(np.allclose(self.t.jacobian(self.x), self.j))

    def test_jacobian_S1(self):
        # Test Jacobian derivatives
        calc_mat, calc_deriv = self.t.jacobian_S1(self.x)
        self.assertTrue(np.allclose(calc_mat, self.j))
        self.assertTrue(np.allclose(calc_deriv, self.j_s1))

    def test_log_jacobian_det(self):
        # Test log-Jacobian determinant
        self.assertEqual(self.t.log_jacobian_det(self.x), self.log_j_det)

    def test_log_jacobian_det_S1(self):
        # Test log-Jacobian determinant derivatives
        calc_val, calc_deriv = self.t.log_jacobian_det_S1(self.x)
        self.assertEqual(calc_val, self.log_j_det)
        self.assertTrue(np.all(np.equal(calc_deriv, self.log_j_det_s1)))

    def test_retransform(self):
        # Test forward transform the inverse transform
        self.assertTrue(
            np.allclose(self.p, self.t.to_model(self.t.to_search(self.p))))
        self.assertTrue(
            np.allclose(self.x, self.t.to_search(self.t.to_model(self.x))))

    def test_elementwise(self):
        # Test is elementwise
        self.assertTrue(self.t.elementwise())


class TestUnitCubeTransformation(unittest.TestCase):
    """
    Tests the UnitCubeTransformation class.

    Most methods are tested in the ScalingTransformation tests
    """

    @classmethod
    def setUpClass(cls):
        # Create Transformation class
        cls.lower = np.array([-1, 2, -3])
        cls.upper = np.array([0, 4, -1])
        cls.t = pints.UnitCubeTransformation(cls.lower, cls.upper)

    def test_creation(self):
        # Tests creation options (at the moment just errors)
        pints.UnitCubeTransformation(self.lower, self.upper)
        pints.UnitCubeTransformation(self.lower, [10, 10, 10])
        pints.UnitCubeTransformation((-10, -20, -30), self.upper)

        self.assertRaisesRegex(
            ValueError, 'same length',
            pints.UnitCubeTransformation, (1, 2), [3])
        self.assertRaisesRegex(
            ValueError, 'same length',
            pints.UnitCubeTransformation, [3, 4, 5], (10, 10, 10, 10))
        self.assertRaisesRegex(
            ValueError, 'must exceed',
            pints.UnitCubeTransformation, (1, 2), (0, 3))
        self.assertRaisesRegex(
            ValueError, 'must exceed',
            pints.UnitCubeTransformation, (1, 2), (3, 1))
        self.assertRaisesRegex(
            ValueError, 'must exceed',
            pints.UnitCubeTransformation, (1, 2), (1, 3))
        self.assertRaisesRegex(
            ValueError, 'must exceed',
            pints.UnitCubeTransformation, (1, 2), (3, 2))

    def test_to_search(self):
        # Test forward transform
        self.assertTrue(np.allclose(self.t.to_search(self.lower), [0, 0, 0]))
        self.assertTrue(np.allclose(self.t.to_search(self.upper), [1, 1, 1]))

    def test_to_model(self):
        # Test inverse transform
        self.assertTrue(np.allclose(self.t.to_model([0, 0, 0]), self.lower))
        self.assertTrue(np.allclose(self.t.to_model([1, 1, 1]), self.upper))


class TestTransformedWrappers(unittest.TestCase):
    """
    Tests the wrapped boundaries, error, logpdf, and logprior classes.
    """

    def test_transformed_boundaries(self):
        # Test the TransformedBoundaries class

        t = pints.LogTransformation(2)
        b = pints.RectangularBoundaries([0.01, 0.95], [0.05, 1.05])
        tb = pints.TransformedBoundaries(b, t)
        xi = [0.02, 1.01]
        txi = [-3.9120230054281460, 0.0099503308531681]
        xo = [10., 50.]
        txo = [2.3025850929940459, 3.9120230054281460]

        # Test before and after transformed give the same result
        self.assertEqual(tb.check(txi), b.check(xi))
        self.assertEqual(tb.check(txo), b.check(xo))
        self.assertEqual(tb.n_parameters(), b.n_parameters())

        # Wrong number of parameters
        self.assertRaisesRegex(
            ValueError, 'Number of parameters for boundaries and transfo',
            pints.TransformedBoundaries, b, pints.LogTransformation(3))

        # Test sampling from untransformed space (but converted to transformed)
        np.random.seed(1)
        x1 = b.sample(1)
        x2 = b.sample(3)
        self.assertEqual(len(x1), 1)
        self.assertEqual(len(x2), 3)
        np.random.seed(1)
        y1 = tb.sample(1)
        y2 = tb.sample(3)
        self.assertEqual(len(y1), 1)
        self.assertEqual(len(y2), 3)
        self.assertEqual(list(t.to_search(x1[0])), list(y1[0]))
        self.assertEqual(list(t.to_search(x2[0])), list(y2[0]))
        self.assertEqual(list(t.to_search(x2[1])), list(y2[1]))
        self.assertEqual(list(t.to_search(x2[2])), list(y2[2]))

    def test_transformed_rectangular_boundaries(self):
        # Test the TransformedRectangularBoundaries class

        # Test automatic creation
        t = pints.LogTransformation(2)
        b = pints.RectangularBoundaries([0.01, 0.95], [0.05, 1.05])
        tb = t.convert_boundaries(b)
        self.assertIsInstance(tb, pints.TransformedRectangularBoundaries)

        # Test before and after transformed give the same result
        xi = [0.02, 1.01]
        txi = [-3.9120230054281460, 0.0099503308531681]
        xo = [10., 50.]
        txo = [2.3025850929940459, 3.9120230054281460]
        self.assertEqual(tb.check(txi), b.check(xi))
        self.assertEqual(tb.check(txo), b.check(xo))
        self.assertEqual(tb.n_parameters(), b.n_parameters())

        # Test sampling still happens in untransformed space
        np.random.seed(1)
        x1 = b.sample(1)
        x2 = b.sample(3)
        self.assertEqual(len(x1), 1)
        self.assertEqual(len(x2), 3)
        np.random.seed(1)
        y1 = tb.sample(1)
        y2 = tb.sample(3)
        self.assertEqual(len(y1), 1)
        self.assertEqual(len(y2), 3)
        self.assertEqual(list(t.to_search(x1[0])), list(y1[0]))
        self.assertEqual(list(t.to_search(x2[0])), list(y2[0]))
        self.assertEqual(list(t.to_search(x2[1])), list(y2[1]))
        self.assertEqual(list(t.to_search(x2[2])), list(y2[2]))

        # Test (a few) rectangular boundary methods
        self.assertEqual(list(tb.lower()), [np.log(0.01), np.log(0.95)])
        self.assertEqual(list(tb.upper()), [np.log(0.05), np.log(1.05)])

        # Test that boundaries get swapped if needed
        t2 = pints.ScalingTransformation([4, -2])
        tb = pints.TransformedRectangularBoundaries(b, t2)
        self.assertEqual(list(tb.lower()), [0.04, -2.1])
        self.assertEqual(list(tb.upper()), [0.20, -1.9])

        # Wrong number of parameters
        self.assertRaisesRegex(
            ValueError, 'Number of parameters for boundaries and transfo',
            pints.TransformedRectangularBoundaries, b,
            pints.LogTransformation(3))

        # Not rectangular boundaries
        cb = CircularBoundaries([0, 0])
        tcb = t.convert_boundaries(cb)
        self.assertIsInstance(tcb, pints.TransformedBoundaries)
        self.assertNotIsInstance(tcb, pints.TransformedRectangularBoundaries)
        self.assertRaisesRegex(
            ValueError, 'can only be created from a RectangularBoundaries obj',
            pints.TransformedRectangularBoundaries, cb, t)

        # Not an elementwise transformation
        cb = CircularBoundaries([0, 0])
        tcb = t.convert_boundaries(cb)
        self.assertIsInstance(tcb, pints.TransformedBoundaries)
        self.assertNotIsInstance(tcb, pints.TransformedRectangularBoundaries)
        self.assertRaisesRegex(
            ValueError, 'can only be created from a RectangularBoundaries obj',
            pints.TransformedRectangularBoundaries, cb, t)

        # Not a piecewise transformation
        t2 = SwappingTransformation(2)
        self.assertRaisesRegex(
            ValueError, 'can only be created from an element-wise trans',
            pints.TransformedRectangularBoundaries, b, t2)

    def test_transformed_error_measure(self):
        # Test TransformedErrorMeasure class

        t = pints.LogTransformation(2)
        r = pints.toy.ParabolicError()
        x = [0.1, 0.1]
        tx = [-2.3025850929940455, -2.3025850929940455]
        j = np.diag(x)
        tr = t.convert_error_measure(r)

        # Test before and after transformed give the same result
        self.assertAlmostEqual(tr(tx), r(x))
        self.assertEqual(tr.n_parameters(), r.n_parameters())

        # Test evaluateS1()
        rx, s1 = r.evaluateS1(x)
        ts1 = np.matmul(s1, j)
        trtx, trts1 = tr.evaluateS1(tx)
        self.assertTrue(np.allclose(trtx, rx))
        self.assertTrue(np.allclose(trts1, ts1))

        # Test invalid transform
        self.assertRaises(ValueError, pints.TransformedErrorMeasure, r,
                          pints.LogTransformation(3))

    def test_transformed_log_pdf(self):
        # Test TransformedLogPDF class

        t = pints.LogTransformation(2)
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = [0.05, 1.01]
        tx = [-2.9957322735539909, 0.0099503308531681]
        j = np.diag(x)
        log_j_det = -2.9857819427008230
        tr = t.convert_log_pdf(r)

        # Test before and after transformed give the same result
        self.assertAlmostEqual(tr(tx), r(x) + log_j_det)
        self.assertEqual(tr.n_parameters(), r.n_parameters())

        # Test evaluateS1()
        rx, s1 = r.evaluateS1(x)
        trx = rx + log_j_det
        ts1 = np.matmul(s1, j) + np.ones(2)
        trtx, trts1 = tr.evaluateS1(tx)
        self.assertTrue(np.allclose(trtx, trx))
        self.assertTrue(np.allclose(trts1, ts1))

        # Test invalid transform
        self.assertRaises(ValueError, pints.TransformedLogPDF, r,
                          pints.LogTransformation(3))

    def test_transformed_log_prior(self):
        # Test TransformedLogPrior class

        d = 2
        t = pints.LogTransformation(2)
        r = pints.UniformLogPrior([0.1, 0.1], [0.9, 0.9])
        tr = t.convert_log_prior(r)

        # Test sample
        n = 1
        x = tr.sample(n)
        self.assertEqual(x.shape, (n, d))
        self.assertTrue(np.all(x < 0.))
        n = 1000
        x = tr.sample(n)
        self.assertEqual(x.shape, (n, d))
        self.assertTrue(np.all(x < 0.))


if __name__ == '__main__':
    unittest.main()
