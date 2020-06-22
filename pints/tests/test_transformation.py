#!/usr/bin/env python3
#
# Tests Transform functions in Pints
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import division
import unittest
import pints
import pints.toy
import numpy as np


class TestTransform(unittest.TestCase):

    def test_log_transform(self):
        # Test LogTransform class

        # Test input parameters
        t1 = pints.LogTransform(1)
        t4 = pints.LogTransform(4)

        p = [0.1, 1., 10., 999.]
        x = [-2.3025850929940455, 0., 2.3025850929940459, 6.9067547786485539]
        j = np.diag(p)
        log_j_det = np.sum(x)

        # Test forward transform
        for xi, pi in zip(x, p):
            calc_xi = t1.to_search(pi)
            self.assertAlmostEqual(calc_xi[0], xi)
        self.assertTrue(np.allclose(t4.to_search(p), x))

        # Test inverse transform
        for xi, pi in zip(x, p):
            calc_pi = t1.to_model(xi)
            self.assertAlmostEqual(calc_pi[0], pi)
        self.assertTrue(np.allclose(t4.to_model(x), p))

        # Test n_parameters
        self.assertEqual(t1.n_parameters(), 1)
        self.assertEqual(t4.n_parameters(), 4)

        # Test Jacobian
        self.assertTrue(np.allclose(t4.jacobian(x), j))

        # Test log-Jacobian determinant
        self.assertEqual(t4.log_jacobian_det(x), log_j_det)

        # Test invalid inputs
        self.assertTrue(np.isnan(t1.to_search(-1.)))
        self.assertTrue(np.isinf(t1.to_search(0)))

    def test_logit_transform(self):
        # Test LogitTransform class

        # Test input parameters
        t1 = pints.LogitTransform(1)
        t4 = pints.LogitTransform(4)

        p = [0.1, 0.333, 0.5, 0.9]
        x = [-2.1972245773362191, -0.6946475559351799, 0., 2.1972245773362196]
        j = np.diag([0.09, 0.222111, 0.25, 0.09])
        log_j_det = -7.7067636004918398

        # Test forward transform
        for xi, pi in zip(x, p):
            calc_xi = t1.to_search(pi)
            self.assertAlmostEqual(calc_xi[0], xi)
        self.assertTrue(np.allclose(t4.to_search(p), x))

        # Test inverse transform
        for xi, pi in zip(x, p):
            calc_pi = t1.to_model(xi)
            self.assertAlmostEqual(calc_pi[0], pi)
        self.assertTrue(np.allclose(t4.to_model(x), p))

        # Test n_parameters
        self.assertEqual(t1.n_parameters(), 1)
        self.assertEqual(t4.n_parameters(), 4)

        # Test Jacobian
        self.assertTrue(np.allclose(t4.jacobian(x), j))

        # Test log-Jacobian determinant
        self.assertEqual(t4.log_jacobian_det(x), log_j_det)

        # Test invalid inputs
        self.assertTrue(np.isnan(t1.to_search(-1.)))
        self.assertTrue(np.isnan(t1.to_search(2.)))
        self.assertTrue(np.isinf(t1.to_search(1.)))

    def test_rectangular_boundaries_transform(self):
        # Test RectangularBoundariesTransform class

        lower1 = np.array([1])
        upper1 = np.array([10])

        lower2 = np.array([1, 2])
        upper2 = np.array([10, 20])

        # Test normal construction with lower and upper
        t1 = pints.RectangularBoundariesTransform(lower1, upper1)
        t2 = pints.RectangularBoundariesTransform(lower2, upper2)

        # Test construction with rectangular boundaries object
        b2 = pints.RectangularBoundaries(lower2, upper2)
        t2b = pints.RectangularBoundariesTransform(b2)

        # Test bad constructor
        self.assertRaises(ValueError, pints.RectangularBoundariesTransform,
                          lower2)

        p = [1.5, 15.]
        x = [-2.8332133440562162, 0.9555114450274365]
        j = np.diag([0.4722222222222225, 3.6111111111111098])
        log_j_det = 0.5337099175995788

        # Test forward transform
        self.assertTrue(np.allclose(t1.to_search([p[0]]), [x[0]]))
        self.assertTrue(np.allclose(t2.to_search(p), x))
        self.assertTrue(np.allclose(t2b.to_search(p), x))

        # Test inverse transform
        self.assertTrue(np.allclose(t1.to_model([x[0]]), [p[0]]))
        self.assertTrue(np.allclose(t2.to_model(x), p))
        self.assertTrue(np.allclose(t2b.to_model(x), p))

        # Test n_parameters
        self.assertEqual(t1.n_parameters(), 1)
        self.assertEqual(t2.n_parameters(), 2)
        self.assertEqual(t2b.n_parameters(), 2)

        # Test Jacobian
        self.assertTrue(np.allclose(t2.jacobian(x), j))
        self.assertTrue(np.allclose(t2b.jacobian(x), j))

        # Test log-Jacobian determinant
        self.assertEqual(t2.log_jacobian_det(x), log_j_det)
        self.assertEqual(t2b.log_jacobian_det(x), log_j_det)

    def test_identity_transform(self):
        # Test IdentityTransform class

        # Test input parameters
        t1 = pints.IdentityTransform(1)
        t4 = pints.IdentityTransform(4)

        p = [-177., 0.333, 10., 99.99]
        x = [-177., 0.333, 10., 99.99]
        j = np.eye(4)
        log_j_det = 0.

        # Test forward transform
        for xi, pi in zip(x, p):
            calc_xi = t1.to_search(pi)
            self.assertAlmostEqual(calc_xi[0], xi)
        self.assertTrue(np.allclose(t4.to_search(p), x))

        # Test inverse transform
        for xi, pi in zip(x, p):
            calc_pi = t1.to_model(xi)
            self.assertAlmostEqual(calc_pi[0], pi)
        self.assertTrue(np.allclose(t4.to_model(x), p))

        # Test n_parameters
        self.assertEqual(t1.n_parameters(), 1)
        self.assertEqual(t4.n_parameters(), 4)

        # Test Jacobian
        self.assertTrue(np.allclose(t4.jacobian(x), j))

        # Test log-Jacobian determinant
        self.assertEqual(t4.log_jacobian_det(x), log_j_det)

    def test_composed_transform(self):
        # Test ComposedTransform class

        # Test input parameters
        t1 = pints.IdentityTransform(1)
        lower2 = np.array([1, 2])
        upper2 = np.array([10, 20])
        t2 = pints.RectangularBoundariesTransform(lower2, upper2)
        t3 = pints.LogTransform(1)

        t = pints.ComposedTransform(t1, t2, t3)

        p = [0.1, 1.5, 15., 999.]
        x = [0.1, -2.8332133440562162, 0.9555114450274365, 6.9067547786485539]
        j = np.diag([1., 0.4722222222222225, 3.6111111111111098, 999.])
        log_j_det = 7.4404646962481324

        # Test forward transform
        self.assertTrue(np.allclose(t.to_search(p), x))

        # Test inverse transform
        self.assertTrue(np.allclose(t.to_model(x), p))

        # Test n_parameters
        self.assertEqual(t.n_parameters(), 4)

        # Test Jacobian
        self.assertTrue(np.allclose(t.jacobian(x), j))

        # Test log-Jacobian determinant
        self.assertEqual(t.log_jacobian_det(x), log_j_det)

        # Test invalid constructors
        self.assertRaises(ValueError, pints.ComposedTransform)
        self.assertRaises(ValueError, pints.ComposedTransform, np.log)


class TestTransformedWrappers(unittest.TestCase):

    def test_transformed_error_measure(self):
        # Test TransformedErrorMeasure class

        t = pints.LogTransform(2)
        r = pints.toy.ParabolicError()
        x = [0.1, 0.1]
        tx = [-2.3025850929940455, -2.3025850929940455]
        j = np.diag(x)
        tr = t.apply_error_measure(r)

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
                          pints.LogTransform(3))

    def test_transformed_log_pdf(self):
        # Test TransformedLogPDF class

        t = pints.LogTransform(2)
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = [0.05, 1.01]
        tx = [-2.9957322735539909, 0.0099503308531681]
        j = np.diag(x)
        log_j_det = -2.9857819427008230
        tr = t.apply_log_pdf(r)

        # Test before and after transformed give the same result
        self.assertAlmostEqual(tr(tx), r(x) + log_j_det)
        self.assertEqual(tr.n_parameters(), r.n_parameters())

        # Test evaluateS1()
        rx, s1 = r.evaluateS1(x)
        trx = rx + log_j_det
        ts1 = np.matmul(s1, j)
        trtx, trts1 = tr.evaluateS1(tx)
        self.assertTrue(np.allclose(trtx, trx))
        self.assertTrue(np.allclose(trts1, ts1))

        # Test invalid transform
        self.assertRaises(ValueError, pints.TransformedLogPDF, r,
                          pints.LogTransform(3))

    def test_transformed_boundaries(self):
        # Test TransformedBoundaries class

        t = pints.LogTransform(2)
        b = pints.RectangularBoundaries([0.01, 0.95], [0.05, 1.05])
        tb = t.apply_boundaries(b)
        xi = [0.02, 1.01]
        txi = [-3.9120230054281460, 0.0099503308531681]
        xo = [10., 50.]
        txo = [2.3025850929940459, 3.9120230054281460]
        tr = [1.6094379124341001, 0.1000834585569826]

        # Test before and after transformed give the same result
        self.assertEqual(tb.check(txi), b.check(xi))
        self.assertEqual(tb.check(txo), b.check(xo))
        self.assertEqual(tb.n_parameters(), b.n_parameters())

        # Test transformed range
        self.assertTrue(np.allclose(tb.range(), tr))

        # Test invalid transform
        self.assertRaises(ValueError, pints.TransformedBoundaries, b,
                          pints.LogTransform(3))


if __name__ == '__main__':
    unittest.main()
