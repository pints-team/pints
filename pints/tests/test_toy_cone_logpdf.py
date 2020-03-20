#!/usr/bin/env python3
#
# Tests the cone distribution.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestConeLogPDF(unittest.TestCase):
    """
    Tests the cone log-pdf toy problems.
    """
    def test_basic(self):
        # Tests moments, calls, CDF evaluations and sampling

        # Default settings
        f = pints.toy.ConeLogPDF()
        self.assertEqual(f.n_parameters(), 2)
        self.assertEqual(f.beta(), 1)
        f1 = f([1, 1])
        f2 = f([0, 0])
        self.assertTrue(np.isscalar(f1))
        self.assertTrue(np.isscalar(f2))
        self.assertAlmostEqual(f1, -1.4142135623730951)
        self.assertEqual(f.mean_normed(), 2.0)
        self.assertEqual(f.var_normed(), 2.0)

        # Change dimensions and beta
        f = pints.toy.ConeLogPDF(10, 0.5)
        self.assertEqual(f.n_parameters(), 10)
        self.assertEqual(f.beta(), 0.5)
        self.assertEqual(f.mean_normed(), 420.0)
        self.assertEqual(f.var_normed(), 36120.0)
        f1 = f(np.repeat(1, 10))
        self.assertAlmostEqual(f1, -1.7782794100389228)

        # Test CDF function
        f = pints.toy.ConeLogPDF()
        self.assertAlmostEqual(f.CDF(1.0), 0.26424111765711533)
        self.assertAlmostEqual(f.CDF(2.5), 0.71270250481635422)
        f = pints.toy.ConeLogPDF(3, 2)
        self.assertAlmostEqual(f.CDF(1.0), 0.42759329552912018)
        self.assertRaises(ValueError, f.CDF, -1)

        # Test sample function
        x = f.sample(10)
        self.assertEqual(len(x), 10)
        f = pints.toy.ConeLogPDF(2, 2)
        self.assertTrue(np.max(f.sample(1000)) < 10)
        self.assertRaises(ValueError, f.sample, 0)

    def test_bad_constructors(self):
        # Tests bad instantiations and calls

        self.assertRaises(
            ValueError, pints.toy.ConeLogPDF, 0, 1)
        self.assertRaises(
            ValueError, pints.toy.ConeLogPDF, 1, 0)
        self.assertRaises(
            ValueError, pints.toy.ConeLogPDF, 3, -1)

        # Bad calls to function
        f = pints.toy.ConeLogPDF(4, 0.3)
        self.assertRaises(ValueError, f.__call__, [1, 2, 3])
        self.assertRaises(ValueError, f.__call__, [1, 2, 3, 4, 5])

    def test_bounds(self):
        # Tests suggested_bounds()

        f = pints.toy.ConeLogPDF()
        bounds = f.suggested_bounds()
        self.assertTrue(np.array_equal([[-1000, -1000], [1000, 1000]],
                                       bounds))
        beta = 3
        dimensions = 4
        f = pints.toy.ConeLogPDF(beta=beta, dimensions=dimensions)
        magnitude = 1000
        bounds = np.tile([-magnitude, magnitude], (dimensions, 1))
        self.assertEqual(bounds[0][0], -magnitude)
        self.assertEqual(bounds[0][1], magnitude)
        self.assertTrue(np.array_equal(np.array(bounds).shape, [4, 2]))

    def test_sensitivities(self):
        # Tests sensitivities

        f = pints.toy.ConeLogPDF()
        l, dl = f.evaluateS1([-1, 3])
        self.assertEqual(len(dl), 2)
        self.assertEqual(l, -np.sqrt(10))
        self.assertAlmostEqual(dl[0], np.sqrt(1.0 / 10))
        self.assertAlmostEqual(dl[1], -3 * np.sqrt(1.0 / 10))
        f = pints.toy.ConeLogPDF(10, 0.3)
        xx = [-1, 3, 2, 4, 5, 6, 7, 8, 9, 10]
        l, dl = f.evaluateS1(xx)
        self.assertEqual(len(dl), 10)
        self.assertEqual(l, -np.sqrt(385)**0.3)
        cons = -(385**(-1 + 0.15)) * 0.3
        for i, elem in enumerate(dl):
            self.assertAlmostEqual(elem, cons * xx[i])

    def test_distance_function(self):
        # Tests distance function

        f = pints.toy.ConeLogPDF()
        x = f.sample(10)
        self.assertTrue(f.distance(x) > 0)
        x = np.ones((100, 3))
        self.assertRaises(ValueError, f.distance, x)
        x = np.ones((100, 3, 2))
        self.assertRaises(ValueError, f.distance, x)

        f = pints.toy.ConeLogPDF(5)
        x = f.sample(10)
        self.assertTrue(f.distance(x) > 0)
        x = np.ones((100, 4))
        self.assertRaises(ValueError, f.distance, x)
        x = np.ones((100, 6))
        self.assertRaises(ValueError, f.distance, x)
        x = np.ones((100, 5, 2))
        self.assertRaises(ValueError, f.distance, x)


if __name__ == '__main__':
    unittest.main()
