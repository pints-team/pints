#!/usr/bin/env python3
#
# Tests the cone distribution.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestEightSchoolsCenteredLogPDF(unittest.TestCase):
    """
    Tests the cone log-pdf toy problems.
    """
    def test_basic(self):
        """ Tests calls and data. """

        # Default settings
        f = pints.toy.EightSchoolsCenteredLogPDF()
        f1, dp = f.evaluateS1(np.ones(10))
        self.assertEqual(f1, f(np.ones(10)))
        self.assertAlmostEqual(f1, -43.02226038161451)
        self.assertAlmostEqual(f1(np.ones(10)), -43.02226038161451, places=6)
        self.assertEqual(dp[0], -1.0 / 25)
        self.assertAlmostEqual(dp[1], -8.076923076923077, places=6)
        self.assertEqual(dp[2], 3.0 / 25)
        val = f([1, 0.5, 0.4, 1, 1, 1, 1, 1, 1, 1])
        self.assertAlmostEqual(val, -38.24061255483484, places=6)

    def test_bad_constructors(self):
        """ Tests bad instantiations and calls """
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
        """ Tests suggested_bounds() """
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
        """ Tests sensitivities """
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


if __name__ == '__main__':
    unittest.main()
