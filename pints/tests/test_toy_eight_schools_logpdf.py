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


class TestEightSchoolsLogPDF(unittest.TestCase):
    """
    Tests the cone log-pdf toy problems.
    """
    def test_basic(self):
        """ Tests calls and data. """

        # Default settings
        f = pints.toy.EightSchoolsLogPDF()
        f1, dp = f.evaluateS1(np.ones(10))
        self.assertEqual(f1, f(np.ones(10)))
        self.assertAlmostEqual(f1, -43.02226038161451)
        self.assertEqual(len(dp), 10)
        self.assertEqual(dp[0], -1.0 / 25)
        self.assertAlmostEqual(dp[1], -8.076923076923077, places=6)
        self.assertEqual(dp[2], 3.0 / 25)
        val = f([1, 0.5, 0.4, 1, 1, 1, 1, 1, 1, 1])
        self.assertAlmostEqual(val, -38.24061255483484, places=6)

        # non-centered paramerisation
        f = pints.toy.EightSchoolsLogPDF(centered=False)
        f1, dp = f.evaluateS1(np.ones(10))
        self.assertEqual(f1, f(np.ones(10)))
        self.assertAlmostEqual(f1, -46.649195204910605)
        self.assertEqual(len(dp), 10)
        self.assertAlmostEqual(dp[0], 0.3029093172890521, places=6)
        self.assertAlmostEqual(dp[1], 0.2659862403659751, places=6)
        self.assertAlmostEqual(dp[2], -0.8844444444444445, places=6)
        val = f([1, 0.5, 0.4, 1, 1, 1, 1, 1, 1, 1])
        self.assertAlmostEqual(val, -46.41445177944207, places=6)

        # Test data
        data = f.data()
        self.assertEqual(len(data), 3)
        self.assertEqual(data["J"], 8)
        self.assertTrue(np.array_equal(data["y"],
                        [28, 8, -3, 7, -1, 1, 18, 12]))
        self.assertTrue(np.array_equal(data["sigma"],
                        [15, 10, 16, 11, 9, 11, 10, 18]))

    def test_bad_calls(self):
        # Tests bad calls

        # Bad calls to function
        f = pints.toy.EightSchoolsLogPDF()
        self.assertRaises(ValueError, f.__call__, [1, 2, 3])
        self.assertRaises(ValueError, f.__call__, np.ones(11))

        # Bad calls to evaluate
        self.assertRaises(ValueError, f.evaluateS1, [1, 2, 3])
        self.assertRaises(ValueError, f.evaluateS1, np.ones(11))

    def test_bounds(self):
        """ Tests suggested_bounds() """
        f = pints.toy.EightSchoolsLogPDF()
        bounds = f.suggested_bounds()
        self.assertEqual(bounds[0][1], 0)


if __name__ == '__main__':
    unittest.main()
