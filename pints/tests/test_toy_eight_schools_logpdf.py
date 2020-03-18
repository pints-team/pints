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
        self.assertEqual(dp[0], -1.0 / 25)
        self.assertEqual(len(dp), 10)
        self.assertAlmostEqual(dp[1], -8.076923076923077, places=6)
        self.assertEqual(dp[2], 3.0 / 25)
        val = f([1, 0.5, 0.4, 1, 1, 1, 1, 1, 1, 1])
        self.assertAlmostEqual(val, -38.24061255483484, places=6)

    def test_bad_calls(self):
        # Tests bad calls

        # Bad calls to function
        f = pints.toy.EightSchoolsCenteredLogPDF()
        self.assertRaises(ValueError, f.__call__, [1, 2, 3])
        self.assertRaises(ValueError, f.__call__, np.ones(11))

        # Bad calls to evaluate
        self.assertRaises(ValueError, f.evaluateS1, [1, 2, 3])
        self.assertRaises(ValueError, f.evaluateS1, np.ones(11))

    def test_bounds(self):
        """ Tests suggested_bounds() """
        f = pints.toy.EightSchoolsCenteredLogPDF()
        bounds = f.suggested_bounds()
        self.assertEqual(bounds[0][1], 0)


if __name__ == '__main__':
    unittest.main()
