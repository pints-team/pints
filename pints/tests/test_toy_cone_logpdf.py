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


class TestConeLogPDF(unittest.TestCase):
    """
    Tests the cone log-pdf toy problems.
    """
    def test_default(self):

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

        # Bad constructors
        self.assertRaises(
            ValueError, pints.toy.ConeLogPDF, 0, 1)
        self.assertRaises(
            ValueError, pints.toy.ConeLogPDF, 1, 0)
        self.assertRaises(
            ValueError, pints.toy.ConeLogPDF, 3, -1)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
