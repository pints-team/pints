#!/usr/bin/env python3
#
# Tests the annulus distribution.
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


class TestAnnulusLogPDF(unittest.TestCase):
    """
    Tests the annulus log-pdf toy problems.
    """
    def test_default(self):

        # Default settings
        f = pints.toy.AnnulusLogPDF()
        self.assertEqual(f.n_parameters(), 2)
        self.assertEqual(f.r0(), 10)
        self.assertEqual(f.sigma(), 1)
        f1 = f([1, 1])
        f2 = f([10, 0])
        self.assertTrue(np.isscalar(f1))
        self.assertTrue(np.isscalar(f2))
        self.assertAlmostEqual(f1, -37.776802909473716)
        self.assertAlmostEqual(f2, -0.91893853320467267)
        self.assertAlmostEqual(f.mean_normed(), 10.099999999999993)
        self.assertAlmostEqual(f.var_normed(), 0.99000000000020805)
        self.assertAlmostEqual(f.moment_normed(3), 1060.3)
        a_mean = f.mean()
        self.assertEqual(a_mean[0], 0)
        self.assertEqual(a_mean[1], 0)

        # Change dimensions and beta
        f = pints.toy.AnnulusLogPDF(10, 15, 0.5)
        self.assertEqual(f.n_parameters(), 10)
        self.assertAlmostEqual(f.mean_normed(), 15.148688458505298)
        self.assertAlmostEqual(f.var_normed(), 0.24756486472776373)
        self.assertAlmostEqual(f.moment_normed(5), 806385.71374340181)

        # Test sample function
        x = f.sample(10)
        self.assertEqual(len(x), 10)
        f = pints.toy.AnnulusLogPDF()
        self.assertTrue(np.max(f.sample(1000)) < 100)
        self.assertRaises(ValueError, f.sample, 0)
        f = pints.toy.AnnulusLogPDF()
        samples = f.sample(100000)
        self.assertTrue(np.abs(np.mean(samples)) < 0.1)

        # Test _reject_sample_r
        self.assertTrue(np.all(f._reject_sample(100) > 0))

        # Bad constructors
        self.assertRaises(
            ValueError, pints.toy.AnnulusLogPDF, 0, 1, 1)
        self.assertRaises(
            ValueError, pints.toy.AnnulusLogPDF, 1, 0, 1)
        self.assertRaises(
            ValueError, pints.toy.AnnulusLogPDF, 3, 1, -1)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
