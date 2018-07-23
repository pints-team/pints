#!/usr/bin/env python3
#
# Tests the twisted gaussian logpdf toy distribution.
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


class TestTwistedGaussianLogPDF(unittest.TestCase):
    """
    Tests the twisted gaussian logpdf toy distribution.
    """
    def test_twisted_gaussian_logpdf(self):
        # Test basics
        f = pints.toy.TwistedGaussianLogPDF()
        self.assertEqual(f.n_parameters(), 10)
        self.assertTrue(np.isscalar(f(np.zeros(10))))

        # TODO: Test more?

        # Test errors
        self.assertRaises(
            ValueError, pints.toy.TwistedGaussianLogPDF, 1)
        self.assertRaises(
            ValueError, pints.toy.TwistedGaussianLogPDF, b=-1)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
