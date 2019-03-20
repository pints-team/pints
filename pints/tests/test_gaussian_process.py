#!/usr/bin/env python3
#
# Tests the basic methods of the CMAES optimiser.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np

import pints
import pints.toy

debug = False

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestGaussianProcess(unittest.TestCase):
    """
    Tests the basic methods of the gaussian process log pdf.
    """
    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

    def problem(self):
        """ Returns a test problem. """
        d = 2
        mean = np.array([3.0, -3.0])
        sigma = np.array([[1, 0], [0, 1]])
        log_pdf = pints.toy.GaussianLogPDF(mean, sigma)

        return log_pdf

    def test_fitting(self):
        """ fits the gp to the problem. """
        log_pdf = self.problem()

        n = 10000
        samples = log_pdf.sample(n)
        values = log_pdf(samples)
        gp = pints.GaussianProcess(samples, values)

if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
