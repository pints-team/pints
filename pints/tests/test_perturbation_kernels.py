#!/usr/bin/env python3
#
# Tests the Perturbation Kernel classes
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import unittest
import numpy as np

# Strings in Python 2 and 3
try:
    basestring
except NameError:
    basestring = str


class TestPerturbationKernels(unittest.TestCase):
    """
    Tests the basic methods of the Timer class.
    """
    def __init__(self, name):
        super(TestPerturbationKernels, self).__init__(name)

    def test_perturb(self):
        """ Test the time() and reset() methods. """
        k = pints.SphericalGaussianKernel(1, 1)
        x = k.perturb(np.array([0]))
        p = k.p(x, np.array([0]))
        self.assertLessEqual(p, 1)

    def test_errors(self):
        k = pints.SphericalGaussianKernel(1, 1)
        k.p([0], [0,1])

if __name__ == '__main__':
    unittest.main()
