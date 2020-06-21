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
import numpy as np


class TestTransform(unittest.TestCase):

    def test_log_transform(self):

        # Test input parameters
        t1 = pints.LogTransform(1)
        t4 = pints.LogTransform(4)

        p = [0.1, 1., 10., 999.]
        x = [-2.3025850929940455, 0., 2.3025850929940459, 6.9067547786485539]

        # Test forward transform
        for xi, pi in zip(x, p):
            calc_xi = t1.to_search(pi)
            self.assertAlmostEqual(calc_xi, xi)

        # Test inverse transform
        for xi, pi in zip(x, p):
            calc_pi = t1.to_model(xi)
            self.assertAlmostEqual(calc_pi, pi)

        # Test n_parameters
        self.assertEqual(t1.n_parameters(), 1)
        self.assertEqual(t4.n_parameters(), 4)

        # Test Jacobian

        # Test log-Jacobian determinant


if __name__ == '__main__':
    unittest.main()
