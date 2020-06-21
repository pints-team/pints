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


class TestLogTransform(unittest.TestCase):

    def test_creation(self):
        # Test transform object is working fine

        # Test input parameters
        t1 = pints.LogTransform(1)
        t4 = pints.LogTransform(4)

        p = [0.1, 1., 10., 999.]
        x = [-2.3025850929940455, 0., 2.3025850929940459, 6.9067547786485539]
        j = np.diag(p)
        log_j_det = np.sum(x)

        # Test forward transform
        for xi, pi in zip(x, p):
            calc_xi = t1.to_search(pi)
            self.assertAlmostEqual(calc_xi, xi)
        self.assertTrue(np.allclose(t4.to_search(p), x))

        # Test inverse transform
        for xi, pi in zip(x, p):
            calc_pi = t1.to_model(xi)
            self.assertAlmostEqual(calc_pi, pi)
        self.assertTrue(np.allclose(t4.to_model(x), p))

        # Test n_parameters
        self.assertEqual(t1.n_parameters(), 1)
        self.assertEqual(t4.n_parameters(), 4)

        # Test Jacobian
        self.assertTrue(np.allclose(t4.jacobian(x), j))

        # Test log-Jacobian determinant
        self.assertEqual(t4.log_jacobian_det(x), log_j_det)

    def test_optimisation(self):
        # Test passing to optimisation
        pass

        # Test sigma0 inputs?

        # Test return solution shape

        # Test return solution transform

    def test_sampling(self):
        # Test passing to sampling
        pass

        # Test sigma0 inputs?

        # Test return chain shape

        # Test return chain transform


if __name__ == '__main__':
    unittest.main()
