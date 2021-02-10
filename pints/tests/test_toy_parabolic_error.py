#!/usr/bin/env python3
#
# Tests the parabolic error toy error measure.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest

import numpy as np

import pints
import pints.toy


class TestParabolicError(unittest.TestCase):
    """
    Tests the parabolic error toy error measure.
    """
    def test_parabolic_error(self):

        # Test a 2d case
        f = pints.toy.ParabolicError()
        self.assertEqual(f.n_parameters(), 2)
        self.assertEqual(list(f.optimum()), [0, 0])
        self.assertEqual(f([0, 0]), 0)
        self.assertTrue(f([0.1, 0.1]) > 0)

        # Test a 3d case
        f = pints.toy.ParabolicError([1, 1, 1])
        self.assertEqual(f.n_parameters(), 3)
        self.assertEqual(list(f.optimum()), [1, 1, 1])
        self.assertEqual(f([1, 1, 1]), 0)
        self.assertTrue(f([1.1, 1.1, 1.1]) > 0)

        # Test sensitivities
        x = [1, 1, 1]
        fx, dfx = f.evaluateS1(x)
        self.assertEqual(fx, f(x))
        self.assertEqual(dfx.shape, (3, ))
        self.assertTrue(np.all(dfx == [0, 0, 0]))

        x = [1.1, 1, 0.8]
        fx, dfx = f.evaluateS1(x)
        self.assertEqual(fx, f(x))
        self.assertEqual(dfx.shape, (3, ))
        self.assertAlmostEqual(dfx[0], 0.2)
        self.assertAlmostEqual(dfx[1], 0)
        self.assertAlmostEqual(dfx[2], -0.4)


if __name__ == '__main__':
    unittest.main()
