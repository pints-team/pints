#!/usr/bin/env python3
#
# Tests if the Repressilator toy model runs.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import pints.toy
import numpy as np


class TestRepressilatorModel(unittest.TestCase):
    """
    Tests if the Repressilator toy model runs.
    """

    def test_run(self):

        # Test basic properties
        model = pints.toy.RepressilatorModel()
        self.assertEqual(model.n_parameters(), 4)
        self.assertEqual(model.n_outputs(), 3)

        # Test simulation
        x = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (len(times), model.n_outputs()))

        # Test setting intial conditions
        model = pints.toy.RepressilatorModel([1, 1, 1, 1, 1, 1])

        # Must have 6 init cond.
        self.assertRaises(
            ValueError, pints.toy.RepressilatorModel, [1, 1, 1, 1, 1])

        # Concentrations are never negative
        self.assertRaises(
            ValueError, pints.toy.RepressilatorModel, [1, 1, 1, -1, 1, 1])

    def test_values(self):
        # value-based tests of repressilator model
        times = np.linspace(0, 10, 101)
        parameters = [2, 900, 6, 1.5]
        y0 = [5, 3, 1, 2, 3.5, 2.5]
        model = pints.toy.RepressilatorModel(y0)
        values = model.simulate(parameters, times)
        self.assertTrue(np.array_equal(values[0, :], y0[:3]))
        self.assertAlmostEqual(values[1, 0], 18.88838, places=5)
        self.assertAlmostEqual(values[1, 1], 13.77623, places=5)
        self.assertAlmostEqual(values[1, 2], 9.05763, places=5)
        self.assertAlmostEqual(values[100, 0], 14.75099, places=5)
        self.assertAlmostEqual(values[100, 1], 16.55494, places=5)
        self.assertAlmostEqual(values[100, 2], 16.60688, places=5)


if __name__ == '__main__':
    unittest.main()
