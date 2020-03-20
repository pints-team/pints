#!/usr/bin/env python3
#
# Tests if the SIR toy model runs.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import pints.toy
import numpy as np


class TestSIRModel(unittest.TestCase):
    """
    Tests if the SIR toy model runs.
    """

    def test_run(self):

        # Test basic properties
        model = pints.toy.SIRModel()
        self.assertEqual(model.n_parameters(), 3)
        self.assertEqual(model.n_outputs(), 2)

        # Test simulation
        x = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (len(times), model.n_outputs()))

        # Test suggested values
        v = model.suggested_values()
        self.assertEqual(v.shape, (len(times), model.n_outputs()))

        # Test setting intial conditions
        model = pints.toy.SIRModel([1, 1, 1])

        # Must have 3 init cond.
        self.assertRaises(
            ValueError, pints.toy.SIRModel, [1, 1])

        # Populations are never negative
        self.assertRaises(
            ValueError, pints.toy.SIRModel, [1, 1, -1])

    def test_values(self):
        # value-based tests of model solution
        S0 = 100
        parameters = [0.05, 0.4, S0]
        times = np.linspace(0, 10, 101)
        model = pints.toy.SIRModel([S0, 10, 1])
        values = model.simulate(parameters, times)
        self.assertEqual(values[0, 0], 10)
        self.assertEqual(values[0, 1], 1)
        self.assertAlmostEqual(values[1, 0], 15.61537, places=5)
        self.assertAlmostEqual(values[1, 1], 1.50528, places=5)
        self.assertAlmostEqual(values[100, 0], 2.45739, places=5)
        self.assertAlmostEqual(values[100, 1], 108.542466, places=5)


if __name__ == '__main__':
    unittest.main()
