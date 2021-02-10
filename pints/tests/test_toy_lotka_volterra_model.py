#!/usr/bin/env python3
#
# Tests if the Lotka-Volterra toy model runs.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy


class TestLotkaVolterraModel(unittest.TestCase):
    """
    Tests if the Lotka-Volterra toy model runs.
    """

    def test_run(self):
        model = pints.toy.LotkaVolterraModel()
        self.assertEqual(model.n_parameters(), 4)
        self.assertEqual(model.n_outputs(), 2)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        values = model.simulate(parameters, times)
        self.assertEqual(values.shape, (len(times), 2))
        self.assertTrue(np.all(values > 0))

        # Test setting and getting init cond.
        self.assertFalse(np.all(model.initial_conditions() == [10, 10]))
        model.set_initial_conditions([10, 10])
        self.assertTrue(np.all(model.initial_conditions() == [10, 10]))

        # Initial conditions cannot be negative
        model = pints.toy.LotkaVolterraModel([0, 0])
        self.assertRaises(ValueError, pints.toy.LotkaVolterraModel, [-1, 0])
        self.assertRaises(ValueError, pints.toy.LotkaVolterraModel, [0, -1])
        self.assertRaises(ValueError, pints.toy.LotkaVolterraModel, [-1, -1])

    def test_values(self):
        # value-based tests of solution
        x0 = 3
        y0 = 5
        model = pints.toy.LotkaVolterraModel([x0, y0])
        parameters = [1, 2, 2, 0.5]
        times = np.linspace(0, 5, 101)
        values = model.simulate(parameters, times)
        self.assertEqual(values[0, 0], x0)
        self.assertEqual(values[0, 1], y0)
        self.assertAlmostEqual(values[1, 0], 1.929494, places=6)
        self.assertAlmostEqual(values[1, 1], 4.806542, places=6)
        self.assertAlmostEqual(values[100, 0], 1.277762, places=6)
        self.assertAlmostEqual(values[100, 1], 0.000529, places=6)

    def test_sensitivities(self):
        # tests sensitivities against standards
        model = pints.toy.LotkaVolterraModel()
        vals = model.suggested_values()
        self.assertEqual(vals.shape[0], 21)
        sols, sens = model.simulateS1([0.43, 0.2, 0.9, 0.28], [5, 10])
        self.assertAlmostEqual(sens[0, 0, 0], -4.889418, 5)
        self.assertAlmostEqual(sens[1, 1, 3], -0.975323, 5)


if __name__ == '__main__':
    unittest.main()
