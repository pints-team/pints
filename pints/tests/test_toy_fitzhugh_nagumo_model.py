#!/usr/bin/env python3
#
# Tests if the Fitzhugh-Nagumo toy model runs.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import pints.toy
import numpy as np


class TestFitzhughNagumoModel(unittest.TestCase):
    """
    Tests if the Fitzhugh-Nagumo toy model runs.
    """

    def test_run(self):

        # Test basic properties
        model = pints.toy.FitzhughNagumoModel()
        self.assertEqual(model.n_parameters(), 3)
        self.assertEqual(model.n_outputs(), 2)

        # Test simulation
        x = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (len(times), 2))

        # Simulation with sensitivities
        values, dvalues_dp = model.simulateS1(x, times)
        self.assertEqual(values.shape, (len(times), 2))
        self.assertEqual(dvalues_dp.shape, (len(times), 2, 3))

        # Test alternative starting position
        model = pints.toy.FitzhughNagumoModel([0.1, 0.1])
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (len(times), 2))

        # Times can't be negative
        times = [-1, 2, 3, 4]
        self.assertRaises(ValueError, model.simulate, x, times)

        # Initial value must have size 2
        pints.toy.FitzhughNagumoModel([1, 1])
        self.assertRaises(ValueError, pints.toy.FitzhughNagumoModel, [1])

    def test_values(self):
        # value-based tests of Fitzhugh-Nagumo model
        parameters = [0.2, 0.4, 2.5]
        y0 = [-2, 1.5]
        times = np.linspace(0, 20, 201)
        model = pints.toy.FitzhughNagumoModel(y0)
        values = model.simulate(parameters, times)
        self.assertAlmostEqual(values[200, 0], 1.675726, places=6)
        self.assertAlmostEqual(values[200, 1], -0.226142, places=6)

    def test_sensitivities(self):
        # compares sensitivities against standards
        model = pints.toy.FitzhughNagumoModel([2, 3])
        parameters = [0.2, 0.7, 2.8]

        # Test with initial point t=0 included in range
        sols, sens = model.simulateS1(parameters, [0, 7, 12])
        self.assertAlmostEqual(sens[1, 0, 2], 5.01378, 5)
        self.assertAlmostEqual(sens[2, 1, 1], 0.82883, 4)

        # Test without initial point in range
        sols, sens = model.simulateS1(parameters, [7, 12])
        self.assertAlmostEqual(sens[0, 0, 2], 5.01378, 5)
        self.assertAlmostEqual(sens[1, 1, 1], 0.82883, 4)

        # Test without any points in range
        sols, sens = model.simulateS1(parameters, [])
        self.assertEqual(sols.shape, (0, 2))
        self.assertEqual(sens.shape, (0, 2, 3))


if __name__ == '__main__':
    unittest.main()
