#!/usr/bin/env python3
#
# Tests if the Fitzhugh-Nagumo toy model runs.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import pints.toy
import numpy as np
from scipy.interpolate import interp1d


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
        model = pints.toy.FitzhughNagumoModel([2, 3])
        parameters = [0.2, 0.7, 2.8]
        times_finer = np.linspace(0, 20, 500)
        sols, sens = model.simulateS1(parameters, times_finer)
        f = interp1d(times_finer, sens[:, 0][:, 2])
        self.assertTrue(np.abs(f([7])[0] - 5.0137868240051535) <= 0.01)
        f = interp1d(times_finer, sens[:, 1][:, 1])
        self.assertTrue(np.abs(f([12])[0] - 0.8288255034841188) <= 0.01)


if __name__ == '__main__':
    unittest.main()
