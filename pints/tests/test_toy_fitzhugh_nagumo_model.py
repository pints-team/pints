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
        self.assertTrue(np.abs(values[200, 0] - 1.6757257661845268) <
                        0.000001)
        self.assertTrue(np.abs(values[200, 1] - -0.22614224385943715) <
                        0.000001)


if __name__ == '__main__':
    unittest.main()
