#!/usr/bin/env python3
#
# Tests if the stochastic degradation (toy) model works.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np
import pints
import pints.toy


class TestStochasticDegradation(unittest.TestCase):
    """
    Tests if the stochastic degradation (toy) model works.
    """

    def test_start_with_zero(self):
        # Test the special case where the initial concentration is zero
        model = pints.toy.StochasticDegradationModel(0)
        times = [0, 1, 2, 100, 1000]
        parameters = 0.1
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertTrue(np.all(values == np.zeros(5)))

    def test_start_with_twenty(self):
        # Run small simulation
        model = pints.toy.StochasticDegradationModel(20)
        times = [0, 1, 2, 100, 1000]
        parameters = 0.1
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 20)
        self.assertEqual(values[-1], 0)

    def test_suggested(self):
        model = pints.toy.StochasticDegradationModel(20)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        self.assertTrue(np.all(times == np.linspace(0, 100, 101)))
        self.assertEqual(parameters, 0.1)
        values = model.simulate(parameters, times)
        self.assertEqual(values[0], 20)
        self.assertEqual(values[-1], 0)

    def test_errors(self):
        model = pints.toy.StochasticDegradationModel(20)
        times = np.linspace(0, 100, 101)
        parameters = -0.1
        self.assertRaises(ValueError, model.simulate, parameters, times)
        times_2 = np.linspace(-10, 10, 21)
        parameters_2 = 0.1
        self.assertRaises(ValueError, model.simulate, parameters_2, times_2)

        # Initial value can't be negative
        self.assertRaises(ValueError, pints.toy.StochasticDegradationModel, -1)


if __name__ == '__main__':
    unittest.main()