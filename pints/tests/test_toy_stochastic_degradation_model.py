#!/usr/bin/env python3
#
# Tests if the stochastic degradation (toy) model works.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
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
        from pints.toy import StochasticDegradationModel
        model = StochasticDegradationModel(0)
        times = [0, 1, 2, 100, 1000]
        parameters = [0.1]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertTrue(np.all(values == np.zeros(5)))

    def test_start_with_twenty(self):
        # Run small simulation
        model = pints.toy.StochasticDegradationModel(20)
        times = [0, 1, 2, 100, 1000]
        parameters = [0.1]
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

    def test_simulate(self):
        model = pints.toy.StochasticDegradationModel(20)
        parameters = [0.1]
        times = np.linspace(0, 100, 101)
        values = model.simulate(parameters, times)

        # Test output of Gillespie algorithm
        self.assertTrue(np.all(model._mol_count ==
                               np.array(range(20, -1, -1))))

        # Test interpolation function
        # Check exact time points from stochastic simulation
        self.assertTrue(np.all(model._interp_func(model._time) ==
                               model._mol_count))

        # Check simulate function returns expected values
        self.assertTrue(np.all(values[np.where(times < model._time[1])] == 20))

        # Check interpolation function works as expected
        self.assertTrue(model._interp_func(np.random.uniform(model._time[0],
                                          model._time[1])) == 20)
        self.assertTrue(model._interp_func(np.random.uniform(model._time[1],
                                          model._time[2])) == 19)

    def test_errors(self):
        model = pints.toy.StochasticDegradationModel(20)
        # parameters, times cannot be negative
        times = np.linspace(0, 100, 101)
        parameters = [-0.1]
        self.assertRaises(ValueError, model.simulate, parameters, times)
        self.assertRaises(ValueError, model.deterministic_mean, parameters,
                          times)
        self.assertRaises(ValueError, model.deterministic_variance, parameters,
                          times)

        times_2 = np.linspace(-10, 10, 21)
        parameters_2 = [0.1]
        self.assertRaises(ValueError, model.simulate, parameters_2, times_2)
        self.assertRaises(ValueError, model.deterministic_mean, parameters_2,
                          times_2)
        self.assertRaises(ValueError, model.deterministic_variance,
                          parameters_2, times_2)

        # this model should have 1 parameter
        parameters_3 = [0.1, 1]
        self.assertRaises(ValueError, model.simulate, parameters_3, times)
        self.assertRaises(ValueError, model.deterministic_mean, parameters_3,
                          times)
        self.assertRaises(ValueError, model.deterministic_variance,
                          parameters_3, times)

        # Initial value can't be negative
        self.assertRaises(ValueError, pints.toy.StochasticDegradationModel, -1)


if __name__ == '__main__':
    unittest.main()
