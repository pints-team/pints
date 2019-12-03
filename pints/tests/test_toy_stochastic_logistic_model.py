#!/usr/bin/env python3
#
# Tests if the stochastic logistic growth (toy) model works.
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
from pints.toy import StochasticLogisticModel


class TestStochasticLogistic(unittest.TestCase):
    """
    Tests if the stochastic logistic growth (toy) model works.
    """
    def test_start_with_zero(self):
        # Test the special case where the initial population count is zero
        model = StochasticLogisticModel(0)
        times = [0, 1, 2, 100, 1000]
        parameters = [0.1, 50]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertTrue(np.all(values == np.zeros(5)))

    def test_start_with_one(self):
        # Run small simulation
        model = pints.toy.StochasticLogisticModel(1)
        times = [0, 1, 2, 100, 1000]
        parameters = [0.1, 50]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 1)
        self.assertEqual(values[-1], 50)
        self.assertTrue(np.all(values[1:] >= values[:-1]))

    def test_suggested(self):
        model = pints.toy.StochasticLogisticModel(1)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        self.assertTrue(len(times) == 101)
        self.assertTrue(np.all(parameters > 0))

    def test_simulate(self):
        times = np.linspace(0, 100, 101)
        model = StochasticLogisticModel(1)
        params = [0.1, 50]
        time, values = model.simulate_raw([0.1, 50])
        values = model.interpolate_values(time, values, times, params)
        self.assertTrue(len(time), len(values))

        # Test output of Gillespie algorithm
        self.assertTrue(np.all(values ==
                               np.array(range(1, 51))))

        # Check simulate function returns expected values
        self.assertTrue(np.all(values[np.where(times < time[1])] == 1))

        # Check interpolation function works as expected
        temp_time = np.array([np.random.uniform(time[0], time[1])])
        self.assertTrue(model.interpolate_values(time, values, temp_time,
                                                 params)[0] == 1)
        temp_time = np.array([np.random.uniform(time[1], time[2])])
        self.assertTrue(model.interpolate_values(time, values, temp_time,
                                                 params)[0] == 2)

    def test_mean_variance(self):
        # test mean
        model = pints.toy.StochasticLogisticModel(1)
        v_mean = model.mean([1, 10], [5, 10])
        self.assertEqual(v_mean[0], 10 / (1 + 9 * np.exp(-5)))
        self.assertEqual(v_mean[1], 10 / (1 + 9 * np.exp(-10)))

    def test_errors(self):
        model = pints.toy.StochasticLogisticModel(1)
        # parameters, times cannot be negative
        times = np.linspace(0, 100, 101)
        parameters = [-0.1, 50]
        self.assertRaises(ValueError, model.simulate, parameters, times)
        self.assertRaises(ValueError, model.mean, parameters, times)

        parameters = [0.1, -50]
        self.assertRaises(ValueError, model.simulate, parameters, times)
        self.assertRaises(ValueError, model.mean, parameters, times)

        times_2 = np.linspace(-10, 10, 21)
        parameters_2 = [0.1, 50]
        self.assertRaises(ValueError, model.simulate, parameters_2, times_2)
        self.assertRaises(ValueError, model.mean, parameters_2, times_2)

        # this model should have 2 parameters
        parameters_3 = [0.1]
        self.assertRaises(ValueError, model.simulate, parameters_3, times)
        self.assertRaises(ValueError, model.mean, parameters_3, times)

        # model variance isn't implemented so we should throw a helpful error
        parameters_4 = [0.1, 50]
        self.assertRaises(NotImplementedError, model.variance, parameters_4, times)

        # Initial value can't be negative
        self.assertRaises(ValueError, pints.toy.StochasticLogisticModel, -1)


if __name__ == '__main__':
    unittest.main()
