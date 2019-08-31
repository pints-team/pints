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
from pints.toy import StochasticDegradationModel


class TestStochasticDegradation(unittest.TestCase):
    """
    Tests if the stochastic degradation (toy) model works.
    """
    def test_start_with_zero(self):
        # Test the special case where the initial molecule count is zero
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
        self.assertTrue(np.all(values[1:] <= values[:-1]))

    def test_suggested(self):
        model = pints.toy.StochasticDegradationModel(20)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        self.assertTrue(len(times) == 101)
        self.assertTrue(parameters > 0)

    def test_simulate(self):
        times = np.linspace(0, 100, 101)
        model = StochasticDegradationModel(20)
        time, mol_count = model.simulate_raw([0.1])
        values = model.interpolate_mol_counts(time, mol_count, times)
        self.assertTrue(len(time), len(mol_count))
        # Test output of Gillespie algorithm
        self.assertTrue(np.all(mol_count ==
                               np.array(range(20, -1, -1))))

        # Check simulate function returns expected values
        self.assertTrue(np.all(values[np.where(times < time[1])] == 20))

        # Check interpolation function works as expected
        temp_time = np.array([np.random.uniform(time[0], time[1])])
        self.assertTrue(model.interpolate_mol_counts(time, mol_count,
                                                     temp_time)[0] == 20)
        temp_time = np.array([np.random.uniform(time[1], time[2])])
        self.assertTrue(model.interpolate_mol_counts(time, mol_count,
                                                     temp_time)[0] == 19)

    def test_mean_variance(self):
        # test mean
        model = pints.toy.StochasticDegradationModel(10)
        v_mean = model.mean([1], [5, 10])
        self.assertEqual(v_mean[0], 10 * np.exp(-5))
        self.assertEqual(v_mean[1], 10 * np.exp(-10))

        model = pints.toy.StochasticDegradationModel(20)
        v_mean = model.mean([5], [7.2])
        self.assertEqual(v_mean[0], 20 * np.exp(-7.2 * 5))

        # test variance
        model = pints.toy.StochasticDegradationModel(10)
        v_var = model.variance([1], [5, 10])
        self.assertEqual(v_var[0], 10 * (np.exp(5) - 1.0) / np.exp(10))
        self.assertAlmostEqual(v_var[1], 10 * (np.exp(10) - 1.0) / np.exp(20))

        model = pints.toy.StochasticDegradationModel(20)
        v_var = model.variance([2.0], [2.0])
        self.assertAlmostEqual(v_var[0], 20 * (np.exp(4) - 1.0) / np.exp(8))

    def test_errors(self):
        model = pints.toy.StochasticDegradationModel(20)
        # parameters, times cannot be negative
        times = np.linspace(0, 100, 101)
        parameters = [-0.1]
        self.assertRaises(ValueError, model.simulate, parameters, times)
        self.assertRaises(ValueError, model.mean, parameters, times)
        self.assertRaises(ValueError, model.variance, parameters, times)

        times_2 = np.linspace(-10, 10, 21)
        parameters_2 = [0.1]
        self.assertRaises(ValueError, model.simulate, parameters_2, times_2)
        self.assertRaises(ValueError, model.mean, parameters_2, times_2)
        self.assertRaises(ValueError, model.variance, parameters_2, times_2)

        # this model should have 1 parameter
        parameters_3 = [0.1, 1]
        self.assertRaises(ValueError, model.simulate, parameters_3, times)
        self.assertRaises(ValueError, model.mean, parameters_3, times)
        self.assertRaises(ValueError, model.variance, parameters_3, times)

        # Initial value can't be negative
        self.assertRaises(ValueError, pints.toy.StochasticDegradationModel, -1)


if __name__ == '__main__':
    unittest.main()
