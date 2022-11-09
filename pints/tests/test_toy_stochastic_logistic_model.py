#!/usr/bin/env python3
#
# Tests if the stochastic logistic growth (toy) model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
import pints
import pints.toy
import pints.toy.stochastic


class TestStochasticLogisticModel(unittest.TestCase):
    """
    Tests if the stochastic logistic growth (toy) model works.
    """
    def test_start_with_zero(self):
        # Test the special case where the initial population count is zero

        # Set seed for random generator
        np.random.seed(1)

        model = pints.toy.stochastic.LogisticModel(0)
        times = [0, 1, 2, 100, 1000]
        parameters = [0.1, 50]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertTrue(np.all(values == np.zeros(5)))

    def test_start_with_one(self):
        # Run a small simulation and check it runs properly

        # Set seed for random generator
        np.random.seed(1)

        model = pints.toy.stochastic.LogisticModel(1)
        times = [0, 1, 2, 100, 1000]
        parameters = [0.1, 50]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 1)
        self.assertEqual(values[-1], 50)
        self.assertTrue(np.all(values[1:] >= values[:-1]))

    def test_suggested(self):
        # Check suggested values
        model = pints.toy.stochastic.LogisticModel(1)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        self.assertTrue(len(times) == 101)
        self.assertTrue(np.all(parameters > 0))

    def test_simulate(self):
        # Check each step in the simulation process
        np.random.seed(1)
        model = pints.toy.stochastic.LogisticModel(1)
        times = np.linspace(0, 100, 101)
        time, raw_values = model.simulate_raw([0.1, 50], 100)
        values = model.interpolate_mol_counts(time, raw_values, times)
        self.assertTrue(len(time), len(raw_values))

        # Test output of Gillespie algorithm
        raw_values = np.concatenate(raw_values)
        self.assertTrue(np.all(raw_values == np.array(range(1, 28))))

        # Check simulate function returns expected values
        self.assertTrue(np.all(values[np.where(times < time[1])] == 1))

        # Check interpolation function works as expected
        temp_time = np.array([np.random.uniform(time[0], time[1])])
        self.assertTrue(model.interpolate_mol_counts(time, raw_values,
                                                     temp_time)[0] == 1)
        temp_time = np.array([np.random.uniform(time[1], time[2])])
        self.assertTrue(model.interpolate_mol_counts(time, raw_values,
                                                     temp_time)[0] == 2)

        # Check parameters, times cannot be negative
        parameters_0 = [-0.1, 50]
        self.assertRaises(ValueError, model.mean, parameters_0, times)

        parameters_1 = [0.1, -50]
        self.assertRaises(ValueError, model.mean, parameters_1, times)

        times_2 = np.linspace(-10, 10, 21)
        parameters_2 = [0.1, 50]
        self.assertRaises(ValueError, model.simulate, parameters_2, times_2)
        self.assertRaises(ValueError, model.mean, parameters_2, times_2)

        # Check this model takes 2 parameters
        parameters_3 = [0.1]
        self.assertRaises(ValueError, model.simulate, parameters_3, times)
        self.assertRaises(ValueError, model.mean, parameters_3, times)

        # Check initial value cannot be negative
        self.assertRaises(ValueError, pints.toy.stochastic.LogisticModel, -1)

    def test_mean(self):
        # Check the mean is what we expected
        model = pints.toy.stochastic.LogisticModel(1)
        v_mean = model.mean([1, 10], [5, 10])
        self.assertEqual(v_mean[0], 10 / (1 + 9 * np.exp(-5)))
        self.assertEqual(v_mean[1], 10 / (1 + 9 * np.exp(-10)))


if __name__ == '__main__':
    unittest.main()
