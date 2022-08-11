#!/usr/bin/env python3
#
# Tests if the degradation (toy) model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest

import numpy as np

from pints.toy.stochastic import DegradationModel


class TestDegradationModel(unittest.TestCase):
    """
    Tests if the degradation (toy) model works.
    """
    def test_n_parameters(self):
        x_0 = 20
        model = DegradationModel(x_0)
        self.assertEqual(model.n_parameters(), 1)

    def test_simulation_length(self):
        x_0 = 20
        model = DegradationModel(x_0)
        times = np.linspace(0, 1, 100)
        k = [0.1]
        values = model.simulate(k, times)
        self.assertEqual(len(values), 100)

    def test_propensities(self):
        x_0 = 20
        k = [0.1]
        model = DegradationModel(x_0)
        self.assertTrue(
            np.allclose(
                model._propensities([x_0], k),
                np.array([2.0])))

    def test_suggested(self):
        model = DegradationModel(20)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        self.assertTrue(len(times) == 101)
        self.assertTrue(parameters > 0)

    def test_mean_variance(self):
        # test mean
        model = DegradationModel(10)
        v_mean = model.mean([1], [5, 10])
        self.assertEqual(v_mean[0], 10 * np.exp(-5))
        self.assertEqual(v_mean[1], 10 * np.exp(-10))

        model = DegradationModel(20)
        v_mean = model.mean([5], [7.2])
        self.assertEqual(v_mean[0], 20 * np.exp(-7.2 * 5))

        # test variance
        model = DegradationModel(10)
        v_var = model.variance([1], [5, 10])
        self.assertEqual(v_var[0], 10 * (np.exp(5) - 1.0) / np.exp(10))
        self.assertAlmostEqual(v_var[1], 10 * (np.exp(10) - 1.0) / np.exp(20))

        model = DegradationModel(20)
        v_var = model.variance([2.0], [2.0])
        self.assertAlmostEqual(v_var[0], 20 * (np.exp(4) - 1.0) / np.exp(8))

    def test_errors(self):
        model = DegradationModel(20)

        # parameters, times cannot be negative
        times = np.linspace(0, 100, 101)
        parameters = [-0.1]
        self.assertRaisesRegex(ValueError, 'constant must be positive',
                               model.mean, parameters, times)
        self.assertRaisesRegex(ValueError, 'constant must be positive',
                               model.variance, parameters, times)

        times_2 = np.linspace(-10, 10, 21)
        parameters_2 = [0.1]
        self.assertRaisesRegex(ValueError, 'Negative times',
                               model.mean, parameters_2, times_2)
        self.assertRaisesRegex(ValueError, 'Negative times',
                               model.variance, parameters_2, times_2)

        # this model should have 1 parameter
        parameters_3 = [0.1, 1]
        self.assertRaisesRegex(ValueError, 'only 1 parameter',
                               model.mean, parameters_3, times)
        self.assertRaisesRegex(ValueError, 'only 1 parameter',
                               model.variance, parameters_3, times)


if __name__ == '__main__':
    unittest.main()
