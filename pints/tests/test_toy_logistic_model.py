#!/usr/bin/env python3
#
# Tests if the logistic (toy) model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
import pints
import pints.toy


class TestLogisticModel(unittest.TestCase):
    """
    Tests if the logistic (toy) model works.
    """

    def test_start_with_zero(self):
        # Test the special case where the initial size is zero
        model = pints.toy.LogisticModel(0)
        times = [0, 1, 2, 10000]
        parameters = [1, 5]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertTrue(np.all(values == np.zeros(4)))

        values, sensitivities = model.simulateS1(parameters, times)
        self.assertTrue(np.all(values == np.zeros(4)))
        self.assertTrue(np.all(sensitivities == np.zeros((4, 2))))

    def test_start_with_two(self):
        # Run small simulation
        model = pints.toy.LogisticModel(2)
        times = [0, 1, 2, 10000]
        parameters = [1, 5]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 2)
        self.assertEqual(values[-1], parameters[-1])
        # Run large simulation
        times = np.arange(0, 1000)
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 2)
        self.assertEqual(values[-1], parameters[-1])
        self.assertTrue(np.all(values[1:] >= values[:-1]))

    def test_suggested(self):
        model = pints.toy.LogisticModel(2)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        values = model.simulate(parameters, times)
        self.assertAlmostEqual(
            values[-1], parameters[-1], delta=0.01 * parameters[1])

    def test_negative_k(self):
        model = pints.toy.LogisticModel(2)
        times = [0, 1, 2, 10000]
        parameters = [1, -1]
        values = model.simulate(parameters, times)
        self.assertTrue(np.all(values == np.zeros(4)))
        values, sensitivities = model.simulateS1(parameters, times)
        self.assertTrue(np.all(values == np.zeros(4)))
        self.assertTrue(np.all(sensitivities == np.zeros((4, 2))))

    def test_errors(self):

        model = pints.toy.LogisticModel(2)

        # Times can't be negative
        times = [0, 1, 2, 10000]
        parameters = [1, 1]
        model.simulate(parameters, times)
        times[1] = -1
        self.assertRaises(ValueError, model.simulate, parameters, times)

        # Initial value can't be negative
        self.assertRaises(ValueError, pints.toy.LogisticModel, -1)

    def test_sensitivities(self):
        model = pints.toy.LogisticModel(2)
        times = [0, 1, 2, 10000]
        parameters = [1, 5]
        values, dvdp = model.simulateS1(
            parameters, times)
        self.assertEqual(dvdp[0, 0], 0)
        self.assertEqual(dvdp[-1, 0], 0.0)
        self.assertEqual(dvdp[0, 1], 0)
        self.assertEqual(dvdp[-1, 1], 1.0)

    def test_values(self):
        # value based tests of logistic model
        p0 = 0.5
        model = pints.toy.LogisticModel(p0)
        times = np.linspace(0, 20, 21)
        parameters = [0.1, 12]
        values = model.simulate(parameters, times)
        self.assertEqual(values[0], p0)
        self.assertEqual(values[1], 0.5501745273651227)
        for i in range(len(times)):
            self.assertEqual(values[i], (
                parameters[1] / (1 + (parameters[1] / p0 - 1) *
                                 np.exp(-parameters[0] * times[i]))))

        # test sensitivities
        values, sensitivities = model.simulateS1(parameters, times)
        self.assertEqual(sensitivities[1, 0], 0.5249501931516697)
        self.assertAlmostEqual(sensitivities[1, 1], 0.0002000343976730004,
                               places=6)
        for i in range(len(times)):
            exp = np.exp(-parameters[0] * times[i])
            exp1 = np.exp(parameters[0] * times[i])
            k = parameters[1]
            other = k / p0 - 1
            c = exp1 - 1
            self.assertAlmostEqual(
                exp * k * other * times[i] / (1 + exp * other)**2,
                sensitivities[i, 0])
            self.assertAlmostEqual(
                exp1 * c * p0**2 / (k + c * p0)**2,
                sensitivities[i, 1])


if __name__ == '__main__':
    unittest.main()
