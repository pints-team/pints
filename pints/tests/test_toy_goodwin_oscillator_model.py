#!/usr/bin/env python3
#
# Tests if the goodwin oscillator (toy) model runs.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import pints.toy
import numpy as np


class TestGoodwinOscillatorModel(unittest.TestCase):
    """
    Tests if the goodwin oscillator (toy) model runs.
    """

    def test_run(self):
        model = pints.toy.GoodwinOscillatorModel()
        self.assertEqual(model.n_parameters(), 5)
        self.assertEqual(model.n_outputs(), 3)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        values = model.simulate(parameters, times)
        self.assertEqual(values.shape, (len(times), 3))

    def test_values(self):
        # value-based tests of Goodwin-oscillator
        model = pints.toy.GoodwinOscillatorModel()
        parameters = [3, 2.5, 0.15, 0.1, 0.12]
        times = np.linspace(0, 10, 101)
        values = model.simulate(parameters, times)
        self.assertEqual(values[0, 0], 0.0054)
        self.assertEqual(values[0, 1], 0.053)
        self.assertEqual(values[0, 2], 1.93)
        self.assertAlmostEqual(values[100, 0], 0.0061854, places=6)
        self.assertAlmostEqual(values[100, 1], 0.1779547, places=6)
        self.assertAlmostEqual(values[100, 2], 2.6074527, places=6)

    def test_sensitivity(self):
        # tests construction of matrices for sensitivity calculation and
        # compares sensitivities vs standards
        model = pints.toy.GoodwinOscillatorModel()
        parameters = [3, 2.5, 0.15, 0.1, 0.12]
        k2, k3, m1, m2, m3 = parameters
        time = np.linspace(0, 10, 101)
        state = [0.01, 0.1, 2]
        x, y, z = state
        ret = model.jacobian(state, 0.0, parameters)
        self.assertEqual(ret[0, 0], -m1)
        self.assertEqual(ret[0, 1], 0)
        self.assertEqual(ret[0, 2], -10 * z**9 / ((1 + z**10)**2))
        self.assertEqual(ret[1, 0], k2)
        self.assertEqual(ret[1, 1], -m2)
        self.assertEqual(ret[1, 2], 0)
        self.assertEqual(ret[2, 0], 0)
        self.assertEqual(ret[2, 1], k3)
        self.assertEqual(ret[2, 2], -m3)
        values = model.simulate(parameters, time)
        values1, dvals = model.simulateS1(parameters, time)
        self.assertTrue(np.array_equal(values.shape, values1.shape))
        self.assertTrue(np.array_equal(
            dvals.shape,
            np.array([len(time), model.n_outputs(), model.n_parameters()])))
        # note -- haven't coded this up separately to check but compare against
        # current output in case of future changes
        self.assertTrue(np.abs(-2.20655371e-05 - dvals[10, 0, 0]) < 10**(-5))
        for i in range(len(time)):
            for j in range(3):
                self.assertTrue(
                    np.abs(values[i, j] - values1[i, j]) < 10**(-3))

        model = pints.toy.GoodwinOscillatorModel()
        parameters = model.suggested_parameters()
        sols, sens = model.simulateS1(parameters, [35, 80])
        self.assertAlmostEqual(sens[0, 0, 2], 0.07705, 4)
        self.assertAlmostEqual(sens[1, 1, 3], 3.35704, 4)


if __name__ == '__main__':
    unittest.main()
