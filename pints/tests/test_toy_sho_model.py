#!/usr/bin/env python3
#
# Tests if the simple harmonic oscillator model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
import pints
import pints.toy


class TestSimpleHarmonicOscillator(unittest.TestCase):
    """
    Tests if the simple harmonic oscillator model works.
    """

    def test_values_and_sensitivities_underdamped(self):
        # test values and sensitivities for some parameter values
        model = pints.toy.SimpleHarmonicOscillatorModel()
        times = [0, 1, 2, 10]
        parameters = [2.5, -3.5, 0.3]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 2.5)
        self.assertAlmostEqual(values[1], -1.0894215, places=6)
        self.assertAlmostEqual(values[2], -2.8830890, places=6)
        self.assertAlmostEqual(values[3], -0.1849098, places=6)

        values1, sensitivities = model.simulateS1(parameters, times)
        self.assertTrue(np.all(values == values))
        self.assertEqual(sensitivities.shape[0], len(times))
        self.assertEqual(sensitivities.shape[1], 3)
        self.assertEqual(sensitivities[0, 0], 1)
        self.assertEqual(sensitivities[0, 1], 0)
        self.assertEqual(sensitivities[0, 2], 0)
        self.assertAlmostEqual(sensitivities[1, 0], 0.5822839, places=6)
        self.assertAlmostEqual(sensitivities[1, 1], 0.7271804, places=6)
        self.assertAlmostEqual(sensitivities[1, 2], 1.5291374, places=6)

    def test_values_and_sensitivities_criticaldamp(self):
        # test values and sensitivities for critical damping
        model = pints.toy.SimpleHarmonicOscillatorModel()
        times = [0, 0.5, 1, 1.5]
        parameters = [1, 2.3, 2]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 1)
        self.assertAlmostEqual(values[1], 1.6073062, places=6)
        self.assertAlmostEqual(values[2], 1.5818816, places=6)
        self.assertAlmostEqual(values[3], 1.3276245, places=6)

        values1, sensitivities = model.simulateS1(parameters, times)
        self.assertTrue(np.all(values == values))
        self.assertEqual(sensitivities.shape[0], len(times))
        self.assertEqual(sensitivities.shape[1], 3)
        self.assertEqual(sensitivities[0, 0], 1)
        self.assertEqual(sensitivities[0, 1], 0)
        self.assertTrue(np.all(sensitivities[:, 2] == np.zeros(len(times))))
        self.assertAlmostEqual(sensitivities[1, 0], 0.9097959, places=6)
        self.assertAlmostEqual(sensitivities[1, 1], 0.3032653, places=6)

    def test_suggested(self):
        # tests suggested values
        model = pints.toy.SimpleHarmonicOscillatorModel()
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        self.assertTrue(np.all(np.linspace(0, 50, 100) == times))
        self.assertTrue(np.all([1, 0, 0.15] == parameters))

    def test_n_parameters(self):
        model = pints.toy.SimpleHarmonicOscillatorModel()
        self.assertEqual(model.n_parameters(), 3)

    def test_errors(self):
        # tests errors
        model = pints.toy.SimpleHarmonicOscillatorModel()
        times = [0, 1, 2, 10000]
        parameters = [1, 1, 0.6]
        times[1] = -1
        self.assertRaises(ValueError, model.simulate, parameters, times)


if __name__ == '__main__':
    unittest.main()
