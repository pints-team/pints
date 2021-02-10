#!/usr/bin/env python3
#
# Tests if the Hodgkin-Huxley IK (toy) model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
import pints
import pints.toy


class TestHodgkinHuxleyIKModel(unittest.TestCase):
    """
    Tests if the Hodgkin-Huxley IK (toy) model works.
    """

    def test_creation(self):

        # Test simple creation
        pints.toy.HodgkinHuxleyIKModel()

        # Test initial condition out of bounds
        self.assertRaises(ValueError, pints.toy.HodgkinHuxleyIKModel, 0)
        self.assertRaises(ValueError, pints.toy.HodgkinHuxleyIKModel, 1)

    def test_suggestions(self):

        model = pints.toy.HodgkinHuxleyIKModel()

        # Parameters
        p0 = model.suggested_parameters()
        self.assertEqual(len(p0), model.n_parameters())

        # Times
        times = model.suggested_times()
        self.assertTrue(np.all(times[1:] > times[:-1]))

        # Maximum duration
        self.assertEqual(model.suggested_duration(), 1200)

    def test_simulation(self):

        model = pints.toy.HodgkinHuxleyIKModel()
        p0 = model.suggested_parameters()
        times = model.suggested_times()

        # Run
        values = model.simulate(p0, times)
        self.assertEqual(len(times), len(values))

        # Test against reference values from a simulation with Myokit

        # Test at time 0ms
        i = 0
        self.assertEqual(times[i], 0)
        self.assertAlmostEqual(values[i], 3.790799999999997)

        # Test at time 0.25ms
        i = 1
        self.assertEqual(times[i], 0.25)
        self.assertAlmostEqual(values[i], 3.83029, places=2)

        # Test during a step
        i = 390
        self.assertEqual(times[i], 97.5)
        self.assertAlmostEqual(values[i], 15.9405, places=2)

        # Test towards end of simulation (in step again)
        i = 4790
        self.assertEqual(times[i], 1197.5)
        self.assertAlmostEqual(values[i], 3862.8, places=0)

        # Test time out of bounds
        self.assertRaises(ValueError, model.simulate, p0, [-1, 0, 1])

    def test_fold_method(self):

        # Tests the method that 'folds' the data for a plot similar to that in
        # the original paper

        model = pints.toy.HodgkinHuxleyIKModel()
        p0 = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(p0, times)

        folded = model.fold(times, values)
        self.assertEqual(len(folded), 12)


if __name__ == '__main__':
    unittest.main()
