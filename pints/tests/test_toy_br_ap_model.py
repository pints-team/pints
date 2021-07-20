#!/usr/bin/env python3
#
# Tests if the Beeler-Reuter AP (toy) model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
import pints
import pints.toy


class TestActionPotentialModel(unittest.TestCase):
    """
    Tests if the Beeler-Reuter AP (toy) model works.
    """

    def test_creation(self):

        # Test creation
        model = pints.toy.ActionPotentialModel()

        # Test setting and getting init cond.
        self.assertFalse(np.all(model.initial_conditions() == [-80, 1e-5]))
        model.set_initial_conditions([-80, 1e-5])
        self.assertTrue(np.all(model.initial_conditions() == [-80, 1e-5]))

        # Initial conditions cannot be negative
        self.assertRaisesRegex(
            ValueError, 'cannot be negative',
            pints.toy.ActionPotentialModel, [-80, -1])

    def test_suggestions(self):

        model = pints.toy.ActionPotentialModel()
        p0 = model.suggested_parameters()
        self.assertEqual(len(p0), model.n_parameters())

        times = model.suggested_times()
        self.assertTrue(np.all(times[1:] > times[:-1]))

    def test_simulation(self):

        model = pints.toy.ActionPotentialModel()
        times = model.suggested_times()
        p0 = model.suggested_parameters()

        # Test simulating all eight states
        states = model.simulate_all_states(p0, times)
        self.assertEqual(len(states.shape), 2)
        self.assertEqual(states.shape[0], len(times))
        self.assertEqual(states.shape[1], 8)

        # Test initial state
        x0 = np.array([-84.622, 2e-7, 0.01, 0.99, 0.98, 0.003, 0.99, 0.0004])
        self.assertTrue(np.all(states[0] == x0))

        # Test state during AP (at 100ms, with stimulus applied at t=0ms)
        # Reference values taken from a Myokit simulation with the same model
        i100 = 200
        self.assertEqual(times[i100], 100)
        x100 = [
            1.09411249975881226e+01,
            6.14592181872724475e-06,
            9.93428572073381311e-01,
            2.17659582339377205e-11,
            -3.78573517405393772e-11,
            9.72515482289432853e-01,
            7.67322448537725688e-01,
            2.34776564989968184e-01,
        ]
        self.assertAlmostEqual(states[i100][0], x100[0], places=2)
        self.assertAlmostEqual(states[i100][1], x100[1], places=2)
        self.assertAlmostEqual(states[i100][2], x100[2], places=2)
        self.assertAlmostEqual(states[i100][3], x100[3], places=2)
        self.assertAlmostEqual(states[i100][4], x100[4], places=2)
        self.assertAlmostEqual(states[i100][5], x100[5], places=2)
        self.assertAlmostEqual(states[i100][6], x100[6], places=2)
        self.assertAlmostEqual(states[i100][7], x100[7], places=2)

        # Test simulation outputting only observables
        partial = model.simulate(p0, times)
        self.assertEqual(len(times), len(partial))
        self.assertEqual(partial.shape[1], model.n_outputs())

        partial = np.array(partial)
        states = np.array(states)
        self.assertTrue(np.all(partial[:, 0] == states[:, 0]))
        self.assertTrue(np.all(partial[:, 1] == states[:, 1]))


if __name__ == '__main__':
    unittest.main()
