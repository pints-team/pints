#!/usr/bin/env python3
#
# Tests if the HES1 Michaelis-Menten toy model runs.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy


class TestHes1Model(unittest.TestCase):
    """
    Tests if the HES1 Michaelis-Menten toy model runs.
    """

    def test_run(self):
        model = pints.toy.Hes1Model()
        self.assertEqual(model.n_parameters(), 4)
        self.assertEqual(model.n_outputs(), 1)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        values = model.simulate(parameters, times)
        self.assertEqual(values.shape, (len(times), ))
        self.assertTrue(np.all(values > 0))
        states = model.simulate_all_states(parameters, times)
        self.assertEqual(states.shape, (len(times), 3))
        self.assertTrue(np.all(states > 0))
        suggested_values = model.suggested_values()
        self.assertEqual(suggested_values.shape, (len(times),))
        self.assertTrue(np.all(suggested_values > 0))

        # Test setting and getting init cond.
        self.assertFalse(np.all(model.m0() == 10))
        model.set_m0(10)
        self.assertTrue(np.all(model.m0() == 10))

        # Test setting and getting implicit param.
        self.assertFalse(np.all(model.fixed_parameters() == [10, 10, 10]))
        model.set_fixed_parameters([10, 10, 10])
        self.assertTrue(np.all(model.fixed_parameters() == [10, 10, 10]))

        # Initial conditions cannot be negative
        model = pints.toy.Hes1Model(0)
        self.assertRaises(ValueError, pints.toy.Hes1Model, -1)

        # Implicit parameters cannot be negative
        model = pints.toy.Hes1Model(0, [0, 0, 0])
        self.assertRaises(ValueError, pints.toy.Hes1Model, *(0, [-1, 0, 0]))
        self.assertRaises(ValueError, pints.toy.Hes1Model, *(0, [0, -1, 0]))
        self.assertRaises(ValueError, pints.toy.Hes1Model, *(0, [0, 0, -1]))
        self.assertRaises(ValueError, pints.toy.Hes1Model, *(0, [-1, -1, -1]))

    def test_values(self):
        # value-based tests for Hes1 Michaelis-Menten
        times = np.linspace(0, 10, 101)
        parameters = [3.8, 0.035, 0.15, 7.5]
        iparameters = [4.5, 4.0, 0.04]
        m0 = 7
        model = pints.toy.Hes1Model(m0=m0, fixed_parameters=iparameters)
        values = model.simulate(parameters, times)
        self.assertEqual(values[0], m0)
        self.assertAlmostEqual(values[1], 7.011333, places=6)
        self.assertAlmostEqual(values[100], 5.420750, places=6)

    def test_sensitivities(self):
        # value based tests of jacobian and dfdp
        parameters = [3.8, 0.035, 0.15, 7.5]
        iparameters = [4.5, 4.0, 0.04]
        m0 = 7
        model = pints.toy.Hes1Model(m0=m0, fixed_parameters=iparameters)
        times = model.suggested_times()
        state = [4.0, 3.0, 3.5]
        jacobian = model.jacobian(state, 0.0, parameters)
        self.assertEqual(jacobian[0, 0], -0.04)
        self.assertEqual(jacobian[0, 1], 0.0)
        self.assertAlmostEqual(jacobian[0, 2], -0.48782911794811024)
        self.assertEqual(jacobian[1, 0], 0.035)
        self.assertEqual(jacobian[1, 1], -0.19)
        self.assertEqual(jacobian[1, 2], 0.0)
        self.assertEqual(jacobian[2, 0], 0.0)
        self.assertEqual(jacobian[2, 1], 0.15)
        self.assertEqual(jacobian[2, 2], -0.04)
        dfdp = model._dfdp(state, 0.0, parameters)
        self.assertAlmostEqual(dfdp[0, 0], 0.4493162928469437)
        self.assertEqual(dfdp[0, 1], 0)
        self.assertEqual(dfdp[0, 2], 0)
        self.assertAlmostEqual(dfdp[0, 3], 0.018721798164847005)
        self.assertEqual(dfdp[1, 0], 0)
        self.assertEqual(dfdp[1, 1], 4)
        self.assertEqual(dfdp[1, 2], -3)
        self.assertEqual(dfdp[1, 3], 0)
        self.assertEqual(dfdp[2, 0], 0)
        self.assertEqual(dfdp[2, 1], 0)
        self.assertEqual(dfdp[2, 2], 3)
        self.assertEqual(dfdp[2, 3], 0)
        values = model.simulate(parameters, times)
        values1, dvals = model.simulateS1(parameters, times)
        self.assertTrue(np.array_equal(values.shape, [len(times), ]))
        self.assertTrue(np.array_equal(values1.shape, [len(times), ]))
        self.assertTrue(
            np.array_equal(dvals.shape,
                           np.array([len(times),
                                    model.n_outputs(), model.n_parameters()])))
        self.assertEqual(dvals[0][0, 0], 0.0)
        self.assertAlmostEqual(dvals[1][0, 0], 2.029494068269803)
        self.assertAlmostEqual(dvals[2][0, 1], -197.7831610097427)
        self.assertAlmostEqual(dvals[2][0, 2], -10.15634595324204)
        self.assertAlmostEqual(dvals[3][0, 3], -0.4269233678552582)


if __name__ == '__main__':
    unittest.main()
