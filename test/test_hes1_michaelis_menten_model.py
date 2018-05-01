#!/usr/bin/env python
#
# Tests if the HES1 Michaelis-Menten toy model runs.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
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
        self.assertEqual(values.shape, (len(times),))
        self.assertTrue(np.all(values > 0))
        states = model.simulate_all_states(parameters, times)
        self.assertEqual(states.shape, (len(times), 3))
        self.assertTrue(np.all(states > 0))
        suggested_values = model.suggested_values()
        self.assertEqual(suggested_values.shape, (len(times),))
        self.assertTrue(np.all(suggested_values > 0))

        # Test setting and getting init cond.
        self.assertFalse(np.all(model.initial_conditions() == 10))
        model.set_initial_conditions(10)
        self.assertTrue(np.all(model.initial_conditions() == 10))

        # Test setting and getting implicit param.
        self.assertFalse(np.all(model.implicit_parameters() == [10, 10, 10]))
        model.set_implicit_parameters([10, 10, 10])
        self.assertTrue(np.all(model.implicit_parameters() == [10, 10, 10]))

        # Initial conditions cannot be negative
        model = pints.toy.Hes1Model(0)
        self.assertRaises(ValueError, pints.toy.Hes1Model, -1)

        # Implicit parameters cannot be negative
        model = pints.toy.Hes1Model(0, [0, 0, 0])
        self.assertRaises(ValueError, pints.toy.Hes1Model, *(0, [-1, 0, 0]))
        self.assertRaises(ValueError, pints.toy.Hes1Model, *(0, [0, -1, 0]))
        self.assertRaises(ValueError, pints.toy.Hes1Model, *(0, [0, 0, -1]))
        self.assertRaises(ValueError, pints.toy.Hes1Model, *(0, [-1, -1, -1]))


if __name__ == '__main__':
    unittest.main()
