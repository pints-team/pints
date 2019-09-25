#!/usr/bin/env python
#
# Tests if the Lotka-Volterra toy model runs.
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


class TestLotkaVolterraModel(unittest.TestCase):
    """
    Tests if the Lotka-Volterra toy model runs.
    """

    def test_run(self):
        model = pints.toy.LotkaVolterraModel()
        self.assertEqual(model.n_parameters(), 4)
        self.assertEqual(model.n_outputs(), 2)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        values = model.simulate(parameters, times)
        self.assertEqual(values.shape, (len(times), 2))
        self.assertTrue(np.all(values > 0))

        # Test setting and getting init cond.
        self.assertFalse(np.all(model.initial_conditions() == [10, 10]))
        model.set_initial_conditions([10, 10])
        self.assertTrue(np.all(model.initial_conditions() == [10, 10]))

        # Initial conditions cannot be negative
        model = pints.toy.LotkaVolterraModel([0, 0])
        self.assertRaises(ValueError, pints.toy.LotkaVolterraModel, [-1, 0])
        self.assertRaises(ValueError, pints.toy.LotkaVolterraModel, [0, -1])
        self.assertRaises(ValueError, pints.toy.LotkaVolterraModel, [-1, -1])

    def test_values(self):
        # value-based tests of solution
        x0 = 3
        y0 = 5
        model = pints.toy.LotkaVolterraModel([x0, y0])
        parameters = [1, 2, 2, 0.5]
        times = np.linspace(0, 5, 101)
        values = model.simulate(parameters, times)
        self.assertEqual(values[0, 0], x0)
        self.assertEqual(values[0, 1], y0)
        self.assertTrue(np.abs(values[1, 0] - 1.9294938874573144) < 0.0001)
        self.assertTrue(np.abs(values[1, 1] - 4.8065419518727595) < 0.0001)
        self.assertTrue(np.abs(values[100, 0] - 1.2777621132036345) < 0.0001)
        self.assertTrue(np.abs(values[100, 1] - 0.0005294772711015946) <
                        0.0001)


if __name__ == '__main__':
    unittest.main()
