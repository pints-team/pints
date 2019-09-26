#!/usr/bin/env python3
#
# Tests if the goodwin oscillator (toy) model runs.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
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


if __name__ == '__main__':
    unittest.main()
