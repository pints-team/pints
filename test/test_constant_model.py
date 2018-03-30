#!/usr/bin/env python3
#
# Tests if the constant (toy) model works.
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


class TestConstantModel(unittest.TestCase):
    """
    Tests if the constant (toy) model with multiple outputs works.
    """

    def test_zero(self):
        # Test the special case where value is zero for a single input
        model = pints.toy.ConstantModel()
        times = [0, 1, 2, 10000]
        parameters = [0]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        for v in values:
            self.assertEqual(v, 0)

    def test_minus_1_2_100(self):
        model = pints.toy.ConstantModel()
        times = [0, 1, 2, 10000]
        parameters = [-1, 2, 100]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        i = 0
        for v in values:
            for x in v:
                self.assertEqual(x, parameters[i])
        i += 1

    def test_random_number_parameters(self):
        model = pints.toy.ConstantModel()
        times = [0, 1, 2, 10000]
        no = np.random.randint(low=1, high=10, size=1)
        parameters = np.random.uniform(low=-100, high=1000, size=no)
        values = model.simulate(parameters, times)
        for v in values:
            self.assertEqual(len(v), len(times))
        i = 0
        for v in values:
            for x in v:
                self.assertEqual(x, parameters[i])
        i += 1

    def test_errors(self):
        model = pints.toy.ConstantModel()
        times = [0, -1, 2, 10000]
        self.assertRaises(ValueError, model.simulate, [1], times)
        times = [0, 1, 2, 10000]
        self.assertRaises(ValueError, model.simulate, [1, 2], times)
        self.assertRaises(ValueError, model.simulate, [-10, np.nan], times)
        self.assertRaises(ValueError, model.simulate, [np.inf], times)
        self.assertRaises(ValueError, model.simulate, [-np.inf], times)


if __name__ == '__main__':
    unittest.main()
