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
    Tests if the constant (toy) model works.
    """

    def test_zero(self):
        # Test the special case where the initial size is zero
        model = pints.toy.ConstantModel()
        times = [0, 1, 2, 10000]
        parameters = [0]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        for v in values:
            self.assertEqual(v, 0)

    def test_100(self):
        # Test the special case where the initial size is zero
        model = pints.toy.ConstantModel()
        times = [0, 1, 2, 10000]
        parameters = [100]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        for v in values:
            self.assertEqual(v, 100)

    def test_errors(self):
        model = pints.toy.ConstantModel()
        times = [0, -1, 2, 10000]
        self.assertRaises(ValueError, model.simulate, [1], times)
        times = [0, 1, 2, 10000]
        self.assertRaises(ValueError, model.simulate, [1, 2], times)
        self.assertRaises(ValueError, model.simulate, [np.nan], times)
        self.assertRaises(ValueError, model.simulate, [np.inf], times)
        self.assertRaises(ValueError, model.simulate, [-np.inf], times)


if __name__ == '__main__':
    unittest.main()
