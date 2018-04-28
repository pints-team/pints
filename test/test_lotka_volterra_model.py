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


class LotkaVolterraModel(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
