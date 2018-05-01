#!/usr/bin/env python3
#
# Tests if the Repressilator toy model runs.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import pints.toy


class TestRepressilatorModel(unittest.TestCase):
    """
    Tests if the Repressilator toy model runs.
    """

    def test_run(self):

        # Test basic properties
        model = pints.toy.RepressilatorModel()
        self.assertEqual(model.n_parameters(), 4)
        self.assertEqual(model.n_outputs(), 3)

        # Test simulation
        x = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (len(times), model.n_outputs()))

        # Test setting intial conditions
        model = pints.toy.RepressilatorModel([1, 1, 1, 1, 1, 1])

        # Must have 6 init cond.
        self.assertRaises(
            ValueError, pints.toy.RepressilatorModel, [1, 1, 1, 1, 1])

        # Concentrations are never negative
        self.assertRaises(
            ValueError, pints.toy.RepressilatorModel, [1, 1, 1, -1, 1, 1])


if __name__ == '__main__':
    unittest.main()
