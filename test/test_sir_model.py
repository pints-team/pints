#!/usr/bin/env python
#
# Tests if the SIR toy model runs.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import pints.toy


class TestSIRModel(unittest.TestCase):
    """
    Tests if the SIR toy model runs.
    """

    def test_run(self):

        # Test basic properties
        model = pints.toy.SIRModel()
        self.assertEqual(model.n_parameters(), 3)
        self.assertEqual(model.n_outputs(), 2)

        # Test simulation
        x = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (len(times), model.n_outputs()))

        # Test suggested values
        v = model.suggested_values()
        self.assertEqual(v.shape, (len(times), model.n_outputs()))

        # Test setting intial conditions
        model = pints.toy.SIRModel([1, 1, 1])

        # Must have 3 init cond.
        self.assertRaises(
            ValueError, pints.toy.SIRModel, [1, 1])

        # Populations are never negative
        self.assertRaises(
            ValueError, pints.toy.SIRModel, [1, 1, 1, -1, 1, 1])


if __name__ == '__main__':
    unittest.main()
