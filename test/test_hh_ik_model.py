#!/usr/bin/env python3
#
# Tests if the Hodgkin-Huxley IK (toy) model works.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np
import pints
import pints.toy


class TestHodgkinHuxleyIKModel(unittest.TestCase):
    """
    Tests if the Hodgkin-Huxley IK (toy) model works.
    """

    def test_hh_ik(self):
        model = pints.toy.HodgkinHuxleyIKModel()
        p0 = model.suggested_parameters()
        self.assertEqual(len(p0), model.dimension())
        times = model.suggested_times()
        self.assertTrue(np.all(times[1:] > times[:-1]))
        values = model.simulate(p0, times)
        self.assertEqual(len(times), len(values))
        folded = model.fold(times, values)
        self.assertEqual(len(folded), 12)
        # Test duration
        self.assertEqual(model.suggested_duration(), 1200)
        # Test protocol out of bounds
        self.assertEqual(model._protocol(-1), -75)
        self.assertEqual(model._protocol(9999), -75)
        # Test initial condition out of bounds
        self.assertRaises(ValueError, pints.toy.HodgkinHuxleyIKModel, 0)
        self.assertRaises(ValueError, pints.toy.HodgkinHuxleyIKModel, 1)


if __name__ == '__main__':
    unittest.main()
