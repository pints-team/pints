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


if __name__ == '__main__':
    unittest.main()
