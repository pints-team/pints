#!/usr/bin/env python3
#
# Tests if the Markov Jump model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
from pints.toy import MichaelisMentenModel


class TestMarkovJumpModel(unittest.TestCase):
    """
    Tests if the Markov Jump model works.
    """
    def test_errors(self):
        # Negative values for the initial population should
        # raise an error
        x_0 = [-1, 0, 1]
        model = MichaelisMentenModel(x_0)
        self.assertRaises(ValueError, MichaelisMentenModel, x_0)

        # Negative times should raise an error
        x_0 = [1e4, 2e3, 2e4, 0]
        model = MichaelisMentenModel(x_0)
        times = [-1, 0, 1]
        k = [1e-5, 0.2, 0.2]
        self.assertRaises(ValueError, model.simulate, k, times)

    def test_simulation(self):
        x_0 = [1e4, 2e3, 2e4, 0]
        model = MichaelisMentenModel(x_0)
        times = np.linspace(0, 1, 100)
        k = [1e-5, 0.2, 0.2]
        values = model.simulate(k, times)
        self.assertEqual(len(values), 100)


if __name__ == '__main__':
    unittest.main()
