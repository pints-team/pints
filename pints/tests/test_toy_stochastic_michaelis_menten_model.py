#!/usr/bin/env python3
#
# Tests if the Michaelis Menten (toy) model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
from pints.toy.stochastic import MichaelisMentenModel


class TestMichaelisMentenModel(unittest.TestCase):
    """
    Tests if the Michaelis Menten (toy) model works.
    """
    def test_n_parameters(self):
        x_0 = [1e4, 2e3, 2e4, 0]
        model = MichaelisMentenModel(x_0)
        self.assertEqual(model.n_parameters(), 3)

    def test_simulation_length(self):
        x_0 = [1e4, 2e3, 2e4, 0]
        model = MichaelisMentenModel(x_0)
        times = np.linspace(0, 1, 100)
        k = [1e-5, 0.2, 0.2]
        values = model.simulate(k, times)
        self.assertEqual(len(values), 100)

    def test_propensities(self):
        x_0 = [1e4, 2e3, 2e4, 0]
        k = [1e-5, 0.2, 0.2]
        model = MichaelisMentenModel(x_0)
        self.assertTrue(
            np.allclose(
                model._propensities(x_0, k),
                np.array([200.0, 4000.0, 4000.0])))

    def test_n_outputs(self):
        x_0 = [1e4, 2e3, 2e4, 0]
        model = MichaelisMentenModel(x_0)
        self.assertEqual(model.n_outputs(), 4)


if __name__ == '__main__':
    unittest.main()
