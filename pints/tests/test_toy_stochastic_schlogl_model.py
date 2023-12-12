#!/usr/bin/env python3
#
# Tests if the Schlogl (toy) model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
from pints.toy.stochastic import SchloglModel


class TestSchloglModel(unittest.TestCase):
    """
    Tests if the degradation (toy) model works.
    """
    def test_n_parameters(self):
        x_0 = 20
        model = SchloglModel(x_0)
        self.assertEqual(model.n_parameters(), 4)

    def test_simulation_length(self):
        x_0 = 20
        model = SchloglModel(x_0)
        times = np.linspace(0, 1, 100)
        k = [0.1, 0.2, 0.3, 0.4]
        values = model.simulate(k, times)
        self.assertEqual(len(values), 100)

    def test_propensities(self):
        x_0 = 20
        model = SchloglModel(x_0)
        k = model.suggested_parameters()
        self.assertTrue(
            np.allclose(
                model._propensities([x_0], k),
                np.array([68.4, 1.71, 2200.0, 750.0])))

    def test_suggested(self):
        model = SchloglModel(20)
        times = model.suggested_times()
        parameters = model.suggested_parameters()
        self.assertTrue(len(times) == 101)
        self.assertTrue(np.all(parameters > 0))


if __name__ == '__main__':
    unittest.main()
