#!/usr/bin/env python3
#
# Tests SingleOutputProblem methods.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import numpy as np
import unittest


class TestSingleOutputProblem(unittest.TestCase):
    """
    Tests SingleOutputProblem methods.
    """
    def test_basics(self):
        # Test everything

        model = pints.toy.LogisticModel()
        times = [0, 1, 2, 3]
        x = [1, 1]
        values = model.simulate(x, times)
        noisy = values + np.array([0.01, -0.01, 0.01, -0.01])
        problem = pints.SingleOutputProblem(model, times, noisy)

        self.assertTrue(np.all(times == problem.times()))
        self.assertTrue(np.all(noisy == problem.values()))
        self.assertTrue(np.all(values == problem.evaluate(x)))
        self.assertEqual(problem.n_parameters(), model.n_parameters(), 2)
        self.assertEqual(problem.n_outputs(), model.n_outputs(), 1)
        self.assertEqual(problem.n_times(), len(times))

        # Test errors
        times[0] = -2
        self.assertRaises(
            ValueError, pints.SingleOutputProblem, model, times, values)
        times = [1, 2, 2, 1]
        self.assertRaises(
            ValueError, pints.SingleOutputProblem, model, times, values)
        times = [1, 2, 3]
        self.assertRaises(
            ValueError, pints.SingleOutputProblem, model, times, values)

        # Multi-output problem not allowed
        model = pints.toy.FitzhughNagumoModel()
        self.assertEqual(model.n_outputs(), 2)
        values = model.simulate([1, 1, 1], times)
        self.assertRaises(
            ValueError, pints.SingleOutputProblem, model, times, values)


if __name__ == '__main__':
    unittest.main()
