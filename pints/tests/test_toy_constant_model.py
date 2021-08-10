#!/usr/bin/env python3
#
# Tests if the constant (toy) model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy


class TestConstantModel(unittest.TestCase):
    """
    Tests if the constant (toy) model with multiple outputs works.
    """

    def test_1d(self):
        # Test in 1d (SingleOutputProblem).

        # Output given for a SingleOutputProblem
        model = pints.toy.ConstantModel(1)
        times = [0, 1, 2, 10000]
        parameters = [0]
        values = model.simulate(parameters, times)

        # Check output shape (n_times, n_outputs) and values
        self.assertEqual(values.shape, (len(times), ))
        self.assertTrue(np.all(values == 0))

        # Output given for a MultiOutputProblem with 1 output
        model = pints.toy.ConstantModel(1, force_multi_output=True)
        times = [0, 1, 2, 10000]
        parameters = [-2]
        values = model.simulate(parameters, times)

        # Check output shape (n_times, n_outputs) and values
        self.assertEqual(values.shape, (len(times), 1))
        self.assertTrue(np.all(values == -2))

    def test_3d(self):
        # Test three-dimensional case (MultiOutputProblem).

        model = pints.toy.ConstantModel(3)
        times = [0, 1, 2, 10000]
        parameters = [-1, 2, 100]
        values = model.simulate(parameters, times)
        multipliers = np.arange(1, 1 + len(parameters))

        # Check output shape (n_times, n_outputs) and values
        self.assertEqual(values.shape, (len(times), len(parameters)))
        for column in values:
            self.assertTrue(np.all(column == parameters * multipliers))

    def test_varying_numbers_of_parameters(self):
        # Tests for different parameter sizes (All using MultiOutputProblem).

        times = [0, 1, 2, 10000]
        for n in range(1, 10):
            model = pints.toy.ConstantModel(n, force_multi_output=True)
            parameters = np.random.uniform(low=-100, high=1000, size=n)
            values = model.simulate(parameters, times)
            multipliers = np.arange(1, 1 + n)

            # Check output shape (n_times, n_outputs) and values
            self.assertEqual(values.shape, (len(times), len(parameters)))
            for column in values:
                self.assertTrue(np.all(column == parameters * multipliers))

    def test_errors(self):
        # Tests the right errors are raised when used improperly.

        # Negative times
        model = pints.toy.ConstantModel(1)
        times = [0, -1, 2, 10000]
        self.assertRaisesRegex(
            ValueError, 'Negative times', model.simulate, [1], times)
        times = [0, 1, 2, 10000]

        # Wrong number of parameters
        self.assertRaisesRegex(
            ValueError, 'Expected 1', model.simulate, [], times)
        self.assertRaisesRegex(
            ValueError, 'Expected 1', model.simulate, [1, 1], times)

        # Non-finite parameters
        self.assertRaisesRegex(
            ValueError, 'must be finite', model.simulate, [np.nan], times)
        self.assertRaisesRegex(
            ValueError, 'must be finite', model.simulate, [np.inf], times)
        self.assertRaisesRegex(
            ValueError, 'must be finite', model.simulate, [-np.inf], times)

        # Invalid number of parameters
        self.assertRaisesRegex(
            ValueError, '1 or greater', pints.toy.ConstantModel, 0)
        self.assertRaisesRegex(
            ValueError, '1 or greater', pints.toy.ConstantModel, -1)

    def test_in_problem(self):
        # Tests using a ConstantModel in single and multi-output problems.

        # Single output
        model = pints.toy.ConstantModel(1)
        times = [0, 1, 2, 1000]
        values = [10, 0, 1, 10]
        problem = pints.SingleOutputProblem(model, times, values)
        problem.evaluate([1])

        # Multi output (n=1)
        problem = pints.MultiOutputProblem(model, times, values)
        problem.evaluate([1])

        # Multi output (n=3)
        model = pints.toy.ConstantModel(3)
        times = [0, 1, 2, 1000]
        values = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [8, 7, 6]]
        problem = pints.MultiOutputProblem(model, times, values)
        problem.evaluate([1, 2, 3])

    def test_derivatives(self):
        # Tests the derivatives are returned correctly.

        # Single output
        model = pints.toy.ConstantModel(1)
        times = [0, 1, 2, 1000]
        x = [3]
        y, dy = model.simulateS1(x, times)
        self.assertEqual(dy.shape, (4, ))
        self.assertTrue(np.all(dy == 1))
        self.assertTrue(np.all(y == model.simulate(x, times)))

        # Multi-output with only 1 output
        model = pints.toy.ConstantModel(1, force_multi_output=True)
        times = [0, 1, 2, 1000]
        x = [3]
        y, dy = model.simulateS1(x, times)
        self.assertEqual(dy.shape, (4, 1, 1))
        self.assertTrue(np.all(dy == 1))
        self.assertTrue(np.all(y == model.simulate(x, times)))

        # Multi-output
        model = pints.toy.ConstantModel(2)
        times = [0, 1, 2]
        x = [1, 2]

        # Check model output
        y, dy = model.simulateS1(x, times)
        # Shape is (times, outputs)
        self.assertEqual(y.shape, (3, 2))
        mx = np.array(
            [[1, 4],
             [1, 4],
             [1, 4]]
        )
        self.assertTrue(np.all(y == mx))
        self.assertTrue(np.all(y == model.simulate(x, times)))
        # Shape is (times, outputs, parameters)
        self.assertEqual(dy.shape, (3, 2, 2))
        dmx = np.array(
            [[[1, 0],   # dx1/dp1 = 1, dx1/dp2 = 0
              [0, 2]],  # dx2/dp2 = 0, dx2/dp2 = 2
             [[1, 0],
              [0, 2]],
             [[1, 0],
              [0, 2]]]
        )
        self.assertTrue(np.all(dy == dmx))


if __name__ == '__main__':
    unittest.main()
