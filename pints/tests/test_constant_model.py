#!/usr/bin/env python3
#
# Tests if the constant (toy) model works.
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


class TestConstantModel(unittest.TestCase):
    """
    Tests if the constant (toy) model with multiple outputs works.
    """

    def test_zero(self):
        # Test the special case where value is zero for a single input
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
        parameters = [0]
        values = model.simulate(parameters, times)

        # Check output shape (n_times, n_outputs) and values
        self.assertEqual(values.shape, (len(times), 1))
        self.assertTrue(np.all(values == 0))

    def test_minus_1_2_100(self):
        model = pints.toy.ConstantModel(3)
        times = [0, 1, 2, 10000]
        parameters = [-1, 2, 100]
        values = model.simulate(parameters, times)

        # Check output shape (n_times, n_outputs) and values
        self.assertEqual(values.shape, (len(times), len(parameters)))
        for i, p in enumerate(parameters):
            self.assertTrue(np.all(values[:, i] == p))

    def test_varying_numbers_of_parameters(self):
        times = [0, 1, 2, 10000]
        for n in range(2, 10):
            model = pints.toy.ConstantModel(n)
            parameters = np.random.uniform(low=-100, high=1000, size=n)
            values = model.simulate(parameters, times)
            # Check output shape (n_times, n_outputs) and values
            self.assertEqual(values.shape, (len(times), len(parameters)))
            for i, p in enumerate(parameters):
                self.assertTrue(np.all(values[:, i] == p))

    def test_errors(self):
        # Negative times
        model = pints.toy.ConstantModel(1)
        times = [0, -1, 2, 10000]
        self.assertRaises(ValueError, model.simulate, [1], times)
        times = [0, 1, 2, 10000]

        # Wrong number of parameters
        self.assertRaises(ValueError, model.simulate, [], times)
        self.assertRaises(ValueError, model.simulate, [1, 1], times)

        # Non-finite parameters
        self.assertRaises(ValueError, model.simulate, [np.nan], times)
        self.assertRaises(ValueError, model.simulate, [np.inf], times)
        self.assertRaises(ValueError, model.simulate, [-np.inf], times)

        # Invalid number of parameters
        self.assertRaises(ValueError, pints.toy.ConstantModel, 0)
        self.assertRaises(ValueError, pints.toy.ConstantModel, -1)

    def test_in_problem(self):
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

        # Single output
        model = pints.toy.ConstantModel(1)
        times = [0, 1, 2, 1000]
        values = [10, 0, 1, 10]
        problem = pints.SingleOutputProblem(model, times, values)
        x = [3]
        y, dy = problem.evaluateS1(x)
        self.assertEqual(dy.shape, (4, ))
        self.assertTrue(np.all(dy == 1))
        self.assertTrue(np.all(y == problem.evaluate(x)))

        # Multi-output
        model = pints.toy.ConstantModel(2)
        times = [0, 1, 2, 1000]
        values = [[0, 0], [1, 10], [2, 20], [3, 30]]
        problem = pints.MultiOutputProblem(model, times, values)
        x = [3, 4]
        y, dy = problem.evaluateS1(x)
        self.assertEqual(dy.shape, (4, 2, 2))
        self.assertTrue(np.all(dy == 1))
        self.assertTrue(np.all(y == problem.evaluate(x)))


if __name__ == '__main__':
    unittest.main()
