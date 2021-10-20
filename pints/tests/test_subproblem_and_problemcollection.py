#!/usr/bin/env python3
#
# Tests SubProblem and ProblemCollection classes
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import numpy as np
import unittest


class TestProblemCollection(unittest.TestCase):
    """
    Tests ProblemCollection methods.
    """
    @classmethod
    def setUpClass(cls):
        """ Prepare problem for tests. """

        model = pints.toy.GoodwinOscillatorModel()
        real_parameters = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(real_parameters, times)
        cls.model = model
        cls.real_parameters = real_parameters
        cls.times = times

        # add noise
        noise1 = 0.001
        noise2 = 0.01
        noise3 = 0.1
        noisy_values = np.array(values, copy=True)
        noisy_values[:, 0] += np.random.normal(0, noise1, len(times))
        noisy_values[:, 1] += np.random.normal(0, noise2, len(times))
        noisy_values[:, 2] += np.random.normal(0, noise3, len(times))

        cls.times_12 = times[:100]
        cls.outputs_12 = noisy_values[:100, :2]
        cls.times_3 = times[100:]
        cls.outputs_3 = noisy_values[100:, 2]

    def test_problem_collection_methods(self):
        # Tests problem collection

        collection = pints.ProblemCollection(
            self.model, self.times_12, self.outputs_12, self.times_3,
            self.outputs_3)

        # check overall times
        timeses = collection.timeses()
        times_stack = [self.times_12, self.times_3]
        k = 0
        for times in timeses:
            self.assertTrue(np.array_equal(times, times_stack[k]))
            k += 1

        # check overall values
        valueses = collection.valueses()
        values_stack = [self.outputs_12, self.outputs_3]
        k = 0
        for values in valueses:
            self.assertTrue(np.array_equal(values, values_stack[k]))
            k += 1

        # test subproblem classes
        problem_0 = collection.subproblem(0)
        problem_1 = collection.subproblem(1)

        self.assertTrue(isinstance(problem_0, pints.SubProblem))
        self.assertTrue(isinstance(problem_1, pints.SubProblem))

        # check model returned
        model = collection.model()
        self.assertTrue(isinstance(model, pints.ForwardModelS1))

    def test_problem_collection_errors(self):
        # Tests problem collection errors

        # supplied no data?
        self.assertRaisesRegex(
            ValueError, 'Must supply at least one time series.',
            pints.ProblemCollection, self.model)

        # supplied only a times vector without data?
        self.assertRaisesRegex(
            ValueError, 'Must supply times and values for each time series.',
            pints.ProblemCollection, self.model, self.times_12,
            self.outputs_12, self.times_3)

        # supplied a 2d time vector?
        self.assertRaisesRegex(
            ValueError, 'Times must be one-dimensional.',
            pints.ProblemCollection, self.model, self.outputs_12,
            self.outputs_12)

        # supplied times that aren't same length as outputs?
        self.assertRaisesRegex(
            ValueError, 'Outputs must be of same length as times.',
            pints.ProblemCollection, self.model, [1, 2, 3, 4, 5],
            self.outputs_12)

        # selected index exceeding number of output chunks?
        collection = pints.ProblemCollection(
            self.model, self.times_12, self.outputs_12, self.times_3,
            self.outputs_3)

        self.assertRaisesRegex(
            ValueError, 'Index must be less than number of output sets.',
            collection.subproblem, 2
        )


class TestSubProblem(unittest.TestCase):
    """
    Tests SubProblem methods.
    """
    @classmethod
    def setUpClass(cls):
        """ Prepare problem for tests. """

        model = pints.toy.GoodwinOscillatorModel()
        real_parameters = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(real_parameters, times)
        cls.model = model
        cls.real_parameters = real_parameters
        cls.times = times
        cls.values = values

        cls.times_12 = times[:100]
        cls.outputs_12 = values[:100, :2]
        cls.times_3 = times[100:]
        cls.outputs_3 = values[100:, 2]

        collection = pints.ProblemCollection(
            cls.model, cls.times_12, cls.outputs_12, cls.times_3,
            cls.outputs_3)
        cls.problem_0 = collection.subproblem(0)
        cls.problem_1 = collection.subproblem(1)

        # also solve using sensitivity methods as ODE solution (due to
        # numerics) very slightly different
        val_s, dy = model.simulateS1(real_parameters, times)

        cls.outputs_12_s = val_s[:100, :2]
        cls.outputs_3_s = val_s[100:, 2]

        cls.dy_0 = dy[:100, :2, :]

        # reshape output here otherwise loses a dimension
        dy_1 = dy[100:, 2, :]
        cls.dy_1 = dy_1.reshape(100, 1, 5)

    def test_evaluate(self):
        # Tests that chunked solution same as splitting overall into bits

        sol_0 = self.problem_0.evaluate(self.real_parameters)
        sol_1 = self.problem_1.evaluate(self.real_parameters)

        self.assertTrue(np.array_equal(sol_0, self.outputs_12))
        self.assertTrue(np.array_equal(sol_1, self.outputs_3))

    def test_evaluateS1(self):
        # Tests that chunked solution and sens same as splitting overall

        sol_0, dy_0 = self.problem_0.evaluateS1(self.real_parameters)
        sol_1, dy_1 = self.problem_1.evaluateS1(self.real_parameters)

        self.assertTrue(np.array_equal(sol_0, self.outputs_12_s))
        self.assertTrue(np.array_equal(sol_1, self.outputs_3_s))
        self.assertTrue(np.array_equal(self.dy_0, dy_0))
        self.assertTrue(np.array_equal(self.dy_1, dy_1))

    def test_methods(self):
        # Tests methods return appropriate values

        # outputs correct?
        self.assertEqual(self.problem_0.n_outputs(), 2)
        self.assertEqual(self.problem_1.n_outputs(), 1)

        # parameters correct?
        self.assertEqual(self.problem_0.n_parameters(), 5)
        self.assertEqual(self.problem_1.n_parameters(), 5)

        # times correct?
        self.assertTrue(np.array_equal(self.times_12, self.problem_0.times()))
        self.assertTrue(np.array_equal(self.times_3, self.problem_1.times()))

        # n_times correct?
        self.assertEqual(len(self.times_12), self.problem_0.n_times())
        self.assertEqual(len(self.times_3), self.problem_1.n_times())

        # values correct?
        self.assertTrue(np.array_equal(
            self.outputs_12, self.problem_0.values()))
        self.assertTrue(np.array_equal(
            self.outputs_3, self.problem_1.values()))


if __name__ == '__main__':
    unittest.main()
