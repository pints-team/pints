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

        # supplied only a times vector without data
        self.assertRaisesRegex(
            ValueError, 'Must supply times and values for each time series.',
            pints.ProblemCollection, self.model, self.times_12,
            self.outputs_12, self.times_3)


if __name__ == '__main__':
    unittest.main()
