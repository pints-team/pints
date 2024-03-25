#!/usr/bin/env python3
#
# Tests the error measure classes.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np


class MiniProblem(pints.SingleOutputProblem):
    def __init__(self):
        self._t = pints.vector([1, 2, 3])
        self._v = pints.vector([-1, 2, 3])

    def n_parameters(self):
        return 3

    def evaluate(self, parameters):
        return np.array(parameters)

    def times(self):
        return self._t

    def values(self):
        return self._v


class MultiMiniProblem(pints.MultiOutputProblem):
    def __init__(self):
        self._t = pints.vector([1, 2, 3])
        self._v = pints.matrix2d(
            np.array([[-1, 2, 3], [-1, 2, 3]]).swapaxes(0, 1))

    def n_parameters(self):
        return 3

    def n_outputs(self):
        return 2

    def evaluate(self, parameters):
        return np.array([parameters, parameters]).swapaxes(0, 1)

    def times(self):
        return self._t

    def values(self):
        return self._v


class BigMiniProblem(MiniProblem):
    def __init__(self):
        super(BigMiniProblem, self).__init__()
        self._t = pints.vector([1, 2, 3, 4, 5, 6])
        self._v = pints.vector([-1, 2, 3, 4, 5, -6])

    def n_parameters(self):
        return 6


class BadMiniProblem(MiniProblem):
    def __init__(self, bad_value=np.inf):
        super(BadMiniProblem, self).__init__()
        self._v = pints.vector([bad_value, 2, -3])

    def n_parameters(self):
        return 3


class BadErrorMeasure(pints.ErrorMeasure):
    def __init__(self, bad_value=-np.inf):
        super(BadErrorMeasure, self).__init__()
        self._v = bad_value

    def n_parameters(self):
        return 3

    def __call__(self, parameters):
        return self._v


class MiniLogPDF(pints.LogPDF):
    def n_parameters(self):
        return 3

    def __call__(self, parameters):
        return 10

    def evaluateS1(self, parameters):
        return 10, np.array([1, 2, 3])


class TestMeanSquaredError(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(2)

        # Generate test data
        cls.times = [1, 2, 3]
        cls.n_times = len(cls.times)
        cls.data_single = np.array([1, 1, 1])
        cls.data_multi = np.array([
            [1, 4],
            [1, 4],
            [1, 4]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.MeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = mean(input - 1) ** 2
        self.assertEqual(score, 4)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.MeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = mean(input - 1) ** 2
        self.assertEqual(score, 4)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.MeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = mean(input - 1) ** 2
        self.assertEqual(score, 4)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create error measure
        error = pints.MeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3, 4]
        score = error(test_parameters)

        # Check that error returns expected value
        # exp = (mean(input[0] - 1) ** 2 + mean(2 * input[1] - 4) ** 2) / 2
        self.assertEqual(score, 10)

    def test_call_two_dim_array_multi_weighted(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create error measure with weighted input
        weights = [1, 2]
        error = pints.MeanSquaredError(problem, weights=weights)

        # Evaluate likelihood for test parameters
        test_parameters = [3, 4]
        score = error(test_parameters)

        # Check that error returns expected value
        # exp = (weight[0] * mean(input[0] - 1) ** 2 +
        # weight[1] * mean(2 * input[1] - 4) ** 2) / 2
        self.assertEqual(score, 18)

    def test_evaluateS1_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.MeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (1,))

        # Check that partials are correct
        # Expected = 2 * mean(input - 1)
        self.assertEqual(deriv, 2 * (test_parameters[0] - 1))

    def test_evaluateS1_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.MeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (1,))

        # Check that partials are correct
        # Expected = 2 * mean(input - 1)
        self.assertEqual(deriv, 2 * (test_parameters[0] - 1))

    def test_evaluateS1_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.MeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (1,))

        # Check that partials are correct
        # Expected = 2 * mean(input - 1)
        self.assertEqual(deriv, 2 * (test_parameters[0] - 1))

    def test_evaluateS1_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create error measure
        error = pints.MeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3, 4]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (2,))

        # Check that partials are correct
        # Expectation = [mean(input[0] - 1), 2 * mean(2 * input[1] - 4)]
        self.assertEqual(deriv[0], test_parameters[0] - 1)
        self.assertEqual(deriv[1], 2 * (2 * test_parameters[1] - 4))

    def test_evaluateS1_two_dim_array_multi_weighted(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create error measure with weighted inputs
        weights = [1, 2]
        error = pints.MeanSquaredError(problem, weights=weights)

        # Evaluate likelihood for test parameters
        test_parameters = [3, 4]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (2,))

        # Check that partials are correct
        # expectation = [weight [0] * mean(input[0] - 1),
        # weight[1] * 2 * mean(2 * input[1] - 4)]
        self.assertEqual(deriv[0], weights[0] * (test_parameters[0] - 1))
        self.assertEqual(
            deriv[1], weights[1] * 2 * (2 * test_parameters[1] - 4))

    def test_bad_constructor(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_single, self.times, self.data_single)

        # Test invalid weight shape
        weights = [1, 2, 3]
        self.assertRaisesRegex(
            ValueError,
            'Number of weights must match number of problem outputs.',
            pints.MeanSquaredError, problem, weights)


class TestNormalisedRootMeanSquaredError(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(2)

        # Generate test data
        cls.times = [1, 2, 3]
        cls.n_times = len(cls.times)
        cls.data_single = np.array([2, 2, 2])
        cls.data_multi = np.array([
            [1, 4],
            [1, 4],
            [1, 4]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.NormalisedRootMeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = sqrt(mean((input - 1) ** 2)) / sqrt(mean(2 ** 2))
        self.assertEqual(score, 0.5)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.NormalisedRootMeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = sqrt(mean((input - 1) ** 2)) / sqrt(mean(2 ** 2))
        self.assertEqual(score, 0.5)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.NormalisedRootMeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = sqrt(mean((input - 1) ** 2)) / sqrt(mean(2 ** 2))
        self.assertEqual(score, 0.5)

    def test_not_implemented_error(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.NormalisedRootMeanSquaredError(problem)

        # Check that not implemented error is raised for evaluateS1
        self.assertRaisesRegex(
            NotImplementedError,
            '',
            error.evaluateS1, 1)

    def test_bad_constructor(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Check that an error is raised for multi-output problems
        self.assertRaisesRegex(
            ValueError,
            'This measure is only defined for single output problems.',
            pints.NormalisedRootMeanSquaredError, problem)


class TestProbabilityBasedError(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create probability based problem
        cls.problem = MiniLogPDF()

    def test_call(self):
        # Create error measure
        error = pints.ProbabilityBasedError(self.problem)

        # Evaluate likelihood for test parameters
        test_parameters = [1, 2, 3]
        score = error(test_parameters)

        # Check that error returns expected value
        self.assertEqual(score, -10)

    def test_evaluateS1(self):
        # Create error measure
        error = pints.ProbabilityBasedError(self.problem)

        # Evaluate likelihood for test parameters
        test_parameters = [1, 2, 3]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that error returns expected value
        self.assertEqual(score, error(test_parameters))

        # Check dimension of partial derivatives
        self.assertEqual(deriv.shape, (3,))

        # Check that partials are computed correctly
        self.assertEqual(deriv[0], -1)
        self.assertEqual(deriv[1], -2)
        self.assertEqual(deriv[2], -3)

    def test_n_parameters(self):
        # Create error measure
        error = pints.ProbabilityBasedError(self.problem)

        # Get number of parameters
        n_parameters = error.n_parameters()

        # Check number of parameters
        self.assertEqual(n_parameters, 3)

    def test_bad_constructor(self):
        # Check that an error is raised for multi-output problems
        self.assertRaisesRegex(
            ValueError,
            'Given log_pdf must be an instance of pints.LogPDF.',
            pints.ProbabilityBasedError, MiniProblem())


class TestRootMeanSquaredError(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(2)

        # Generate test data
        cls.times = [1, 2, 3]
        cls.n_times = len(cls.times)
        cls.data_single = np.array([1, 1, 1])
        cls.data_multi = np.array([
            [1, 4],
            [1, 4],
            [1, 4]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.RootMeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = sqrt(mean((input - 1) ** 2))
        self.assertEqual(score, 2)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.RootMeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = sqrt(mean((input - 1) ** 2))
        self.assertEqual(score, 2)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.RootMeanSquaredError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = sqrt(mean((input - 1) ** 2))
        self.assertEqual(score, 2)

    def test_not_implemented_error(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.RootMeanSquaredError(problem)

        # Check that not implemented error is raised for evaluateS1
        self.assertRaisesRegex(
            NotImplementedError,
            '',
            error.evaluateS1, 1)

    def test_bad_constructor(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Check that an error is raised for multi-output problems
        self.assertRaisesRegex(
            ValueError,
            'This measure is only defined for single output problems.',
            pints.RootMeanSquaredError, problem)


class TestSumOfErrors(unittest.TestCase):

    def test_sum_of_errors(self):
        # Tests :class:`pints.SumOfErrors`.

        e1 = pints.SumOfSquaresError(MiniProblem())
        e2 = pints.MeanSquaredError(MiniProblem())
        e3 = pints.RootMeanSquaredError(BigMiniProblem())
        e4 = pints.SumOfSquaresError(BadMiniProblem())

        # Basic use
        e = pints.SumOfErrors([e1, e2])
        x = [0, 0, 0]
        self.assertEqual(e.n_parameters(), 3)
        self.assertEqual(e(x), e1(x) + e2(x))
        e = pints.SumOfErrors([e1, e2], [3.1, 4.5])
        x = [0, 0, 0]
        self.assertEqual(e.n_parameters(), 3)
        self.assertEqual(e(x), 3.1 * e1(x) + 4.5 * e2(x))
        e = pints.SumOfErrors(
            [e1, e1, e1, e1, e1, e1], [1, 2, 3, 4, 5, 6])
        self.assertEqual(e.n_parameters(), 3)
        self.assertEqual(e(x), e1(x) * 21)
        self.assertNotEqual(e(x), 0)

        with np.errstate(all='ignore'):
            e = pints.SumOfErrors(
                [e4, e1, e1, e1, e1, e1], [10, 1, 1, 1, 1, 1])
            self.assertEqual(e.n_parameters(), 3)
            self.assertEqual(e(x), np.inf)
            e = pints.SumOfErrors(
                [e4, e1, e1, e1, e1, e1], [0, 2, 0, 2, 0, 2])
            self.assertEqual(e.n_parameters(), 3)
            self.assertTrue(e(x), 6 * e1(x))
            e5 = pints.SumOfSquaresError(BadMiniProblem(-np.inf))
            e = pints.SumOfErrors([e1, e5, e1], [2.1, 3.4, 6.5])
            self.assertTrue(np.isinf(e(x)))
            e = pints.SumOfErrors([e4, e5, e1], [2.1, 3.4, 6.5])
            self.assertTrue(np.isinf(e(x)))
            e5 = pints.SumOfSquaresError(BadMiniProblem(float('nan')))
            e = pints.SumOfErrors(
                [BadErrorMeasure(np.inf), BadErrorMeasure(np.inf)],
                [1, 1])
            self.assertEqual(e(x), np.inf)
            e = pints.SumOfErrors(
                [BadErrorMeasure(np.inf),
                 BadErrorMeasure(-np.inf)],
                [1, 1])
            self.assertTrue(np.isnan(e(x)))
            e = pints.SumOfErrors(
                [BadErrorMeasure(5), BadErrorMeasure(float('nan'))], [1, 1])
            self.assertTrue(np.isnan(e(x)))
            e = pints.SumOfErrors([e1, e5, e1], [2.1, 3.4, 6.5])
            self.assertTrue(np.isnan(e(x)))
            e = pints.SumOfErrors([e4, e5, e1], [2.1, 3.4, 6.5])
            self.assertTrue(np.isnan(e(x)))

        # Wrong number of ErrorMeasures
        self.assertRaises(ValueError, pints.SumOfErrors, [], [])

        # Wrong argument types
        self.assertRaises(
            TypeError, pints.SumOfErrors, [e1, e1], [e1, 1])
        self.assertRaises(
            ValueError, pints.SumOfErrors, [e1, 3], [2, 1])

        # Mismatching sizes
        self.assertRaises(
            ValueError, pints.SumOfErrors, [e1, e1, e1], [1, 1])

        # Mismatching problem dimensions
        self.assertRaises(
            ValueError, pints.SumOfErrors, [e1, e1, e3], [1, 2, 3])

        # Single-output derivatives
        model = pints.toy.ConstantModel(1)
        times = [1, 2, 3]
        p1 = pints.SingleOutputProblem(model, times, [1, 1, 1])
        p2 = pints.SingleOutputProblem(model, times, [2, 2, 2])
        e1 = pints.SumOfSquaresError(p1)
        e2 = pints.SumOfSquaresError(p2)
        e = pints.SumOfErrors([e1, e2], [1, 2])
        x = [4]
        y, dy = e.evaluateS1(x)
        self.assertEqual(y, e(x))
        self.assertEqual(dy.shape, (1, ))
        y1, dy1 = e1.evaluateS1(x)
        y2, dy2 = e2.evaluateS1(x)
        self.assertTrue(np.all(dy == dy1 + 2 * dy2))

        # Multi-output derivatives
        model = pints.toy.ConstantModel(2)
        times = [1, 2, 3]
        p1 = pints.MultiOutputProblem(model, times, [[3, 2], [1, 7], [3, 2]])
        p2 = pints.MultiOutputProblem(model, times, [[2, 3], [3, 4], [5, 6]])
        e1 = pints.SumOfSquaresError(p1)
        e2 = pints.SumOfSquaresError(p2)
        e = pints.SumOfErrors([e1, e2], [1, 2])
        x = [4, -2]
        y, dy = e.evaluateS1(x)
        self.assertEqual(y, e(x))
        self.assertEqual(dy.shape, (2, ))
        y1, dy1 = e1.evaluateS1(x)
        y2, dy2 = e2.evaluateS1(x)
        self.assertTrue(np.all(dy == dy1 + 2 * dy2))


class TestSumOfSquaresError(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(2)

        # Generate test data
        cls.times = [1, 2, 3]
        cls.n_times = len(cls.times)
        cls.data_single = np.array([1, 1, 1])
        cls.data_multi = np.array([
            [1, 4],
            [1, 4],
            [1, 4]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.SumOfSquaresError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = sum((input - 1) ** 2)
        self.assertEqual(score, 12)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.SumOfSquaresError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = sum((input - 1) ** 2)
        self.assertEqual(score, 12)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.SumOfSquaresError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score = error(test_parameters)

        # Check that error returns expected value
        # Expected = sum((input - 1) ** 2)
        self.assertEqual(score, 12)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create error measure
        error = pints.SumOfSquaresError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3, 4]
        score = error(test_parameters)

        # Check that error returns expected value
        # Exp = sum((input[0] - 1) ** 2) + sum((2 * input[1] - 4) ** 2)
        self.assertEqual(score, 60)

    def test_call_two_dim_array_multi_weighted(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create error measure with weighted input
        weights = [1, 2]
        error = pints.SumOfSquaresError(problem, weights=weights)

        # Evaluate likelihood for test parameters
        test_parameters = [3, 4]
        score = error(test_parameters)

        # Check that error returns expected value
        # Exp = weight[0] * sum((input[0] - 1) ** 2) +
        # weight[1] * sum((2 * input[1] - 4) ** 2)
        self.assertEqual(score, 108)

    def test_evaluateS1_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.SumOfSquaresError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (1,))

        # Check that partials are correct
        # Expected = 2 * sum(input - 1)
        self.assertEqual(deriv, 2 * 3 * (test_parameters[0] - 1))

    def test_evaluateS1_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.SumOfSquaresError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (1,))

        # Check that partials are correct
        # Expected = 2 * sum(input - 1)
        self.assertEqual(deriv, 2 * 3 * (test_parameters[0] - 1))

    def test_evaluateS1_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create error measure
        error = pints.SumOfSquaresError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (1,))

        # Check that partials are correct
        # Expected = 2 * sum(input - 1)
        self.assertEqual(deriv, 2 * 3 * (test_parameters[0] - 1))

    def test_evaluateS1_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create error measure
        error = pints.SumOfSquaresError(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [3, 4]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (2,))

        # Check that partials are correct
        # Expectation = [2 * sum(input[0] - 1), 4 * sum(2 * input[1] - 4)]
        self.assertEqual(deriv[0], 2 * 3 * (test_parameters[0] - 1))
        self.assertEqual(deriv[1], 4 * 3 * (2 * test_parameters[1] - 4))

    def test_evaluateS1_two_dim_array_multi_weighted(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create error measure with weighted inputs
        weights = [1, 2]
        error = pints.SumOfSquaresError(problem, weights=weights)

        # Evaluate likelihood for test parameters
        test_parameters = [3, 4]
        score, deriv = error.evaluateS1(test_parameters)

        # Check that returned error is correct
        self.assertEqual(score, error(test_parameters))

        # Check that partial derivatives are returned for each parameter
        self.assertEqual(deriv.shape, (2,))

        # Check that partials are correct
        # Expectation = [weight [0] * 2 * sum(input[0] - 1),
        # weight[1] * 4 * sum(2 * input[1] - 4)]
        self.assertEqual(
            deriv[0], weights[0] * 2 * 3 * (test_parameters[0] - 1))
        self.assertEqual(
            deriv[1], weights[1] * 4 * 3 * (2 * test_parameters[1] - 4))

    def test_bad_constructor(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_single, self.times, self.data_single)

        # Test invalid weight shape
        weights = [1, 2, 3]
        self.assertRaisesRegex(
            ValueError,
            'Number of weights must match number of problem outputs.',
            pints.SumOfSquaresError, problem, weights)


if __name__ == '__main__':
    unittest.main()
