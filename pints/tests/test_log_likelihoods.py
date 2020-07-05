#!/usr/bin/env python3
#
# Tests the log likelihood classes.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import pints.toy
import numpy as np


class TestAR1LogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(4)

        # Generate test data
        cls.times = np.asarray([1, 2, 3])
        cls.n_times = len(cls.times)
        cls.data_single = np.asarray([1.0, -10.7, 15.5])
        cls.data_multi = np.asarray([
            [3.5, 7.6, 8.5, 3.4],
            [1.1, -10.3, 15.6, 5.5],
            [-10, -30.5, -5, 7.6]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.AR1LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.5, 5]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -19.706737485492436)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.AR1LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.5, 5]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -19.706737485492436)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.AR1LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.5, 5]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -19.706737485492436)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.AR1LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            0, 0, 0, 0, 0.5, 1.0, -0.25, 3.0, 0.9, 10.0, 0.0, 2.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -179.22342804581092)


class TestARMA11LogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(4)

        # Generate test data
        cls.times = np.asarray([1, 2, 3, 4])
        cls.n_times = len(cls.times)
        cls.data_single = np.asarray([3, -4.5, 10.5, 0.3])
        cls.data_multi = np.asarray([
            [3.5, 7.6, 8.5, 3.4],
            [1.1, -10.3, 15.6, 5.5],
            [-10, -30.5, -5, 7.6],
            [-12, -10.1, -4, 2.3]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ARMA11LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.9, -0.4, 1]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -171.53031588534171)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ARMA11LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.9, -0.4, 1]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -171.53031588534171)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ARMA11LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.9, -0.4, 1]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -171.53031588534171)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.ARMA11LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            0, 0, 0, 0, 0.5, 0.34, 1.0, -0.25, 0.1, 3.0, 0.9, 0.0, 10.0, 0.0,
            0.9, 2.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -214.17034137601107)


class TestCauchyLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(4)

        # Generate test data
        cls.times = np.asarray([1, 2, 3])
        cls.n_times = len(cls.times)
        cls.data_single = np.asarray([1.0, -10.7, 15.5])
        cls.data_multi = np.asarray([
            [3.5, 7.6, 8.5, 3.4],
            [1.1, -10.3, 15.6, 5.5],
            [-10, -30.5, -5, 7.6]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 10]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -12.339498654173603)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 10]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -12.339498654173603)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 10]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -12.339498654173603)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            0, 0, 0, 0, 13, 8, 13.5, 10.5]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertEqual(score, -49.51182454195375)


class TestScaledLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.LogisticModel()
        cls.model_multi = pints.toy.ConstantModel(2)

        # Generate test data
        cls.n_times = 10
        cls.times = np.linspace(0, 1000, cls.n_times)
        cls.data_single = cls.model_single.simulate(
            parameters=[0.015, 500], times=cls.times)
        cls.data_multi = cls.model_multi.simulate(
            parameters=[1, 2], times=cls.times)

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Evaluate likelihoods for test parameters
        test_parameters = [0.014, 501]
        score_not_scaled = log_likelihood_not_scaled(test_parameters)
        score_scaled = log_likelihood_scaled(test_parameters)

        # Check that unscaled likelihood returns expected value
        self.assertEqual(int(score_not_scaled), -1897896120)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score_scaled * self.n_times, score_not_scaled)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Evaluate likelihoods for test parameters
        test_parameters = [0.014, 501]
        score_not_scaled = log_likelihood_not_scaled(test_parameters)
        score_scaled = log_likelihood_scaled(test_parameters)

        # Check that unscaled likelihood returns expected value
        self.assertEqual(int(score_not_scaled), -1897896120)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score_scaled * self.n_times, score_not_scaled)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Evaluate likelihoods for test parameters
        test_parameters = [0.014, 501]
        score_not_scaled = log_likelihood_not_scaled(test_parameters)
        score_scaled = log_likelihood_scaled(test_parameters)

        # Check that unscaled likelihood returns expected value
        self.assertEqual(int(score_not_scaled), -1897896120)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score_scaled * self.n_times, score_not_scaled)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Evaluate likelihoods for test parameters
        test_parameters = [2, 1]
        score_not_scaled = log_likelihood_not_scaled(test_parameters)
        score_scaled = log_likelihood_scaled(test_parameters)

        # Check that unscaled likelihood returns expected value
        self.assertEqual(int(score_not_scaled), -24999880)

        # Check that scaled likelihood returns expected value
        number_model_outputs = self.model_multi.n_outputs()
        self.assertAlmostEqual(
            score_scaled * self.n_times * number_model_outputs,
            score_not_scaled)

    def test_evaluateS1_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [0.014, 501]
        score_not_scaled, deriv_not_scaled = \
            log_likelihood_not_scaled.evaluateS1(test_parameters)
        score_scaled, deriv_scaled = log_likelihood_scaled.evaluateS1(
            test_parameters)

        # Check that score is computed correctly
        self.assertEqual(score_not_scaled, log_likelihood_not_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_not_scaled.shape, (2, ))

        # Check that score is computed correctly
        self.assertEqual(score_scaled, log_likelihood_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_scaled.shape, (2, ))

        # Check that partials of likelihoods agree
        unscaled_deriv = deriv_scaled * self.n_times
        self.assertAlmostEqual(deriv_not_scaled[0], unscaled_deriv[0])
        self.assertAlmostEqual(deriv_not_scaled[1], unscaled_deriv[1])

    def test_evaluateS1_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [0.014, 501]
        score_not_scaled, deriv_not_scaled = \
            log_likelihood_not_scaled.evaluateS1(test_parameters)
        score_scaled, deriv_scaled = log_likelihood_scaled.evaluateS1(
            test_parameters)

        # Check that score is computed correctly
        self.assertEqual(score_not_scaled, log_likelihood_not_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_not_scaled.shape, (2, ))

        # Check that score is computed correctly
        self.assertEqual(score_scaled, log_likelihood_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_scaled.shape, (2, ))

        # Check that partials of likelihoods agree
        unscaled_deriv = deriv_scaled * self.n_times
        self.assertAlmostEqual(deriv_not_scaled[0], unscaled_deriv[0])
        self.assertAlmostEqual(deriv_not_scaled[1], unscaled_deriv[1])

    def test_evaluateS1_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [0.014, 501]
        score_not_scaled, deriv_not_scaled = \
            log_likelihood_not_scaled.evaluateS1(test_parameters)
        score_scaled, deriv_scaled = log_likelihood_scaled.evaluateS1(
            test_parameters)

        # Check that score is computed correctly
        self.assertEqual(score_not_scaled, log_likelihood_not_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_not_scaled.shape, (2, ))

        # Check that score is computed correctly
        self.assertEqual(score_scaled, log_likelihood_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_scaled.shape, (2, ))

        # Check that partials of likelihoods agree
        unscaled_deriv = deriv_scaled * self.n_times
        self.assertAlmostEqual(deriv_not_scaled[0], unscaled_deriv[0])
        self.assertAlmostEqual(deriv_not_scaled[1], unscaled_deriv[1])

    def test_evaluateS1_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [2, 1]
        score_not_scaled, deriv_not_scaled = \
            log_likelihood_not_scaled.evaluateS1(test_parameters)
        score_scaled, deriv_scaled = log_likelihood_scaled.evaluateS1(
            test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score_not_scaled, log_likelihood_not_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_not_scaled.shape, (2, ))

        # Check that score is computed correctly
        self.assertAlmostEqual(score_scaled, log_likelihood_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_scaled.shape, (2, ))

        # Check that partials of likelihoods agree
        number_model_outputs = self.model_multi.n_outputs()
        unscaled_deriv = deriv_scaled * self.n_times * number_model_outputs
        self.assertAlmostEqual(deriv_not_scaled[0], unscaled_deriv[0])
        self.assertAlmostEqual(deriv_not_scaled[1], unscaled_deriv[1])

    def test_bad_constructor(self):
        self.assertRaises(
            ValueError, pints.ScaledLogLikelihood, self.model_single)


class TestLogLikelihood(unittest.TestCase):

    def test_gaussian_known_sigma_log_likelihood_single(self):
        # Tests :class:`pints.GaussianKnownSigmaLogLikelihood` for instances of
        # :class:`pints.SingleOutputProblem`.

        # Known noise value checks
        model = pints.toy.ConstantModel(1)
        n_times = 10
        times = np.linspace(0, 10, n_times)
        bare_values = np.arange(10) / 5.0

        # Test Case I: values as list
        values = bare_values.tolist()
        problem = pints.SingleOutputProblem(model, times, values)

        # Check evaluation
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)
        self.assertAlmostEqual(log_likelihood([-1]), -21.999591968683927)

        # Check derivatives
        l, dl = log_likelihood.evaluateS1([3])
        self.assertAlmostEqual(l, -23.777369746461702)
        self.assertAlmostEqual(dl[0], -9.3333333333333321)
        self.assertEqual(len(dl), 1)

        # Test deprecated aliases
        l1 = pints.KnownNoiseLogLikelihood(problem, 0.1)
        self.assertIsInstance(l1, pints.GaussianKnownSigmaLogLikelihood)

        # Test Case II: values as array of shape (n_times,)
        values = np.reshape(bare_values, (n_times,))
        problem = pints.SingleOutputProblem(model, times, values)

        # Check evaluation
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)
        self.assertAlmostEqual(log_likelihood([-1]), -21.999591968683927)

        # Check derivatives
        l, dl = log_likelihood.evaluateS1([3])
        self.assertAlmostEqual(l, -23.777369746461702)
        self.assertAlmostEqual(dl[0], -9.3333333333333321)
        self.assertEqual(len(dl), 1)

        # Test deprecated aliases
        l1 = pints.KnownNoiseLogLikelihood(problem, 0.1)
        self.assertIsInstance(l1, pints.GaussianKnownSigmaLogLikelihood)

        # Test Case III: values as array of shape (n_times, 1)
        values = np.reshape(bare_values, (n_times, 1))
        problem = pints.SingleOutputProblem(model, times, values)

        # Check evaluation
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)
        self.assertAlmostEqual(log_likelihood([-1]), -21.999591968683927)

        # Check derivatives
        l, dl = log_likelihood.evaluateS1([3])
        self.assertAlmostEqual(l, -23.777369746461702)
        self.assertAlmostEqual(dl[0], -9.3333333333333321)
        self.assertEqual(len(dl), 1)

        # Test deprecated aliases
        l1 = pints.KnownNoiseLogLikelihood(problem, 0.1)
        self.assertIsInstance(l1, pints.GaussianKnownSigmaLogLikelihood)

        # Test invalid constructors
        self.assertRaises(
            ValueError, pints.GaussianKnownSigmaLogLikelihood, problem, 0)
        self.assertRaises(
            ValueError,
            pints.GaussianKnownSigmaLogLikelihood, problem, [0.1, 0.2])
        self.assertRaises(
            ValueError, pints.GaussianKnownSigmaLogLikelihood, problem, -1)

    def test_gaussian_known_sigma_log_likelihood_multi(self):
        # Tests :class:`pints.GaussianKnownSigmaLogLikelihood` for instances of
        # :class:`pints.MultiOutputProblem`.

        # Check evaluation
        model = pints.toy.ConstantModel(3)
        parameters = [0, 0, 0]
        sigma = 1
        times = [1, 2, 3, 4]
        values = [[10.7, 3.5, 3.8],
                  [1.1, 3.2, -1.4],
                  [9.3, 0.0, 4.5],
                  [1.2, -3, -10]]
        problem = pints.MultiOutputProblem(model, times, values)
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)
        # Test Gaussian_logpdf((10.7, 1.1, 9.3, 1.2)|mean=0, sigma=1) +
        #      Gaussian_logpdf((3.5, 3.2, 0.0, -3)|mean=0, sigma=1) +
        #      Gaussian_logpdf((3.8, -1.4, 4.5, -10)|mean=0, sigma=1)
        #      = -196.91...
        self.assertAlmostEqual(
            log_likelihood(parameters),
            -196.9122623984561
        )

        # Check derivatives
        l, dl = log_likelihood.evaluateS1(parameters)
        self.assertAlmostEqual(l, -196.9122623984561)
        self.assertAlmostEqual(dl[0], 22.3)
        self.assertAlmostEqual(dl[1], 2 * 3.7000000000000002)
        self.assertAlmostEqual(dl[2], -9.3)

        # Test multiple output model dimensions of sensitivities
        d = 20
        model = pints.toy.ConstantModel(d)
        parameters = [0 for i in range(d)]
        times = [1, 2, 3, 4]
        values = np.ones((len(times), d))
        problem = pints.MultiOutputProblem(model, times, values)
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)
        l = log_likelihood(parameters)
        l1, dl = log_likelihood.evaluateS1(parameters)
        self.assertEqual(len(dl), len(parameters))
        self.assertEqual(l, l1)

    def test_gaussian_log_likelihood_single(self):
        # Tests :class:`pints.GaussianLogLikelihood` for instances of
        # :class:`pints.SingleOutputProblem`.

        # Check unknown nose
        model = pints.toy.LogisticModel()
        parameters = [0.015, 500]
        sigma = 0.1
        n_times = 100
        times = np.linspace(0, 1000, n_times)
        bare_values = model.simulate(parameters, times)
        np.random.seed(42)
        bare_values += np.random.normal(0, sigma, bare_values.shape)

        # Test case I: values as list
        values = bare_values.tolist()
        problem = pints.SingleOutputProblem(model, times, values)

        # Test if known/unknown give same result
        l1 = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)
        l2 = pints.GaussianLogLikelihood(problem)
        self.assertAlmostEqual(l1(parameters), l2(parameters + [sigma]))

        # Check evaluation
        log_likelihood = pints.GaussianLogLikelihood(problem)
        self.assertAlmostEqual(
            log_likelihood(parameters + [sigma]), 96.99934128549947)

        # Test case II: values as array of shape (n_times,)
        values = np.reshape(bare_values, (n_times,))
        problem = pints.SingleOutputProblem(model, times, values)

        # Test if known/unknown give same result
        l1 = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)
        l2 = pints.GaussianLogLikelihood(problem)
        self.assertAlmostEqual(l1(parameters), l2(parameters + [sigma]))

        # Check evaluation
        log_likelihood = pints.GaussianLogLikelihood(problem)
        self.assertAlmostEqual(
            log_likelihood(parameters + [sigma]), 96.99934128549947)

        # Test case III: values as array of shape (n_times, 1)
        values = np.reshape(bare_values, (n_times, 1))
        problem = pints.SingleOutputProblem(model, times, values)

        # Test if known/unknown give same result
        l1 = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)
        l2 = pints.GaussianLogLikelihood(problem)
        self.assertAlmostEqual(l1(parameters), l2(parameters + [sigma]))

        # Check evaluation
        log_likelihood = pints.GaussianLogLikelihood(problem)
        self.assertAlmostEqual(
            log_likelihood(parameters + [sigma]), 96.99934128549947)

        # Check derivatives
        model = pints.toy.ConstantModel(1)
        n_times = 10
        times = np.linspace(0, 10, n_times)
        bare_values = np.arange(10) / 5.0

        # Test Case I: values as list
        values = bare_values.tolist()
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        l, dl = log_likelihood.evaluateS1([7, 2.0])
        self.assertAlmostEqual(l, -63.04585713764618)
        self.assertAlmostEqual(dl[0], -15.25)
        self.assertAlmostEqual(dl[1], 41.925000000000004)

        # Test deprecated alias
        l2 = pints.UnknownNoiseLogLikelihood(problem)
        self.assertIsInstance(l2, pints.GaussianLogLikelihood)

        # Test case II: values as array of shape (n_times,)
        values = np.reshape(bare_values, (n_times,))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        l, dl = log_likelihood.evaluateS1([7, 2.0])
        self.assertAlmostEqual(l, -63.04585713764618)
        self.assertAlmostEqual(dl[0], -15.25)
        self.assertAlmostEqual(dl[1], 41.925000000000004)

        # Test deprecated alias
        l2 = pints.UnknownNoiseLogLikelihood(problem)
        self.assertIsInstance(l2, pints.GaussianLogLikelihood)

        # Test case III: values as array of shape (n_times, 1)
        values = np.reshape(bare_values, (n_times, 1))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        l, dl = log_likelihood.evaluateS1([7, 2.0])
        self.assertAlmostEqual(l, -63.04585713764618)
        self.assertAlmostEqual(dl[0], -15.25)
        self.assertAlmostEqual(dl[1], 41.925000000000004)

        # Test deprecated alias
        l2 = pints.UnknownNoiseLogLikelihood(problem)
        self.assertIsInstance(l2, pints.GaussianLogLikelihood)

    def test_gaussian_log_likelihood_multi(self):
        # Tests :class:`pints.GaussianLogLikelihood` for instances of
        # :class:`pints.MultiOutputProblem`.

        # Check unknown noise
        model = pints.toy.ConstantModel(3)
        parameters = [0, 0, 0]
        times = [1, 2, 3, 4]
        values = [[10.7, 3.5, 3.8],
                  [1.1, 3.2, -1.4],
                  [9.3, 0.0, 4.5],
                  [1.2, -3, -10]]
        problem = pints.MultiOutputProblem(model, times, values)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        # Test Gaussian_logpdf((10.7, 1.1, 9.3, 1.2)|mean=0, sigma=3.5) +
        #      Gaussian_logpdf((3.5, 3.2, 0.0, -3)|mean=0, sigma=1) +
        #      Gaussian_logpdf((3.8, -1.4, 4.5, -10)|mean=0, sigma=12)
        #      = -50.5088...
        self.assertAlmostEqual(
            log_likelihood(parameters + [3.5, 1, 12]),
            -50.508848609684783
        )
        l, dl = log_likelihood.evaluateS1(parameters + [3.5, 1, 12])
        self.assertAlmostEqual(l, -50.508848609684783)
        self.assertAlmostEqual(dl[0], 1.820408163265306)
        self.assertAlmostEqual(dl[1], 2 * 3.7000000000000002)
        self.assertAlmostEqual(dl[2], 3 * -0.021527777777777774)
        self.assertAlmostEqual(dl[3], 3.6065306122448981)
        self.assertAlmostEqual(dl[4], 27.490000000000002)
        self.assertAlmostEqual(dl[5], -0.25425347222222222)

        # Test multiple output model dimensions of sensitivities
        d = 20
        model = pints.toy.ConstantModel(d)
        parameters = [0 for i in range(d)]
        times = [1, 2, 3, 4]
        values = model.simulate(parameters, times)
        org_values = np.ones((len(times), d))
        extra_params = np.ones(d).tolist()
        problem = pints.MultiOutputProblem(model, times, org_values)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        l = log_likelihood(parameters + extra_params)
        l1, dl = log_likelihood.evaluateS1(parameters + extra_params)
        self.assertTrue(np.array_equal(len(dl),
                                       len(parameters + extra_params)))
        self.assertEqual(l, l1)

    def test_gaussian_integrated_uniform_log_likelihood_single(self):
        # Tests :class:`pints.GaussianIntegratedUniformLogLikelihood` for
        # instances of :class:`pints.SingleOutputProblem`.
        model = pints.toy.ConstantModel(1)
        parameters = [0]
        times = np.asarray([1, 2, 3])
        n_times = len(times)
        model.simulate(parameters, times)
        bare_values = np.asarray([1.0, -10.7, 15.5])

        # Test Case I: values as list
        values = bare_values.tolist()
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, 2, 4)
        self.assertAlmostEqual(log_likelihood([0]), -20.441037907121299)

        # test incorrect constructors
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, -1, 2)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 0, 0)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 2, 1)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 2], [2, 3])

        # Test Case II: values as array of shape (n_times,)
        values = np.reshape(bare_values, (n_times,))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, 2, 4)
        self.assertAlmostEqual(log_likelihood([0]), -20.441037907121299)

        # test incorrect constructors
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, -1, 2)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 0, 0)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 2, 1)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 2], [2, 3])

        # Test Case III: values as array of shape (n_times, 1)
        values = np.reshape(bare_values, (n_times, 1))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, 2, 4)
        self.assertAlmostEqual(log_likelihood([0]), -20.441037907121299)

        # test incorrect constructors
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, -1, 2)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 0, 0)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 2, 1)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 2], [2, 3])

    def test_gaussian_integrated_uniform_log_likelihood_multi(self):
        # Tests :class:`pints.GaussianIntegratedUniformLogLikelihood` for
        # instances of :class:`pints.MultiOutputProblem`.
        model = pints.toy.ConstantModel(4)
        parameters = [0, 0, 0, 0]
        times = np.asarray([1, 2, 3])
        model.simulate(parameters, times)
        values = np.asarray([[3.4, 4.3, 22.0, -7.3],
                             [11.1, 12.2, 13.9, 5.0],
                             [-0.4, -12.3, -8.3, -1.2]])
        problem = pints.MultiOutputProblem(model, times, values)
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, 2, 4)
        self.assertAlmostEqual(log_likelihood(parameters), -75.443307614807225)

        # test non-equal prior limits
        model = pints.toy.ConstantModel(4)
        parameters = [0, 0, 0, 0]
        times = np.asarray([1, 2, 3])
        model.simulate(parameters, times)
        values = np.asarray([[3.4, 4.3, 22.0, -7.3],
                             [11.1, 12.2, 13.9, 5.0],
                             [-0.4, -12.3, -8.3, -1.2]])
        problem = pints.MultiOutputProblem(model, times, values)
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, [1, 0, 5, 2], [2, 4, 7, 8])
        self.assertAlmostEqual(log_likelihood(parameters), -71.62076263891457)

        # test incorrect constructors
        model = pints.toy.ConstantModel(2)
        parameters = [0, 0]
        times = np.asarray([1, 2, 3])
        model.simulate(parameters, times)
        values = [[1, 2],
                  [3, 4],
                  [5, 6]]
        problem = pints.MultiOutputProblem(model, times, values)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 2, 2)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 2, 3], [2, 4])
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 2], [2, 4, 5])
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 3], [2, 2])

    def test_student_t_log_likelihood_single(self):
        # Tests :class:`pints.StudentTLogLikelihood` for
        # instances of :class:`pints.SingleOutputProblem`.

        # Check evaluation
        model = pints.toy.ConstantModel(1)
        times = np.asarray([1, 2, 3])
        n_times = len(times)
        bare_values = np.asarray([1.0, -10.7, 15.5])

        # Test Case I: values as list
        values = bare_values.tolist()
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.StudentTLogLikelihood(problem)
        # Test Student-t_logpdf(values|mean=0, df = 3, scale = 10) = -11.74..
        self.assertAlmostEqual(log_likelihood([0, 3, 10]), -11.74010919785115)

        # Test Case II: values as array of shape (n_times,)
        values = np.reshape(bare_values, (n_times,))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.StudentTLogLikelihood(problem)
        # Test Student-t_logpdf(values|mean=0, df = 3, scale = 10) = -11.74..
        self.assertAlmostEqual(log_likelihood([0, 3, 10]), -11.74010919785115)

        # Test Case III: values as array of shape (n_times, 1)
        values = np.reshape(bare_values, (n_times, 1))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.StudentTLogLikelihood(problem)
        # Test Student-t_logpdf(values|mean=0, df = 3, scale = 10) = -11.74..
        self.assertAlmostEqual(log_likelihood([0, 3, 10]), -11.74010919785115)

    def test_student_t_log_likelihood_multi(self):
        # Multi-output test for Student-t noise log-likelihood methods

        model = pints.toy.ConstantModel(4)
        parameters = [0, 0, 0, 0]
        times = np.arange(1, 4)
        values = np.asarray([[3.5, 7.6, 8.5, 3.4],
                             [1.1, -10.3, 15.6, 5.5],
                             [-10, -30.5, -5, 7.6]])
        problem = pints.MultiOutputProblem(model, times, values)
        log_likelihood = pints.StudentTLogLikelihood(problem)
        # Test Student-t_logpdf((3.5,1.1,-10)|mean=0, df=2, scale=13) +
        #      Student-t_logpdf((7.6,-10.3,-30.5)|mean=0, df=1, scale=8) +
        #      Student-t_logpdf((8.5,15.6,-5)|mean=0, df=2.5, scale=13.5) +
        #      Student-t_logpdf((3.4,5.5,7.6)|mean=0, df=3.4, scale=10.5)
        #      = -47.83....
        self.assertAlmostEqual(
            log_likelihood(parameters + [2, 13, 1, 8, 2.5, 13.5, 3.4, 10.5]),
            -47.83720347766945)

    def test_cauchy_log_likelihood_single(self):
        # Tests :class:`pints.CauchyLogLikelihood` for
        # instances of :class:`pints.SingleOutputProblem`.

        # Check evaluation
        model = pints.toy.ConstantModel(1)
        times = np.asarray([1, 2, 3])
        n_times = len(times)
        bare_values = np.asarray([1.0, -10.7, 15.5])

        # Test Case I: values as list
        values = bare_values.tolist()
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.CauchyLogLikelihood(problem)
        # Test Cauchy_logpdf(values|mean=0, scale = 10) = -12.34..
        self.assertAlmostEqual(log_likelihood([0, 10]), -12.3394986541736)

        # Test Case II: values as array of shape (n_times,)
        values = np.reshape(bare_values, (n_times,))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.CauchyLogLikelihood(problem)
        # Test Cauchy_logpdf(values|mean=0, scale = 10) = -12.34..
        self.assertAlmostEqual(log_likelihood([0, 10]), -12.3394986541736)

        # Test Case III: values as array of shape (n_times, 1)
        values = np.reshape(bare_values, (n_times, 1))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.CauchyLogLikelihood(problem)
        # Test Cauchy_logpdf(values|mean=0, scale = 10) = -12.34..
        self.assertAlmostEqual(log_likelihood([0, 10]), -12.3394986541736)

    def test_cauchy_log_likelihood_multi(self):
        # Tests :class:`pints.CauchyLogLikelihood` for
        # instances of :class:`pints.MultiOutputProblem`.

        model = pints.toy.ConstantModel(4)
        parameters = [0, 0, 0, 0]
        times = np.arange(1, 4)
        values = np.asarray([[3.5, 7.6, 8.5, 3.4],
                             [1.1, -10.3, 15.6, 5.5],
                             [-10, -30.5, -5, 7.6]])
        problem = pints.MultiOutputProblem(model, times, values)
        log_likelihood = pints.CauchyLogLikelihood(problem)
        # Test Cauchy_logpdf((3.5,1.1,-10)|mean=0, scale=13) +
        #      Cauchy_logpdf((7.6,-10.3,-30.5)|mean=0, scale=8) +
        #      Cauchy_logpdf((8.5,15.6,-5)|mean=0, scale=13.5) +
        #      Cauchy_logpdf((3.4,5.5,7.6)|mean=0, scale=10.5)
        #      = -49.51....
        self.assertAlmostEqual(
            log_likelihood(parameters + [13, 8, 13.5, 10.5]),
            -49.51182454195375)

    def test_known_noise_gaussian_single_and_multi(self):
        # Tests the output of single-series against multi-series known noise
        # log-likelihoods.

        # Define boring 1-output and 2-output models
        class NullModel1(pints.ForwardModel):
            def n_parameters(self):
                return 1

            def simulate(self, x, times):
                return np.zeros(times.shape)

        class NullModel2(pints.ForwardModel):
            def n_parameters(self):
                return 1

            def n_outputs(self):
                return 2

            def simulate(self, x, times):
                return np.zeros((len(times), 2))

        # Create two single output problems
        times = np.arange(10)
        np.random.seed(1)
        sigma1 = 3
        sigma2 = 5
        values1 = np.random.uniform(0, sigma1, times.shape)
        values2 = np.random.uniform(0, sigma2, times.shape)
        model1d = NullModel1()
        problem1 = pints.SingleOutputProblem(model1d, times, values1)
        problem2 = pints.SingleOutputProblem(model1d, times, values2)
        log1 = pints.GaussianKnownSigmaLogLikelihood(problem1, sigma1)
        log2 = pints.GaussianKnownSigmaLogLikelihood(problem2, sigma2)

        # Create one multi output problem
        values3 = np.array([values1, values2]).swapaxes(0, 1)
        model2d = NullModel2()
        problem3 = pints.MultiOutputProblem(model2d, times, values3)
        log3 = pints.GaussianKnownSigmaLogLikelihood(
            problem3, [sigma1, sigma2])

        # Check if we get the right output
        self.assertAlmostEqual(log1(0) + log2(0), log3(0))

    def test_sum_of_independent_log_pdfs(self):

        # Test single output
        model = pints.toy.LogisticModel()
        x = [0.015, 500]
        sigma = 0.1
        times = np.linspace(0, 1000, 100)
        values = model.simulate(x, times) + 0.1
        problem = pints.SingleOutputProblem(model, times, values)

        l1 = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)
        l2 = pints.GaussianLogLikelihood(problem)
        ll = pints.SumOfIndependentLogPDFs([l1, l1, l1])
        self.assertEqual(l1.n_parameters(), ll.n_parameters())
        self.assertEqual(3 * l1(x), ll(x))

        # Test single output derivatives
        y, dy = ll.evaluateS1(x)
        self.assertEqual(y, ll(x))
        self.assertEqual(dy.shape, (2, ))
        y1, dy1 = l1.evaluateS1(x)
        self.assertTrue(np.all(3 * dy1 == dy))

        # Wrong number of arguments
        self.assertRaises(TypeError, pints.SumOfIndependentLogPDFs)
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogPDFs, [l1])

        # Wrong types
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogPDFs, [l1, 1])
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogPDFs, [problem, l1])

        # Mismatching dimensions
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogPDFs, [l1, l2])

        # Test multi-output
        model = pints.toy.FitzhughNagumoModel()
        x = model.suggested_parameters()
        nt = 10
        nx = model.n_parameters()
        times = np.linspace(0, 10, nt)
        values = model.simulate(x, times) + 0.01
        problem = pints.MultiOutputProblem(model, times, values)
        sigma = 0.01
        l1 = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)
        ll = pints.SumOfIndependentLogPDFs([l1, l1, l1])
        self.assertEqual(l1.n_parameters(), ll.n_parameters())
        self.assertEqual(3 * l1(x), ll(x))

        # Test multi-output derivatives
        y, dy = ll.evaluateS1(x)

        # Note: y and ll(x) differ a bit, because the solver acts slightly
        # different when evaluating with and without sensitivities!
        self.assertAlmostEqual(y, ll(x), places=3)

        self.assertEqual(dy.shape, (nx, ))
        y1, dy1 = l1.evaluateS1(x)
        self.assertTrue(np.all(3 * dy1 == dy))



    def test_multiplicative_gaussian_single(self):
        # Tests :class:`pints.MultiplicativeGaussianLogLikelihood` for
        # instances of :class:`pints.SingleOutputProblem`.
        model = pints.toy.ConstantModel(1)
        parameters = [2]
        times = np.asarray([1, 2, 3, 4])
        n_times = len(times)
        bare_values = np.asarray([1.9, 2.1, 1.8, 2.2])

        # Test Case I: values as list
        values = bare_values.tolist()
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)
        self.assertAlmostEqual(log_likelihood(parameters + [2.0, 1.0]),
                               -9.224056577298253)

        # Test Case II: values as array of shape (n_times,)
        values = np.reshape(bare_values, (n_times,))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)
        self.assertAlmostEqual(log_likelihood(parameters + [2.0, 1.0]),
                               -9.224056577298253)

        # Test Case III: values as array of shape (n_times, 1)
        values = np.reshape(bare_values, (n_times, 1))
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)
        self.assertAlmostEqual(log_likelihood(parameters + [2.0, 1.0]),
                               -9.224056577298253)

    def test_multiplicative_gaussian_multi(self):
        # Tests :class:`pints.MultiplicativeGaussianLogLikelihood` for
        # instances of :class:`pints.MultiOutputProblem`.
        model = pints.toy.ConstantModel(2)
        parameters = [1, 2]
        times = np.asarray([1, 2, 3])
        values = np.asarray([[1.1, 0.9, 1.5], [1.5, 2.5, 2.0]]).transpose()
        problem = pints.MultiOutputProblem(model, times, values)
        log_likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        self.assertAlmostEqual(
            log_likelihood(parameters + [1.0, 2.0, 1.0, 1.0]),
            -12.176330824267543)


if __name__ == '__main__':
    unittest.main()
