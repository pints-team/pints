#!/usr/bin/env python3
#
# Tests the log likelihood classes.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import pints.toy
import numpy as np


class TestLogLikelihood(unittest.TestCase):

    def test_scaled_log_likelihood(self):

        model = pints.toy.LogisticModel()
        real_parameters = [0.015, 500]
        test_parameters = [0.014, 501]
        sigma = 0.001
        times = np.linspace(0, 1000, 100)
        values = model.simulate(real_parameters, times)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.KnownNoiseLogLikelihood(
            problem, sigma)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        eval_not_scaled = log_likelihood_not_scaled(test_parameters)
        eval_scaled = log_likelihood_scaled(test_parameters)

        self.assertEqual(int(eval_not_scaled), -20959169232)
        self.assertAlmostEqual(eval_scaled * len(times), eval_not_scaled)

        # Test bad constructor
        self.assertRaises(ValueError, pints.ScaledLogLikelihood, model)

        # Test single-output derivatives
        y1, dy1 = log_likelihood_not_scaled.evaluateS1(test_parameters)
        y2, dy2 = log_likelihood_scaled.evaluateS1(test_parameters)
        self.assertEqual(y1, log_likelihood_not_scaled(test_parameters))
        self.assertEqual(dy1.shape, (2, ))
        self.assertEqual(y2, log_likelihood_scaled(test_parameters))
        self.assertEqual(dy2.shape, (2, ))
        dy3 = dy2 * len(times)
        self.assertAlmostEqual(dy1[0] / dy3[0], 1)
        self.assertAlmostEqual(dy1[1] / dy3[1], 1)

        # Test on multi-output problem
        model = pints.toy.FitzhughNagumoModel()
        nt = 10
        no = model.n_outputs()
        times = np.linspace(0, 100, nt)
        values = model.simulate([0.5, 0.5, 0.5], times)
        problem = pints.MultiOutputProblem(model, times, values)
        unscaled = pints.KnownNoiseLogLikelihood(problem, 1)
        scaled = pints.ScaledLogLikelihood(unscaled)
        p = [0.1, 0.1, 0.1]
        x = unscaled(p)
        y = scaled(p)
        self.assertAlmostEqual(y, x / nt / no)

        # Test multi-output derivatives
        y1, dy1 = unscaled.evaluateS1(p)
        y2, dy2 = scaled.evaluateS1(p)
        self.assertAlmostEqual(y1, unscaled(p))
        self.assertEqual(dy1.shape, (3, ))
        self.assertAlmostEqual(y2, scaled(p))
        self.assertEqual(dy2.shape, (3, ))
        dy3 = dy2 * nt * no
        self.assertAlmostEqual(dy1[0] / dy3[0], 1)
        self.assertAlmostEqual(dy1[1] / dy3[1], 1)

    def test_known_and_unknown_noise_single(self):
        """
        Single-output test for known/unknown noise log-likelihood methods
        """
        model = pints.toy.LogisticModel()
        parameters = [0.015, 500]
        sigma = 0.1
        times = np.linspace(0, 1000, 100)
        values = model.simulate(parameters, times)
        values += np.random.normal(0, sigma, values.shape)
        problem = pints.SingleOutputProblem(model, times, values)

        # Test if known/unknown give same result
        l1 = pints.KnownNoiseLogLikelihood(problem, sigma)
        l2 = pints.UnknownNoiseLogLikelihood(problem)
        self.assertAlmostEqual(l1(parameters), l2(parameters + [sigma]))

        # Test invalid constructors
        self.assertRaises(
            ValueError, pints.KnownNoiseLogLikelihood, problem, 0)
        self.assertRaises(
            ValueError, pints.KnownNoiseLogLikelihood, problem, -1)

        # known noise value checks
        model = pints.toy.ConstantModel(1)
        times = np.linspace(0, 10, 10)
        values = model.simulate([2], times)
        org_values = np.arange(10) / 5.0
        problem = pints.SingleOutputProblem(model, times, org_values)
        log_likelihood = pints.KnownNoiseLogLikelihood(problem, 1.5)
        self.assertAlmostEqual(log_likelihood([-1]), -21.999591968683927)
        l, dl = log_likelihood.evaluateS1([3])
        self.assertAlmostEqual(l, -23.777369746461702)
        self.assertAlmostEqual(dl[0], -9.3333333333333321)

        # unknown noise value checks
        log_likelihood = pints.UnknownNoiseLogLikelihood(problem)
        self.assertAlmostEqual(log_likelihood([-3, 1.5]), -47.777369746461702)

        # unknown noise check sensitivity
        model = pints.toy.ConstantModel(1)
        times = np.linspace(0, 10, 10)
        values = model.simulate([2], times)
        org_values = np.arange(10) / 5.0
        problem = pints.SingleOutputProblem(model, times, org_values)
        log_likelihood = pints.UnknownNoiseLogLikelihood(problem)
        l, dl = log_likelihood.evaluateS1([7, 2.0])
        self.assertAlmostEqual(l, -63.04585713764618)
        self.assertAlmostEqual(dl[0], -15.25)
        self.assertAlmostEqual(dl[1], 41.925000000000004)

    def test_known_noise_single_S1(self):
        """
        Simple tests for single known noise log-likelihood with sensitivities.
        """
        model = pints.toy.LogisticModel()
        x = [0.015, 500]
        sigma = 0.1
        times = np.linspace(0, 1000, 100)
        values = model.simulate(x, times)
        values += np.random.normal(0, sigma, values.shape)
        problem = pints.SingleOutputProblem(model, times, values)

        # Test if values are correct
        f = pints.KnownNoiseLogLikelihood(problem, sigma)
        L1 = f(x)
        L2, dL = f.evaluateS1(x)
        self.assertEqual(L1, L2)
        self.assertEqual(dL.shape, (2,))

        # Test with MultiOutputProblem
        problem = pints.MultiOutputProblem(model, times, values)
        f2 = pints.KnownNoiseLogLikelihood(problem, sigma)
        L3 = f2(x)
        L4, dL = f2.evaluateS1(x)
        self.assertEqual(L3, L4)
        self.assertEqual(L1, L3)
        self.assertEqual(dL.shape, (2,))

        # Test without noise
        values = model.simulate(x, times)
        problem = pints.SingleOutputProblem(model, times, values)
        f = pints.KnownNoiseLogLikelihood(problem, sigma)
        L1 = f(x)
        L2, dL = f.evaluateS1(x)
        self.assertEqual(L1, L2)
        self.assertEqual(dL.shape, (2,))

        # Test if zero at optimum
        self.assertTrue(np.all(dL == 0))

        # Test if positive to the left, negative to the right
        L, dL = f.evaluateS1(x + np.array([-1e-9, 0]))
        self.assertTrue(dL[0] > 0)
        L, dL = f.evaluateS1(x + np.array([1e-9, 0]))
        self.assertTrue(dL[0] < 0)

        # Test if positive to the left, negative to the right
        L, dL = f.evaluateS1(x + np.array([0, -1e-9]))
        self.assertTrue(dL[1] > 0)
        L, dL = f.evaluateS1(x + np.array([0, 1e-9]))
        self.assertTrue(dL[1] < 0)

        # Plot derivatives
        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            r = np.linspace(x[0] * 0.95, x[0] * 1.05, 100)
            L = []
            dL1 = []
            dL2 = []
            for y in r:
                a, b = f.evaluateS1([y, x[1]])
                L.append(a)
                dL1.append(b[0])
                dL2.append(b[1])
            plt.subplot(3, 1, 1)
            plt.plot(r, L)
            plt.subplot(3, 1, 2)
            plt.plot(r, dL1)
            plt.grid(True)
            plt.subplot(3, 1, 3)
            plt.plot(r, dL2)
            plt.grid(True)

            plt.figure()
            r = np.linspace(x[1] * 0.95, x[1] * 1.05, 100)
            L = []
            dL1 = []
            dL2 = []
            for y in r:
                a, b = f.evaluateS1([x[0], y])
                L.append(a)
                dL1.append(b[0])
                dL2.append(b[1])
            plt.subplot(3, 1, 1)
            plt.plot(r, L)
            plt.subplot(3, 1, 2)
            plt.plot(r, dL1)
            plt.grid(True)
            plt.subplot(3, 1, 3)
            plt.plot(r, dL2)
            plt.grid(True)

            plt.show()

    def test_student_t_log_likelihood_single(self):
        """
        Single-output test for Student-t noise log-likelihood methods
        """
        model = pints.toy.ConstantModel(1)
        parameters = [0]
        times = np.asarray([1, 2, 3])
        model.simulate(parameters, times)
        values = np.asarray([1.0, -10.7, 15.5])
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.StudentTLogLikelihood(problem)
        # Test Student-t_logpdf(values|mean=0, df = 3, scale = 10) = -11.74..
        self.assertAlmostEqual(log_likelihood([0, 3, 10]), -11.74010919785115)

    def test_student_t_log_likelihood_multi(self):
        """
        Multi-output test for Student-t noise log-likelihood methods
        """
        model = pints.toy.ConstantModel(4)
        parameters = [0, 0, 0, 0]
        times = np.arange(1, 4)
        model.simulate(parameters, times)
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
        """
        Single-output test for Cauchy noise log-likelihood methods
        """
        model = pints.toy.ConstantModel(1)
        parameters = [0]
        times = np.asarray([1, 2, 3])
        model.simulate(parameters, times)
        values = np.asarray([1.0, -10.7, 15.5])
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.CauchyLogLikelihood(problem)
        # Test Cauchy_logpdf(values|mean=0, scale = 10) = -12.34..
        self.assertAlmostEqual(log_likelihood([0, 10]), -12.3394986541736)

    def test_cauchy_log_likelihood_multi(self):
        """
        Multi-output test for Cauchy noise log-likelihood methods
        """
        model = pints.toy.ConstantModel(4)
        parameters = [0, 0, 0, 0]
        times = np.arange(1, 4)
        model.simulate(parameters, times)
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

    def test_known_and_unknown_noise_multi(self):
        """
        Multi-output test for known/unknown noise log-likelihood methods
        """
        model = pints.toy.FitzhughNagumoModel()
        parameters = [0.5, 0.5, 0.5]
        sigma = 0.1
        times = np.linspace(0, 100, 100)
        values = model.simulate(parameters, times)
        values += np.random.normal(0, sigma, values.shape)
        problem = pints.MultiOutputProblem(model, times, values)

        # Test if known/unknown give same result
        l1 = pints.KnownNoiseLogLikelihood(problem, sigma)
        l2 = pints.KnownNoiseLogLikelihood(problem, [sigma, sigma])
        l3 = pints.UnknownNoiseLogLikelihood(problem)
        self.assertAlmostEqual(
            l1(parameters),
            l2(parameters),
            l3(parameters + [sigma, sigma]))

        # Test invalid constructors
        self.assertRaises(
            ValueError, pints.KnownNoiseLogLikelihood, problem, 0)
        self.assertRaises(
            ValueError, pints.KnownNoiseLogLikelihood, problem, -1)
        self.assertRaises(
            ValueError, pints.KnownNoiseLogLikelihood, problem, [1])
        self.assertRaises(
            ValueError, pints.KnownNoiseLogLikelihood, problem, [1, 2, 3, 4])
        self.assertRaises(
            ValueError, pints.KnownNoiseLogLikelihood, problem, [1, 2, -3])

    def test_known_noise_single_and_multi(self):
        """
        Tests the output of single-series against multi-series known noise
        log-likelihoods.
        """

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
        log1 = pints.KnownNoiseLogLikelihood(problem1, sigma1)
        log2 = pints.KnownNoiseLogLikelihood(problem2, sigma2)

        # Create one multi output problem
        values3 = np.array([values1, values2]).swapaxes(0, 1)
        model2d = NullModel2()
        problem3 = pints.MultiOutputProblem(model2d, times, values3)
        log3 = pints.KnownNoiseLogLikelihood(
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

        l1 = pints.KnownNoiseLogLikelihood(problem, sigma)
        l2 = pints.UnknownNoiseLogLikelihood(problem)
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
        l1 = pints.KnownNoiseLogLikelihood(problem, sigma)
        ll = pints.SumOfIndependentLogPDFs([l1, l1, l1])
        self.assertEqual(l1.n_parameters(), ll.n_parameters())
        self.assertEqual(3 * l1(x), ll(x))

        # Test multi-output derivatives
        y, dy = ll.evaluateS1(x)

        # Note: y and ll(x) differ a bit, because the solver acts slightly
        # different when evaluating with and without sensitivities!
        self.assertTrue(np.abs(y - ll(x) < 1e-6))

        self.assertEqual(dy.shape, (nx, ))
        y1, dy1 = l1.evaluateS1(x)
        self.assertTrue(np.all(3 * dy1 == dy))


if __name__ == '__main__':
    unittest.main()
