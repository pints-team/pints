#!/usr/bin/env python3
#
# Tests the log_posterior class
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import pints.toy
import numpy as np


class TestLogPosterior(unittest.TestCase):

    def test_log_posterior(self):

        # Create a toy problem and log likelihood
        model = pints.toy.LogisticModel()
        real_parameters = [0.015, 500]
        x = [0.014, 501]
        sigma = 0.001
        times = np.linspace(0, 1000, 100)
        values = model.simulate(real_parameters, times)
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)

        # Create a prior
        log_prior = pints.UniformLogPrior([0, 0], [1, 1000])

        # Test
        p = pints.LogPosterior(log_likelihood, log_prior)
        self.assertEqual(p(x), log_likelihood(x) + log_prior(x))
        y = [-1, 500]
        self.assertEqual(log_prior(y), -np.inf)
        self.assertEqual(p(y), -np.inf)
        self.assertEqual(p(y), log_prior(y))

        # Test derivatives
        log_prior = pints.ComposedLogPrior(
            pints.GaussianLogPrior(0.015, 0.3),
            pints.GaussianLogPrior(500, 100))
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        x = [0.013, 540]
        y, dy = log_posterior.evaluateS1(x)
        self.assertEqual(y, log_posterior(x))
        self.assertEqual(dy.shape, (2, ))
        y1, dy1 = log_prior.evaluateS1(x)
        y2, dy2 = log_likelihood.evaluateS1(x)
        self.assertTrue(np.all(dy == dy1 + dy2))

        # Test getting the prior and likelihood back again
        self.assertIs(log_posterior.log_prior(), log_prior)
        self.assertIs(log_posterior.log_likelihood(), log_likelihood)

        # First arg must be a LogPDF
        self.assertRaises(ValueError, pints.LogPosterior, 'hello', log_prior)

        # Second arg must be a log_prior
        self.assertRaises(
            ValueError, pints.LogPosterior, log_likelihood, log_likelihood)

        # Prior and likelihood must have same dimension
        self.assertRaises(
            ValueError, pints.LogPosterior, log_likelihood,
            pints.GaussianLogPrior(0.015, 0.3))


class TestPooledLogPDF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test log-pdfs
        model = pints.toy.ConstantModel(1)

        problem = pints.SingleOutputProblem(
            model=model, times=[1, 2, 3, 4], values=[1, 2, 3, 4])
        cls.log_pdf_1 = pints.GaussianLogLikelihood(problem)

        problem = pints.SingleOutputProblem(
            model=model, times=[1, 2, 3, 4], values=[1, 1, 1, 1])
        cls.log_pdf_2 = pints.GaussianLogLikelihood(problem)

    def test_bad_number_log_pdfs(self):
        log_pdfs = [self.log_pdf_1]
        pooled = [True, True]
        self.assertRaisesRegex(
            ValueError, 'PooledLogPDF requires', pints.PooledLogPDF, log_pdfs,
            pooled)

    def test_bad_log_pdfs_objects(self):
        log_pdfs = ['log_pdf_1', 'log_pdf_2']
        pooled = [True, True]
        self.assertRaisesRegex(
            ValueError, 'All log-pdfs passed', pints.PooledLogPDF, log_pdfs,
            pooled)

    def test_bad_log_pdfs_parameters(self):
        model = pints.toy.ConstantModel(1)
        problem = pints.SingleOutputProblem(
            model=model, times=[1, 2, 3, 4], values=[1, 2, 3, 4])
        log_pdf = pints.ConstantAndMultiplicativeGaussianLogLikelihood(problem)

        log_pdfs = [self.log_pdf_1, log_pdf]
        pooled = [True, True]

        self.assertRaisesRegex(
            ValueError, 'All log-pdfs passed to PooledLogPDFs',
            pints.PooledLogPDF, log_pdfs, pooled)

    def test_bad_pooled_length(self):
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [True, True, True]
        self.assertRaisesRegex(
            ValueError, 'The array-like input `pooled` needs',
            pints.PooledLogPDF, log_pdfs, pooled)

    def test_bad_pooled_content(self):
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [True, 'Yes']
        self.assertRaisesRegex(
            ValueError, 'The array-like input `pooled` passed',
            pints.PooledLogPDF, log_pdfs, pooled)

    def test_n_parameters(self):
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]

        # Pool nothing
        pooled = [False, False]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        n_parameters = \
            self.log_pdf_1.n_parameters() + self.log_pdf_2.n_parameters()
        self.assertEqual(log_pdf.n_parameters(), n_parameters)

        # Pool first parameter
        pooled = [True, False]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        n_parameters = 2 * 1 + 1
        self.assertEqual(log_pdf.n_parameters(), n_parameters)

        # Pool second parameter
        pooled = [False, True]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        n_parameters = 2 * 1 + 1
        self.assertEqual(log_pdf.n_parameters(), n_parameters)

        # Pool both parameters
        pooled = [True, True]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        n_parameters = 2 * 1
        self.assertEqual(log_pdf.n_parameters(), n_parameters)

    def test_call_unpooled(self):
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [False, False]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        param_1 = [1, 0.2]
        param_2 = [2, 0.1]

        score = self.log_pdf_1(param_1) + self.log_pdf_2(param_2)
        self.assertEqual(log_pdf(param_1 + param_2), score)

    def test_call_partially_pooled(self):
        # Pool first parameter
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [True, False]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        param_1 = [1, 0.2]
        param_2 = [1, 0.1]
        param = [0.2, 0.1, 1]

        score = self.log_pdf_1(param_1) + self.log_pdf_2(param_2)
        self.assertEqual(log_pdf(param), score)

        # Pool second parameter
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [False, True]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        param_1 = [1, 0.1]
        param_2 = [2, 0.1]
        param = [1, 2, 0.1]

        score = self.log_pdf_1(param_1) + self.log_pdf_2(param_2)
        self.assertEqual(log_pdf(param), score)

    def test_call_pooled(self):
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [True, True]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        param = [1, 0.1]

        score = self.log_pdf_1(param) + self.log_pdf_2(param)
        self.assertEqual(log_pdf(param), score)

    def test_evaluateS1_unpooled(self):
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [False, False]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        param_1 = [1, 0.2]
        param_2 = [2, 0.1]

        score_1, dscore_1 = self.log_pdf_1.evaluateS1(param_1)
        score_2, dscore_2 = self.log_pdf_2.evaluateS1(param_2)

        score, dscore = log_pdf.evaluateS1(param_1 + param_2)
        self.assertEqual(score, score_1 + score_2)

        self.assertEqual(len(dscore), 4)
        self.assertEqual(dscore[0], dscore_1[0])
        self.assertEqual(dscore[1], dscore_1[1])
        self.assertEqual(dscore[2], dscore_2[0])
        self.assertEqual(dscore[3], dscore_2[1])

    def test_evaluateS1_partially_pooled(self):
        # Pool first parameter
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [True, False]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        param_1 = [1, 0.2]
        param_2 = [1, 0.1]
        param = [0.2, 0.1, 1]

        score_1, dscore_1 = self.log_pdf_1.evaluateS1(param_1)
        score_2, dscore_2 = self.log_pdf_2.evaluateS1(param_2)

        score, dscore = log_pdf.evaluateS1(param)
        self.assertEqual(score, score_1 + score_2)

        self.assertEqual(len(dscore), 3)
        self.assertEqual(dscore[0], dscore_1[1])
        self.assertEqual(dscore[1], dscore_2[1])
        self.assertEqual(dscore[2], dscore_1[0] + dscore_2[0])

        # Pool second parameter
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [False, True]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        param_1 = [1, 0.1]
        param_2 = [2, 0.1]
        param = [1, 2, 0.1]

        score_1, dscore_1 = self.log_pdf_1.evaluateS1(param_1)
        score_2, dscore_2 = self.log_pdf_2.evaluateS1(param_2)

        score, dscore = log_pdf.evaluateS1(param)
        self.assertEqual(score, score_1 + score_2)

        self.assertEqual(len(dscore), 3)
        self.assertEqual(dscore[0], dscore_1[0])
        self.assertEqual(dscore[1], dscore_2[0])
        self.assertEqual(dscore[2], dscore_1[1] + dscore_2[1])

    def test_evaluateS1_pooled(self):
        log_pdfs = [self.log_pdf_1, self.log_pdf_2]
        pooled = [True, True]
        log_pdf = pints.PooledLogPDF(log_pdfs, pooled)

        param_1 = [1, 0.2]
        param_2 = [1, 0.2]
        param = [1, 0.2]

        score_1, dscore_1 = self.log_pdf_1.evaluateS1(param_1)
        score_2, dscore_2 = self.log_pdf_2.evaluateS1(param_2)

        score, dscore = log_pdf.evaluateS1(param)
        self.assertEqual(score, score_1 + score_2)

        self.assertEqual(len(dscore), 2)
        self.assertEqual(dscore[0], dscore_1[0] + dscore_2[0])
        self.assertEqual(dscore[1], dscore_1[1] + dscore_2[1])


class TestSumOfIndependentLogPDFs(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
