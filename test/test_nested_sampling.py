#!/usr/bin/env python2
#
# Tests the basic methods of the adaptive covariance MCMC routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import re
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture, TemporaryDirectory

debug = False


class TestNestedRejectionSampler(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`NestedRejectionSampler`.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare for the test. """
        # Create toy model
        model = pints.toy.LogisticModel()
        cls.real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)
        values = model.simulate(cls.real_parameters, times)

        # Add noise
        np.random.seed(1)
        cls.noise = 10
        values += np.random.normal(0, cls.noise, values.shape)
        cls.real_parameters.append(cls.noise)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.log_prior = pints.UniformLogPrior(
            [0.01, 400],
            [0.02, 600]
        )

        # Create a log-likelihood
        cls.log_likelihood = pints.KnownNoiseLogLikelihood(problem, cls.noise)

    def test_quick_run(self):
        """ Test a single run. """

        sampler = pints.NestedRejectionSampler(
            self.log_likelihood, self.log_prior)
        sampler.set_posterior_samples(10)
        sampler.set_iterations(50)
        sampler.set_active_points_rate(50)
        sampler.set_log_to_screen(False)
        samples, margin = sampler.run()
        # Check output: Note n returned samples = n posterior samples
        self.assertEqual(samples.shape, (10, 2))

    def test_logging(self):
        """ Tests logging to screen and file. """

        # No logging
        with StreamCapture() as c:
            sampler = pints.NestedRejectionSampler(
                self.log_likelihood, self.log_prior)
            sampler.set_posterior_samples(2)
            sampler.set_iterations(10)
            sampler.set_active_points_rate(10)
            sampler.set_log_to_screen(False)
            samples, margin = sampler.run()
        self.assertEqual(c.text(), '')

        # Log to screen
        with StreamCapture() as c:
            sampler = pints.NestedRejectionSampler(
                self.log_likelihood, self.log_prior)
            sampler.set_posterior_samples(2)
            sampler.set_iterations(10)
            sampler.set_active_points_rate(10)
            sampler.set_log_to_screen(True)
            samples, margin = sampler.run()
        lines = c.text().splitlines()
        self.assertEqual(len(lines), 25)
        self.assertEqual(lines[0], 'Running nested rejection sampling')
        self.assertEqual(lines[1], 'Number of active points: 10')
        self.assertEqual(lines[2], 'Total number of iterations: 10')
        self.assertEqual(lines[3], 'Total number of posterior samples: 2')
        self.assertEqual(lines[4], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))

        # Log to file
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                sampler = pints.NestedRejectionSampler(
                    self.log_likelihood, self.log_prior)
                sampler.set_posterior_samples(2)
                sampler.set_iterations(10)
                sampler.set_active_points_rate(10)
                sampler.set_log_to_screen(False)
                sampler.set_log_to_file(filename)
                samples, margin = sampler.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
            self.assertEqual(c.text(), '')
        self.assertEqual(len(lines), 21)
        self.assertEqual(lines[0], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))


class TestNestedEllipsoidSampler(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`NestedEllipsoidSampler`.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare for the test. """
        # Create toy model
        model = pints.toy.LogisticModel()
        cls.real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)
        values = model.simulate(cls.real_parameters, times)

        # Add noise
        np.random.seed(1)
        cls.noise = 10
        values += np.random.normal(0, cls.noise, values.shape)
        cls.real_parameters.append(cls.noise)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.log_prior = pints.UniformLogPrior(
            [0.01, 400],
            [0.02, 600]
        )

        # Create a log-likelihood
        cls.log_likelihood = pints.KnownNoiseLogLikelihood(problem, cls.noise)

    def test_quick(self):
        """ Test a single run. """

        sampler = pints.NestedEllipsoidSampler(
            self.log_likelihood, self.log_prior)

        sampler.set_posterior_samples(10)
        sampler.set_rejection_samples(20)
        sampler.set_iterations(50)
        sampler.set_active_points_rate(50)
        sampler.set_log_to_screen(False)
        samples, margin = sampler.run()
        # Check output: Note n returned samples = n posterior samples
        self.assertEqual(samples.shape, (10, 2))

    def test_logging(self):
        """ Tests logging to screen and file. """

        # No logging
        with StreamCapture() as c:
            sampler = pints.NestedEllipsoidSampler(
                self.log_likelihood, self.log_prior)
            sampler.set_posterior_samples(2)
            sampler.set_rejection_samples(5)
            sampler.set_iterations(10)
            sampler.set_active_points_rate(10)
            sampler.set_log_to_screen(False)
            samples, margin = sampler.run()
        self.assertEqual(c.text(), '')

        # Log to screen
        with StreamCapture() as c:
            sampler = pints.NestedEllipsoidSampler(
                self.log_likelihood, self.log_prior)
            sampler.set_posterior_samples(2)
            sampler.set_rejection_samples(5)
            sampler.set_iterations(10)
            sampler.set_active_points_rate(10)
            sampler.set_log_to_screen(True)
            samples, margin = sampler.run()
        lines = c.text().splitlines()
        self.assertEqual(len(lines), 26)
        self.assertEqual(lines[0], 'Running nested rejection sampling')
        self.assertEqual(lines[1], 'Number of active points: 10')
        self.assertEqual(lines[2], 'Total number of iterations: 10')
        self.assertEqual(lines[3], 'Enlargement factor: 1.5')
        self.assertEqual(lines[4], 'Total number of posterior samples: 2')
        self.assertEqual(lines[5], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[6:]:
            self.assertTrue(pattern.match(line))

        # Log to file
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                sampler = pints.NestedEllipsoidSampler(
                    self.log_likelihood, self.log_prior)
                sampler.set_posterior_samples(2)
                sampler.set_rejection_samples(5)
                sampler.set_iterations(10)
                sampler.set_active_points_rate(10)
                sampler.set_log_to_screen(False)
                sampler.set_log_to_file(filename)
                samples, margin = sampler.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
            self.assertEqual(c.text(), '')
        self.assertEqual(len(lines), 21)
        self.assertEqual(lines[0], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))


#TODO: Test remaining methods, errors, etc.

if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
