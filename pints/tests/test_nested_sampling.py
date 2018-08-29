#!/usr/bin/env python
#
# Tests the basic methods of the nested sampling routines.
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

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

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

    def test_construction_errors(self):
        """ Tests if invalid constructor calls are picked up. """

        # First arg must be a log likelihood
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogLikelihood',
            pints.NestedRejectionSampler, 'hello', self.log_prior)

        # First arg must be a log prior
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.NestedRejectionSampler,
            self.log_likelihood, self.log_likelihood)

        # Both must have same number of parameters
        log_prior = pints.UniformLogPrior([0.01, 400, 1], [0.02, 600, 3])
        self.assertRaisesRegex(
            ValueError, 'same number of parameters',
            pints.NestedRejectionSampler, self.log_likelihood, log_prior)

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
            sampler.set_log_to_file(False)
            samples, margin = sampler.run()
        self.assertEqual(c.text(), '')

        # Log to screen
        with StreamCapture() as c:
            sampler = pints.NestedRejectionSampler(
                self.log_likelihood, self.log_prior)
            sampler.set_posterior_samples(2)
            sampler.set_iterations(20)
            sampler.set_active_points_rate(10)
            sampler.set_log_to_screen(True)
            sampler.set_log_to_file(False)
            samples, margin = sampler.run()
        lines = c.text().splitlines()
        self.assertEqual(lines[0], 'Running nested rejection sampling')
        self.assertEqual(lines[1], 'Number of active points: 10')
        self.assertEqual(lines[2], 'Total number of iterations: 20')
        self.assertEqual(lines[3], 'Total number of posterior samples: 2')
        self.assertEqual(lines[4], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))
        self.assertEqual(len(lines), 11)

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
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))

    def test_settings_check(self):
        """
        Tests the settings check at the start of a run.
        """
        sampler = pints.NestedRejectionSampler(
            self.log_likelihood, self.log_prior)
        sampler.set_posterior_samples(2)
        sampler.set_iterations(10)
        sampler.set_active_points_rate(10)
        sampler.set_log_to_screen(False)
        sampler.run()

        sampler.set_posterior_samples(10)
        self.assertRaisesRegex(ValueError, 'exceed 0.25', sampler.run)

    def test_getters_and_setters(self):
        """
        Tests various get() and set() methods.
        """
        sampler = pints.NestedRejectionSampler(
            self.log_likelihood, self.log_prior)

        # Iterations
        x = sampler.iterations() + 1
        self.assertNotEqual(sampler.iterations(), x)
        sampler.set_iterations(x)
        self.assertEqual(sampler.iterations(), x)
        self.assertRaisesRegex(
            ValueError, 'negative', sampler.set_iterations, -1)

        # Active points rate
        x = sampler.active_points_rate() + 1
        self.assertNotEqual(sampler.active_points_rate(), x)
        sampler.set_active_points_rate(x)
        self.assertEqual(sampler.active_points_rate(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than 5', sampler.set_active_points_rate, 5)

        # Posterior samples
        x = sampler.posterior_samples() + 1
        self.assertNotEqual(sampler.posterior_samples(), x)
        sampler.set_posterior_samples(x)
        self.assertEqual(sampler.posterior_samples(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than zero', sampler.set_posterior_samples, 0)


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

    def test_construction_errors(self):
        """ Tests if invalid constructor calls are picked up. """

        # First arg must be a log likelihood
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogLikelihood',
            pints.NestedEllipsoidSampler, 'hiya', self.log_prior)

        # First arg must be a log prior
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.NestedEllipsoidSampler,
            self.log_likelihood, self.log_likelihood)

        # Both must have same number of parameters
        log_prior = pints.UniformLogPrior([0.01, 400, 1], [0.02, 600, 3])
        self.assertRaisesRegex(
            ValueError, 'same number of parameters',
            pints.NestedEllipsoidSampler, self.log_likelihood, log_prior)

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

    def test_settings_check(self):
        """
        Tests the settings check at the start of a run.
        """
        sampler = pints.NestedEllipsoidSampler(
            self.log_likelihood, self.log_prior)
        sampler.set_posterior_samples(2)
        sampler.set_rejection_samples(5)
        sampler.set_iterations(10)
        sampler.set_active_points_rate(10)
        sampler.set_log_to_screen(False)
        sampler.run()

        sampler.set_posterior_samples(10)
        self.assertRaisesRegex(ValueError, 'exceed 0.25', sampler.run)
        sampler.set_posterior_samples(2)
        sampler.set_iterations(4)
        self.assertRaisesRegex(
            ValueError, 'exceed number of iterations', sampler.run)

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
            sampler.set_log_to_file(False)
            samples, margin = sampler.run()
        self.assertEqual(c.text(), '')

        # Log to screen
        with StreamCapture() as c:
            sampler = pints.NestedEllipsoidSampler(
                self.log_likelihood, self.log_prior)
            sampler.set_posterior_samples(2)
            sampler.set_rejection_samples(5)
            sampler.set_iterations(20)
            sampler.set_active_points_rate(10)
            sampler.set_log_to_screen(True)
            sampler.set_log_to_file(False)
            samples, margin = sampler.run()
        lines = c.text().splitlines()
        self.assertEqual(lines[0], 'Running nested rejection sampling')
        self.assertEqual(lines[1], 'Number of active points: 10')
        self.assertEqual(lines[2], 'Total number of iterations: 20')
        self.assertEqual(lines[3], 'Enlargement factor: 1.5')
        self.assertEqual(lines[4], 'Total number of posterior samples: 2')
        self.assertEqual(lines[5], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[6:]:
            self.assertTrue(pattern.match(line))
        self.assertEqual(len(lines), 12)

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
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))

    def test_getters_and_setters(self):
        """
        Tests various get() and set() methods.
        """
        sampler = pints.NestedEllipsoidSampler(
            self.log_likelihood, self.log_prior)

        # Iterations
        x = sampler.iterations() + 1
        self.assertNotEqual(sampler.iterations(), x)
        sampler.set_iterations(x)
        self.assertEqual(sampler.iterations(), x)
        self.assertRaisesRegex(
            ValueError, 'negative', sampler.set_iterations, -1)

        # Active points rate
        x = sampler.active_points_rate() + 1
        self.assertNotEqual(sampler.active_points_rate(), x)
        sampler.set_active_points_rate(x)
        self.assertEqual(sampler.active_points_rate(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than 5', sampler.set_active_points_rate, 5)

        # Posterior samples
        x = sampler.posterior_samples() + 1
        self.assertNotEqual(sampler.posterior_samples(), x)
        sampler.set_posterior_samples(x)
        self.assertEqual(sampler.posterior_samples(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than zero', sampler.set_posterior_samples, 0)

        # Rejection samples
        x = sampler.rejection_samples() + 1
        self.assertNotEqual(sampler.rejection_samples(), x)
        sampler.set_rejection_samples(x)
        self.assertEqual(sampler.rejection_samples(), x)
        self.assertRaisesRegex(
            ValueError, 'negative', sampler.set_rejection_samples, -1)

        # Enlargement factor
        x = sampler.enlargement_factor() * 2
        self.assertNotEqual(sampler.enlargement_factor(), x)
        sampler.set_enlargement_factor(x)
        self.assertEqual(sampler.enlargement_factor(), x)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_enlargement_factor, 0.5)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_enlargement_factor, 1)

        # Ellipsoid update gap
        x = sampler.ellipsoid_update_gap() * 2
        self.assertNotEqual(sampler.ellipsoid_update_gap(), x)
        sampler.set_ellipsoid_update_gap(x)
        self.assertEqual(sampler.ellipsoid_update_gap(), x)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_ellipsoid_update_gap, 0.5)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_ellipsoid_update_gap, 1)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
