#!/usr/bin/env python
#
# Tests the basic methods of the seqential Monte Carlo routines.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#import re
import unittest
import numpy as np

import pints
import pints.toy

#from shared import StreamCapture, TemporaryDirectory

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestSMC(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`SMC`.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare for the test. """
        cls.log_pdf = pints.toy.MultimodalNormalLogPDF()
        cls.log_prior = pints.UniformLogPrior([-5, -5], [15, 15])
        cls.x0 = [5, 5]
        cls.sigma0 = [[2, 0], [0, 2]]

    def test_quick_run(self):
        """ Test a single run. """
        n = 100
        d = 2
        sampler = pints.SMC(self.log_pdf, self.log_prior, self.x0, self.sigma0)
        sampler.set_temperature_schedule(3)
        sampler.set_n_particles(n)
        sampler.set_n_kernel_samples(3)
        samples = sampler.run()

        # Check output has desired shape
        self.assertEqual(samples.shape, (n, d))

        # Check creation with sigma vector
        sigma0 = np.array([2, 2])
        sampler = pints.SMC(self.log_pdf, self.log_prior, self.x0, sigma0)
        sampler.set_temperature_schedule(3)
        sampler.set_n_particles(n)
        sampler.set_n_kernel_samples(3)
        samples = sampler.run()
        self.assertEqual(samples.shape, (n, d))

        # Check creation without sigma
        sampler = pints.SMC(self.log_pdf, self.log_prior, self.x0)
        sampler.set_temperature_schedule(3)
        sampler.set_n_particles(n)
        sampler.set_n_kernel_samples(3)
        samples = sampler.run()
        self.assertEqual(samples.shape, (n, d))

        # Check creation without log_prior
        sampler = pints.SMC(self.log_pdf, self.log_prior, self.x0)
        sampler.set_temperature_schedule(3)
        sampler.set_n_particles(n)
        sampler.set_n_kernel_samples(3)
        samples = sampler.run()
        self.assertEqual(samples.shape, (n, d))

    def test_construction_errors(self):
        """ Tests if invalid constructor calls are picked up. """

        # First arg must be a LogPDF
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPDF',
            pints.SMC, 'hello', self.log_prior, self.x0)

        # Log prior must be a LogPrior
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.SMC, self.log_pdf, 12, self.x0)

    '''
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
            sampler.set_iterations(10)
            sampler.set_active_points_rate(10)
            sampler.set_log_to_screen(True)
            sampler.set_log_to_file(False)
            samples, margin = sampler.run()
        lines = c.text().splitlines()
        self.assertEqual(len(lines), 10)
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
            sampler.set_iterations(10)
            sampler.set_active_points_rate(10)
            sampler.set_log_to_screen(True)
            sampler.set_log_to_file(False)
            samples, margin = sampler.run()
        lines = c.text().splitlines()
        self.assertEqual(len(lines), 11)
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
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))
    '''


if __name__ == '__main__':
    unittest.main()
