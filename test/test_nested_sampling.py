#!/usr/bin/env python2
#
# Tests the basic methods of the adaptive covariance MCMC routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy
import unittest
import numpy as np

#from shared import StreamCapture, TemporaryDirectory


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

    def test_quick(self):
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

#TODO: Test remaining methods, errors, etc.


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
