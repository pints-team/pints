#!/usr/bin/env python
#
# Unit tests for the SMC ABC class.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

import unittest

import numpy as np

import pints
import pints.toy as toy
import pints.toy.stochastic


class TestABCSMC(unittest.TestCase):
    """
    Tests the basic methods of the ABC SMC routine.
    """
# Set up toy model, parameter values, problem, error measure
    @classmethod
    def setUpClass(cls):
        """ Set up problem for tests. """
        # Create toy model
        cls.model = toy.stochastic.DegradationModel()
        cls.real_parameters = [0.1]
        cls.times = np.linspace(0, 10, 10)
        cls.values = cls.model.simulate(cls.real_parameters, cls.times)

        # Create an object (problem) with links to the model and time series
        cls.problem = pints.SingleOutputProblem(
            cls.model, cls.times, cls.values)

        # Create a uniform prior over both the parameters
        cls.log_prior = pints.UniformLogPrior(
            [0.0],
            [0.3]
        )

        cls.transition_kernel = pints.MultivariateGaussianLogPrior(
            np.zeros(1),
            0.001 * np.identity(1)
        )

        # Set error measure
        cls.error_measure = pints.RootMeanSquaredError(cls.problem)

    def test_method(self):
        # Create abc smc sampler
        abc = pints.ABCSMC(self.log_prior)

        # Configure
        n_draws = 1
        niter = 20
        abc.set_intermediate_size(niter)
        abc.set_threshold_schedule([6, 4, 2])

        # Perform short run using ask and tell framework
        samples = []
        while len(samples) < 3 * niter:
            xs = abc.ask(n_draws)
            fxs = [self.error_measure(x) for x in xs]
            sample = abc.tell(fxs)
            while sample is None:
                xs = abc.ask(n_draws)
                fxs = [self.error_measure(x) for x in xs]
                sample = abc.tell(fxs)
            samples.append(sample)

        samples = np.array(samples)
        self.assertEqual(samples.shape[0], 3 * niter)

    def test_method_list(self):
        # Create abc smc sampler
        abc = pints.ABCSMC(self.log_prior,
                           pints.MultivariateGaussianLogPrior(
                               np.zeros(1),
                               0.001 * np.identity(1)))

        # Configure
        n_draws = 2
        niter = 10
        abc.set_intermediate_size(niter)
        abc.set_threshold_schedule([6, 4, 2])
        abc.set_perturbation_kernel(
            pints.MultivariateGaussianLogPrior(
                np.zeros(1),
                0.001 * np.identity(1)))

        # Perform short run using ask and tell framework
        samples = []
        while len(samples) < 3 * niter:
            xs = abc.ask(n_draws)
            fxs = [self.error_measure(x) for x in xs]
            sample = abc.tell(fxs)
            while sample is None:
                xs = abc.ask(n_draws)
                fxs = [self.error_measure(x) for x in xs]
                sample = abc.tell(fxs)
            samples.append(sample)
        self.assertEqual(len(samples), 3 * niter)

    def test_errors(self):
        # test errors in abc rejection
        abc = pints.ABCSMC(self.log_prior)
        abc.ask(1)
        # test two asks raises error
        self.assertRaises(RuntimeError, abc.ask, 1)

        # test tell with large value
        self.assertEqual(None, abc.tell(100))
        # test error raised if tell called before ask
        self.assertRaises(RuntimeError, abc.tell, 2.5)

        self.assertRaises(ValueError, abc.set_threshold_schedule, [1, -1])

        self.assertRaises(ValueError, abc.set_perturbation_kernel, [1])

        self.assertRaises(
            ValueError,
            pints.ABCSMC,
            self.log_prior,
            np.array([0]))

    def test_name(self):
        abc = pints.ABCSMC(self.log_prior)
        self.assertEqual(abc.name(), 'ABC-SMC')


if __name__ == '__main__':
    unittest.main()
