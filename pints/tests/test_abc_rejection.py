#!/usr/bin/env python
#
# Tests the basic methods of the adaptive covariance MCMC routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy as toy
import unittest
import numpy as np

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestABCRejection(unittest.TestCase):
    """
    Tests the basic methods of the ABC Rejection routine.
    """
# Set up toy model, parameter values, problem, error measure
    @classmethod
    def setUpClass(cls):
        """ Set up problem for tests. """

        # Create toy model
        cls.model = toy.LogisticModel()
        cls.real_parameters = [0.1, 50]
        cls.times = np.linspace(0, 100, 100)
        cls.values = cls.model.simulate(cls.real_parameters, cls.times)

        # Add noise
        cls.noise = 1
        cls.values += np.random.normal(0, cls.noise, cls.values.shape)
        cls.real_parameters.append(cls.noise)
        cls.real_parameters = np.array(cls.real_parameters)

        # Create an object (problem) with links to the model and time series
        cls.problem = pints.SingleOutputProblem(
            cls.model, cls.times, cls.values)

        # Create a uniform prior over both the parameters
        cls.log_prior = pints.UniformLogPrior(
            [0, 0],
            [0.2, 100]
        )

        # Set error measure
        cls.error_measure = pints.RootMeanSquaredError(cls.problem)

    def test_method(self):

        # Create abc rejection scheme
        threshold = 1.2
        abc = pints.ABCRejection(self.log_prior, threshold)

        # Configure
        n_draws = 10

        # Perform short run using ask and tell framework
        samples = []
        while len(samples) < self._n_target:
            x = abc.ask(n_draws)
            fx = self.error_measure(x)
            sample = abc.tell(fx)

            samples.extend(sample)

        samples = np.array(samples)
        self.assertEqual(samples.shape[0], 3)
        self.assertEqual(samples.shape[1], self._n_target)

    def test_tell_error(self):
        # Create abc rejection scheme
        threshold = 1.2
        abc = pints.ABCRejection(self.log_prior, threshold)

        # Perform one iteration of ask and tell
        x = abc.ask(10)
        fx = [1, 2, 3]
        self.assertRaises(ValueError, abc.tell, fx)


if __name__ == '__main__':
    unittest.main()
