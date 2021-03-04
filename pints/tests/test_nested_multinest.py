#!/usr/bin/env python3
#
# Tests ellipsoidal nested sampler.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestMultiNestSampler(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`MultinestSampler`.
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
        cls.log_likelihood = pints.GaussianKnownSigmaLogLikelihood(
            problem, cls.noise)

    def test_getters_and_setters(self):
        # Tests various get() and set() methods.
        sampler = pints.NestedController(self.log_likelihood, self.log_prior,
                                         method=pints.MultinestSampler)
        self.assertEqual(sampler.f_s_threshold(), 1.1)
        sampler.set_f_s_threshold(4)
        self.assertEqual(sampler.f_s_threshold(), 4)
        self.assertRaises(ValueError, sampler.set_f_s_threshold, 0.5)



if __name__ == '__main__':
    unittest.main()
