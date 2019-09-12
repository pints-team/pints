#!/usr/bin/env python3
#
# Tests the ABC Controller.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
import pints
import pints.toy
import unittest
import numpy as np

from shared import StreamCapture, TemporaryDirectory

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


debug = False


class TestABCController(unittest.TestCase):
    """
    Tests the ABCController class.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare problem for tests. """

        # Create toy model
        model = pints.toy.LogisticModel()
        cls.real_parameters = [0.1, 50]
        times = np.linspace(0, 100, 100)
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
            [0.0, 0],
            [0.2, 100]
        )

        # Set error measure
        cls.error_measure = pints.RootMeanSquaredError(cls.problem)

    def test_nparameters_error(self):
        """ Test that error is thrown when parameters from log prior and error
        measure do not match"""
        log_prior = pints.UniformLogPrior(
            [0.0, 0, 0],
            [0.2, 100, 1]

        self.assertRaises(ValueError, pints.ABCController, self.error_measure,
                          log_prior)

    def test_stopping(self):
        """ Test different stopping criteria. """

        abc = pints.ABCController(self.error_measure, self.log_prior)

        # Test setting max iterations
        maxi = abc.max_iterations() + 2
        self.assertNotEqual(maxi, abc.max_iterations())
        abc.set_max_iterations(maxi)
        self.assertEqual(maxi, abc.max_iterations())
        self.assertRaisesRegex(
            ValueError, 'negative', abc.set_max_iterations, -1)

        # Test without stopping criteria
        abc.set_max_iterations(None)
        self.assertIsNone(abc.max_iterations())
        self.assertRaisesRegex(
            ValueError, 'At least one stopping criterion', abc.run)

    def test_threshold(self):
        """ Test threshold value is acceptable"""
        abc = pints.ABCController(self.error_measure, self.log_prior)

        abc.set_threshold()
        self.assertEqual(abc._threshold, 1.5)
        abc.set_threshold(2))
        self.assertEqual(abc._threshold, 2)
        self.assertRaisesRegex(ValueError, 'negative', abc.set_threshold, -1)

    def test_parallel(self):
        """ Test running ABC with parallisation. """

        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nparameters = len(xs[0])
        niterations = 1000
        threshold = 1.2
        ntarget = 200
        ndraws = 1

        abc = pints.ABCController(
            self.error_measure, self.log_posterior, method=pints.ABCRejection)
        abc.set_max_iterations(niterations)
        abc.set_threshold(threshold)
        abc.set_n_target(ntarget)
        abc.set_n_draws(ndraws)

        # Test with auto-detected number of worker processes
        self.assertFalse(abc.parallel())
        abc.set_parallel(True)
        self.assertTrue(abc.parallel())
        self.assertEqual(abc._n_workers, pints.ParallelEvaluator.cpu_count())     # Check how to test this properly!!!

        # Test with fixed number of worker processes
        abc.set_parallel(2)
        self.assertIs(abc._parallel, True)
        self.assertEqual(abc._n_workers, 2)

        with StreamCapture() as c:
            chains = mcmc.run()
        self.assertIn('with 2 worker', c.text())
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)



    def test_deprecated_alias(self):

        mcmc = pints.MCMCSampling(
            self.log_posterior, 1, [self.real_parameters])
        self.assertIsInstance(mcmc, pints.MCMCController)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
