#!/usr/bin/env python3
#
# Tests the basic methods of the Slice Sampling routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

import unittest
import numpy as np

import pints
import pints.toy as toy

debug = False


class TestSliceHyperrectangles(unittest.TestCase):
    """
    Tests the basic methods of the Hyperrectangles-based Slice Samplingroutine.

    Please refer to the _slice_hyperrectangles.py script in ..\_mcmc
    """
    def test_initialisation(self):
        """
        Tests whether all instance attributes are initialised correctly.
        """
        # Create mcmc
        x0 = np.array([2, 4])
        mcmc = pints.SliceHyperrectanglesMCMC(x0)

        # Test attributes initialisation
        self.assertFalse(mcmc._running)
        self.assertFalse(mcmc._ready_for_tell)
        self.assertFalse(mcmc._hyperrectangle_positioned)

        self.assertEqual(mcmc._current, None)
        self.assertEqual(mcmc._current_log_y, None)
        self.assertEqual(mcmc._proposed, None)

    def test_first_run(self):
        """
        Tests the very first run of the sampler.
        """
        # Create log pdf
        log_pdf = toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceHyperrectanglesMCMC(x0)

        # Ask should fail if _ready_for_tell flag is True
        with self.assertRaises(RuntimeError):
            mcmc._ready_for_tell = True
            mcmc.ask()

        # Undo changes
        mcmc._ready_for_tell = False

        # Check whether _running flag becomes True when ask() is called
        # Check whether first iteration of ask() returns x0
        self.assertFalse(mcmc._running)
        self.assertTrue(np.all(mcmc.ask() == x0))
        self.assertTrue(mcmc._running)
        self.assertTrue(mcmc._ready_for_tell)

        # Calculate log pdf for x0
        fx, grad = log_pdf.evaluateS1(x0)

        # Tell should fail when _ready_for_tell is False
        with self.assertRaises(RuntimeError):
            mcmc._ready_for_tell = False
            mcmc.tell((fx, grad))

        # Undo changes
        mcmc._ready_for_tell = True

        # Test first iteration of tell(). The first point in the chain
        # should be x0
        sample_0 = mcmc.tell((fx, grad))
        self.assertTrue(np.all(sample_0 == x0))

        # We update the current sample
        self.assertTrue(np.all(mcmc._current == x0))
        self.assertTrue(np.all(mcmc._current == mcmc._proposed))

        # We update the _current_log_pdf value used to generate the new slice
        self.assertEqual(mcmc._current_log_pdf, fx)

        # Check that the new slice has been constructed appropriately
        self.assertTrue(mcmc._current_log_y < mcmc._current_log_pdf)

    def test_mcmc_step(self):
        # Set seed
        np.random.seed(2)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceHyperrectanglesMCMC(x0)

        # First MCMC step
        x = mcmc.ask()
        fx, grad = log_pdf.evaluateS1(x)
        sample_0 = mcmc.tell((fx, grad))
        self.assertTrue(np.all(sample_0 == x0))
        self.assertFalse(mcmc._hyperrectangle_positioned)

        # Next step
        x = mcmc.ask()
        fx, grad = log_pdf.evaluateS1(x)
        sample_1 = mcmc.tell((fx, grad))
        self.assertTrue(np.all(sample_1 is not None))
        self.assertTrue(np.all(mcmc._current == mcmc._proposed))

    def test_run(self):
        # Set seed for monitoring
        np.random.seed(2)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceHyperrectanglesMCMC(x0)

        # Set scales
        mcmc.set_w(5)

        # Run multiple iterations of the sampler
        chain = []
        while len(chain) < 10000:
            x = mcmc.ask()
            fx, grad = log_pdf.evaluateS1(x)
            sample = mcmc.tell((fx, grad))
            if sample is not None:
                chain.append(np.copy(sample))

        # Fit Multivariate Gaussian to chain samples
        '''print(np.mean(chain, axis=0))
        print(np.cov(chain, rowvar=0))'''

    def test_basic(self):
        """
        Test basic methods of the class.
        """
        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceHyperrectanglesMCMC(x0)

        # Test name
        self.assertEqual(mcmc.name(), 'Slice Sampling - Hyperrectangles')

        # Test set_w
        mcmc.set_w(2)
        self.assertTrue(np.all(mcmc._w == np.array([2, 2])))
        with self.assertRaises(ValueError):
            mcmc.set_w(-1)

        # Test get_w
        self.assertTrue(np.all(mcmc.get_w() == np.array([2, 2])))

        # Test current_slice height
        self.assertEqual(mcmc.get_current_slice_height(), mcmc._current_log_y)

        # Test number of hyperparameters
        self.assertEqual(mcmc.n_hyper_parameters(), 2)

        # Test setting hyperparameters
        mcmc.set_hyper_parameters([3, False])
        self.assertTrue((np.all(mcmc._w == np.array([3, 3]))))
        self.assertFalse(mcmc._adaptive)

    def test_logistic(self):
        """
        Test sampler on a logistic task.
        """
        # Load a forward model
        model = toy.LogisticModel()

        times = np.linspace(0, 1000, 50)

        # Create some toy data
        real_parameters = np.array([0.015, 500])
        org_values = model.simulate(real_parameters, times)

        # Add noise
        np.random.seed(1)
        noise = 10
        values = org_values + np.random.normal(0, noise, org_values.shape)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a log-likelihood function
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, noise)

        # Create a uniform prior over the parameters
        log_prior = pints.UniformLogPrior(
            [0.01, 400],
            [0.02, 600]
        )

        # Create a posterior log-likelihood (log(likelihood * prior))
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)

        # Choose starting points for 3 mcmc chains
        xs = [
            real_parameters * 1.01,
            real_parameters * 0.9,
            real_parameters * 1.1,
        ]

        # Create mcmc routine
        mcmc = pints.MCMCController(log_posterior, len(xs), xs,
                                    method=pints.SliceHyperrectanglesMCMC)

        # Add stopping criterion
        mcmc.set_max_iterations(1000)

        # Set up modest logging
        mcmc.set_log_to_screen(True)
        mcmc.set_log_interval(100)

        # Run!
        print('Running...')
        mcmc.run()
        print('Done!')
