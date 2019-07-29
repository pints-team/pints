# -*- coding: utf-8 -*-
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


class TestSliceStepout(unittest.TestCase):
    """
    Tests the basic methods of the Slice Sampling with Stepout routine.

    Please refer to the _slice_stepout.py script in ..\_mcmc\_slice_stepout.py
    """
    def test_initialisation(self):
        """
        Tests whether all instance attributes are initialised correctly.
        """
        # Create mcmc
        x0 = np.array([2, 4])
        mcmc = pints.SliceStepoutMCMC(x0)

        # Test attributes initialisation
        self.assertFalse(mcmc._running)
        self.assertFalse(mcmc._ready_for_tell)
        self.assertFalse(mcmc._first_expansion)
        self.assertFalse(mcmc._interval_found)
        self.assertFalse(mcmc._set_l)
        self.assertFalse(mcmc._set_r)
        self.assertFalse(mcmc._init_left)
        self.assertFalse(mcmc._init_right)

        self.assertEqual(mcmc._current, None)
        self.assertEqual(mcmc._current_log_pdf, None)
        self.assertEqual(mcmc._current_log_y, None)
        self.assertEqual(mcmc._proposed, None)
        self.assertEqual(mcmc._l, None)
        self.assertEqual(mcmc._r, None)
        self.assertEqual(mcmc._temp_l, None)
        self.assertEqual(mcmc._temp_r, None)
        self.assertEqual(mcmc._k, None)
        self.assertEqual(mcmc._j, None)
        self.assertEqual(mcmc._fx_l, None)
        self.assertEqual(mcmc._fx_r, None)

        self.assertEqual(mcmc._m, 50)
        self.assertEqual(mcmc._active_param_index, 0)

    def test_first_run(self):
        """
        Tests the very first run of the sampler.
        """
        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceStepoutMCMC(x0)

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

        # Tell should fail when log pdf of x0 is infinite
        with self.assertRaises(ValueError):
            fx = np.inf
            mcmc.tell(fx)

        # Calculate log pdf for x0
        fx = log_pdf.evaluateS1(x0)[0]

        # Tell should fail when _ready_for_tell is False
        with self.assertRaises(RuntimeError):
            mcmc._ready_for_tell = False
            mcmc.tell(fx)

        # Undo changes
        mcmc._ready_for_tell = True

        # Test first iteration of tell(). The first point in the chain
        # should be x0
        self.assertTrue(np.all(mcmc.tell(fx) == x0))

        # We update the current sample
        self.assertTrue(np.all(mcmc._current == x0))
        self.assertTrue(np.all(mcmc._current == mcmc._proposed))

        # We update the _current_log_pdf value used to generate the new slice
        self.assertEqual(mcmc._current_log_pdf, fx)

        # Check that the new slice has been constructed appropriately
        self.assertTrue(mcmc._current_log_y < mcmc._current_log_pdf)

        # Check flag
        self.assertTrue(mcmc._first_expansion)

    def test_run(self):
        """
        Test multiple MCMC iterations of the sampler on a
        Multivariate Gaussian.
        """
        # Set seed for monitoring
        np.random.seed(2)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceStepoutMCMC(x0)

        # Run multiple iterations of the sampler
        chain = []
        while len(chain) < 100:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
            if sample is not None:
                chain.append(np.copy(sample))

        # Fit Multivariate Gaussian to chain samples
        np.mean(chain, axis=0)
        np.cov(chain, rowvar=0)

    def test_basic(self):
        """
        Test basic methods of the class.
        """
        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceStepoutMCMC(x0)

        # Test name
        self.assertEqual(mcmc.name(), 'Slice Sampling - Stepout')

        # Test set_w
        mcmc.set_w(2)
        self.assertTrue(np.all(mcmc._w == np.array([2, 2])))
        with self.assertRaises(ValueError):
            mcmc.set_w(-1)

        # Test set_m
        mcmc.set_m(3)
        self.assertEqual(mcmc._m, 3.)
        with self.assertRaises(ValueError):
            mcmc.set_m(-1)

        # Test set_prob_overrelaxed
        mcmc.set_prob_overrelaxed(.5)
        self.assertEqual(mcmc._prob_overrelaxed, .5)
        with self.assertRaises(ValueError):
            mcmc.set_prob_overrelaxed(-1)
        with self.assertRaises(ValueError):
            mcmc.set_prob_overrelaxed(4)

        # Test set_a
        mcmc.set_a(40)
        self.assertEqual(mcmc._a, 40)
        self.assertEqual(mcmc.get_a(), 40)
        with self.assertRaises(ValueError):
            mcmc.set_a(-30)
        with self.assertRaises(ValueError):
            mcmc.set_prob_overrelaxed(-1)

        # Test get_w
        self.assertTrue(np.all(mcmc.get_w() == np.array([2, 2])))

        # Test get_m
        self.assertEqual(mcmc.get_m(), 3.)

        # Test current_log_pdf
        self.assertEqual(mcmc.get_current_log_pdf(), mcmc._current_log_pdf)

        # Test current_slice height
        self.assertEqual(mcmc.get_current_slice_height(), mcmc._current_log_y)

        # Test number of hyperparameters
        self.assertEqual(mcmc.n_hyper_parameters(), 4)

        # Test setting hyperparameters
        mcmc.set_hyper_parameters([3, 100, .7, 50])
        self.assertTrue((np.all(mcmc._w == np.array([3, 3]))))
        self.assertEqual(mcmc._m, 100)
        self.assertEqual(mcmc._prob_overrelaxed, 0.7)
        self.assertEqual(mcmc._a, 50)

        mcmc.set_w([5, 5])
        self.assertTrue(np.all(mcmc._w == np.array([5, 5])))

        mcmc.set_prob_overrelaxed(1)
        self.assertEqual(mcmc._prob_overrelaxed, 1)

    def test_logistic(self):
        """
        Test sampler on a logistic task.
        """
        # Load a forward model
        model = toy.LogisticModel()

        # Create some toy data
        real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)
        org_values = model.simulate(real_parameters, times)

        # Add noise
        noise = 10
        values = org_values + np.random.normal(0, noise, org_values.shape)
        real_parameters = np.array(real_parameters + [noise])

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a log-likelihood function (adds an extra parameter!)
        log_likelihood = pints.GaussianLogLikelihood(problem)

        # Create a uniform prior over both the parameters and the new
        # noise variable
        log_prior = pints.UniformLogPrior(
            [0.01, 400, noise * 0.1],
            [0.02, 600, noise * 100],
        )

        # Create a posterior log-likelihood (log(likelihood * prior))
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)

        # Choose starting points for 3 mcmc chains
        num_chains = 1
        xs = [real_parameters * (1 + 0.1 * np.random.rand())]

        # Create mcmc routine
        mcmc = pints.MCMCController(
            log_posterior, num_chains, xs, method=pints.SliceStepoutMCMC)

        for sampler in mcmc.samplers():
            sampler.set_w(0.1)

        # Add stopping criterion
        mcmc.set_max_iterations(1000)

        # Set up modest logging
        mcmc.set_log_to_screen(True)
        mcmc.set_log_interval(500)

        # Run!
        print('Running...')
        mcmc.run()
        print('Done!')

    def test_overrelaxed(self):

        # Set seed for monitoring
        np.random.seed(2)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceStepoutMCMC(x0)

        # Set probability of overrelaxed step
        mcmc.set_prob_overrelaxed(1)

        # Set w
        mcmc.set_w(10)

        # Check that variables are initialised correctly
        self.assertFalse(mcmc._overrelaxed_step)
        self.assertIsNone(mcmc._l_bar)
        self.assertIsNone(mcmc._r_bar)
        self.assertIsNone(mcmc._l_hat)
        self.assertIsNone(mcmc._r_hat)

        # First MCMC step: set flags to True
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        mcmc.tell(fx)

        self.assertTrue(mcmc._overrelaxed_step)
        self.assertTrue(mcmc._init_overrelaxation)
        self.assertTrue(mcmc._bisection)

        """FIRST PARAMETER"""
        # Expand interval using stepout
        while not mcmc._interval_found:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            mcmc.tell(fx)

        # Check initialisation of overrelaxed step - start narrowing ``w```
        self.assertTrue(mcmc._init_narrowing)
        self.assertFalse(mcmc._init_overrelaxation)
        self.assertTrue(((mcmc._r - mcmc._l) < 1.1 *
                         mcmc._w[mcmc._active_param_index]) and
                        mcmc._init_narrowing)

        # Continue narrowing ``w``
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        mcmc.tell(fx)
        self.assertFalse(mcmc._init_narrowing)
        self.assertFalse(mcmc._set_l_bisection)

        # Now that we have narrowed ``w``, init bisection
        while mcmc._a_bar > 0:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            mcmc.tell(fx)

        # Update parameter and index
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        self.assertFalse(mcmc._set_l_bisection)
        self.assertFalse(mcmc._set_r_bisection)
        mcmc.tell(fx)
        self.assertEqual(mcmc._active_param_index, 1)
        self.assertTrue(mcmc._first_expansion)
        self.assertFalse(mcmc._interval_found)

        """SECOND PARAMETER"""
        # Expand interval using stepout
        while not mcmc._interval_found:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            mcmc.tell(fx)

        # Check initialisation of overrelaxed step - start narrowing ``w```
        self.assertTrue(mcmc._init_narrowing)
        self.assertFalse(mcmc._init_overrelaxation)
        self.assertFalse(((mcmc._r - mcmc._l) < 1.1 *
                         mcmc._w[mcmc._active_param_index]) and
                         mcmc._init_narrowing)

        # Now that we have narrowed ``w``, init bisection
        while mcmc._a_bar > 0:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            mcmc.tell(fx)

        # Update parameter and index
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        self.assertFalse(mcmc._set_l_bisection)
        self.assertFalse(mcmc._set_r_bisection)
        mcmc.tell(fx)
        self.assertEqual(mcmc._active_param_index, 0)
        self.assertTrue(mcmc._first_expansion)
        self.assertFalse(mcmc._interval_found)

    def test_overrelaxed_run(self):

        # Set seed for monitoring
        np.random.seed(2)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceStepoutMCMC(x0)

        # Set probability of overrelaxed step
        mcmc.set_prob_overrelaxed(1)

        # Run multiple iterations of the sampler
        chain = []
        while len(chain) < 1000:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
            if sample is not None:
                chain.append(np.copy(sample))

        # Fit Multivariate Gaussian to chain samples
        np.mean(chain, axis=0)
        np.cov(chain, rowvar=0)

    def test_multimodal_run(self):
        """
        Test multiple MCMC iterations of the sample
        """
        # Set seed for monitoring
        np.random.seed(1)

        # Create problem
        log_pdf = pints.toy.MultimodalGaussianLogPDF(
            modes=[[0, 2], [0, 7], [5, 0], [4, 4]])
        x0 = np.random.uniform([2, 2], [8, 8], size=(4, 2))
        mcmc = pints.MCMCController(
            log_pdf, 4, x0, method=pints.SliceStepoutMCMC)

        for sampler in mcmc.samplers():
            sampler.set_w(20)

        # Set maximum number of iterations
        mcmc.set_max_iterations(1000)

        # Disable logging
        mcmc.set_log_to_screen(False)

        # Run!
        print('Running...')
        mcmc.run()
        print('Done!')

    def test_multimodal_overrelaxed_run(self):
        """
        Test multiple MCMC iterations of the sample
        """
        # Set seed for monitoring
        np.random.seed(1)

        # Create problem
        log_pdf = pints.toy.MultimodalGaussianLogPDF(
            modes=[[0, 2], [0, 7], [5, 0], [4, 4]])
        x0 = np.random.uniform([2, 2], [8, 8], size=(4, 2))
        mcmc = pints.MCMCController(
            log_pdf, 4, x0, method=pints.SliceStepoutMCMC)

        for sampler in mcmc.samplers():
            sampler.set_w(20)
            sampler.set_prob_overrelaxed = 0.98

        # Set maximum number of iterations
        mcmc.set_max_iterations(1000)

        # Disable logging
        mcmc.set_log_to_screen(False)

        # Run!
        print('Running...')
        mcmc.run()
        print('Done!')


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
