#!/usr/bin/env python3
#
# Tests the basic methods of the Slice Sampling with Doubling routine.
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

from shared import StreamCapture

debug = False


class TestSliceDoubling(unittest.TestCase):
    """
    Tests the basic methods of the Slice Sampling with Doubling routine.

    Please refer to the _slice_doubling.py script in ..\_mcmc\_slice_doubling.py
    """

    def test_initialisation(self):
        """
        Tests whether all instance attributes are initialised correctly.
        """
        # Create mcmc
        x0 = np.array([2, 4])
        mcmc = pints.SliceDoublingMCMC(x0)

        # Test attributes initialisation
        self.assertFalse(mcmc._running)
        self.assertFalse(mcmc._ready_for_tell)
        self.assertFalse(mcmc._first_expansion)
        self.assertFalse(mcmc._interval_found)
        self.assertFalse(mcmc._d)
        self.assertFalse(mcmc._init_check)
        self.assertFalse(mcmc._continue_check)
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
        self.assertEqual(mcmc._l_hat, None)
        self.assertEqual(mcmc._r_hat, None)
        self.assertEqual(mcmc._temp_l_hat, None)
        self.assertEqual(mcmc._temp_r_hat, None)
        self.assertEqual(mcmc._fx_l, None)
        self.assertEqual(mcmc._fx_r, None)

        self.assertTrue(np.all(mcmc._w == [1,1]))
        self.assertEqual(mcmc._p, 10)
        self.assertEqual(mcmc._k, 0)
        self.assertEqual(mcmc._i, 0)
        self.assertEqual(mcmc._mcmc_iteration, 0)


    def test_first_run(self):
        """
        Tests the very first run of the sampler. 
        """
        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceDoublingMCMC(x0)

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

        # Test first iteration of tell(). The first point in the chain should be x0
        self.assertTrue(np.all(mcmc.tell(fx) == x0))

        # We update the current sample
        self.assertTrue(np.all(mcmc._current == x0))
        self.assertTrue(np.all(mcmc._current == mcmc._proposed))

        # We update the _current_log_pdf value used to generate the new slice
        self.assertEqual(mcmc._current_log_pdf, fx)

        # Check that the new slice has been constructed appropriately 
        self.assertTrue(mcmc._current_log_y == (mcmc._current_log_pdf - mcmc._e))
        self.assertTrue(mcmc._current_log_y < mcmc._current_log_pdf)

        # Check flag
        self.assertTrue(mcmc._first_expansion)

    
    def test_cycle(self):
        """
        Tests every step of a single MCMC iteration.
        """
        # Set seed for monitoring
        np.random.seed(1)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceDoublingMCMC(x0)

        # VERY FIRST RUN
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertTrue(np.all(sample == x0))
        self.assertEqual(mcmc._fx_l, None)
        self.assertEqual(mcmc._fx_r, None)

    
        ##################################
        ### FIRST PARAMETER  - INDEX 0 ###
        ##################################

        # FIRST 2 RUNS: create initial interval edges

        # Start with initiating edges and ask for log_pdf of left edge
        self.assertTrue(mcmc._first_expansion)
        x = mcmc.ask()
        self.assertFalse(mcmc._first_expansion)
        
        # Check that interval edges are initialised appropriately
        self.assertTrue(mcmc._l < mcmc._current[0])
        self.assertTrue(mcmc._r > mcmc._current[0])

        # Check that interval I expansion steps are correct
        self.assertEqual(mcmc._k, 10)

        # We calculate the log pdf of the initial left interval edge
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)

        # Check that we have set the log_pdf of the initialised left edge correctly
        self.assertEqual(fx, mcmc._fx_l)
        
        # Ask for log_pdf of right edge
        self.assertFalse(mcmc._first_expansion)
        x = mcmc.ask()

        # We calculate the log pdf of the initial right interval edge
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)

        # Check that we have set the log_pdf of the initialised right edge correctly
        self.assertEqual(fx, mcmc._fx_r)
        
        # Check that flags for pdf of initial edges are False
        self.assertFalse(mcmc._init_left)
        self.assertFalse(mcmc._init_right)

        
        # THIRD RUN: begin expanding the interval
        x = mcmc.ask()
        
        # v < .5, therefore we expand the left edge
        self.assertEqual(x[0], mcmc._l)
        
        # Check that we are still within the slice or that k > 0
        self.assertTrue(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
        
        # The edges are inside the slice: return None and update the log pdf of the left edge
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(sample, None)
        self.assertEqual(fx, mcmc._fx_l)
        
        
        # SUBSEQUENT EXPANSIONS: expand left edge n-1 times and on the nth iteration we expand the right edge
        while mcmc._v < .5 or mcmc._k == 0:
            x = mcmc.ask()
            self.assertTrue(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
            self.assertEqual(sample, None)

        
        # COMPLETE INTERVAL EXPANSION: Check whether the edges are now outside the slice
        self.assertFalse(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
           
        # PROPOSE PARAMETER: now that we have estimated the interval, we sample a new parameter,
        # set the interval_found, _init_check and _continue_check flags to True 
        x = mcmc.ask()
        self.assertTrue(mcmc._interval_found)
        self.assertTrue(mcmc._init_check)
        self.assertTrue(mcmc._continue_check)

        
        # The log pdf of the proposed point is smaller than the slice height. The ``Threshold Check``
        # was not passed, so we reject, shrink and set the _init_check and _continue_check flags to 
        # False
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(sample, None)
        self.assertEqual(mcmc._l, mcmc._proposed[mcmc._i])
        self.assertFalse(mcmc._init_check)
        self.assertFalse(mcmc._continue_check)
        
        
        # TRY NEW PROPOSALS: Stop when a point is proposed within the slice
        while mcmc._current_log_y >= fx:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
        self.assertTrue(mcmc._current_log_y <= fx)

        
        # START ACCEPTANCE CHECK: Since the new proposed point passed the threshold test (fx > log_y),
        # we mantain the flags for proceeding with the ``Acceptance Check``
        self.assertTrue(mcmc._init_check)
        self.assertTrue(mcmc._continue_check)     

        # Check whether we are initialising the ``Acceptance Check`` interval appropriately 
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)     
        self.assertEqual(fx, mcmc._fx_l_hat)
        self.assertEqual(None, mcmc._fx_r_hat)
        self.assertTrue(np.all(mcmc._temp_l == mcmc._temp_l_hat))

        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)     
        self.assertEqual(fx, mcmc._fx_r_hat)
        self.assertTrue(np.all(mcmc._temp_r == mcmc._temp_r_hat))

        # Since the intervals generated from the new point do not differ from the ones generated 
        # by the current sample, the following condition should be false, therefore d should remain
        # False
        self.assertFalse((mcmc._current[mcmc._i] < mcmc._m and mcmc._proposed[mcmc._i] >= mcmc._m) or (mcmc._current[mcmc._i] >= mcmc._m and mcmc._proposed[mcmc._i] < mcmc._m))
        self.assertFalse(mcmc._d)

        # We proceed with the ``Acceptance Check``
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(fx, mcmc._fx_r_hat)

        # The rejection condition in the ``Acceptance Check`` procedure should be False
        self.assertFalse(mcmc._d == True and mcmc._current_log_y >= fx[0] and mcmc._current_log_y >= fx[1])

        # Since the point hasn't been rejected, we will continue the ``Acceptance Check`` process
        while (mcmc._r_hat - mcmc._l_hat) > 1.1 * mcmc._w[mcmc._i]:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx) 
        
        # The loop ends once the interval is smaller than ``1.1*w```
        self.assertFalse((mcmc._r_hat - mcmc._l_hat) > 1.1 * mcmc._w[mcmc._i])

        # Since the ``Acceptance Check`` is finished, the _continue_check has been set to False
        # and we accepted the point
        x = mcmc.ask()
        self.assertFalse(mcmc._continue_check)
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)

        # As we have accepted the new point, we reset the interval expansion flags
        self.assertTrue(mcmc._first_expansion)
        self.assertFalse(mcmc._interval_found)

        # We increase the index _i to 1 to move to the next parameter to update
        self.assertEqual(mcmc._i, 1)

        ##################################
        ### SECOND PARAMETER - INDEX 1 ###
        ##################################

        # Start with initiating edges and ask for log_pdf of left edge
        self.assertTrue(mcmc._first_expansion)
        x = mcmc.ask()
        self.assertFalse(mcmc._first_expansion)
        
        # Check that interval edges are initialised appropriately
        self.assertTrue(mcmc._l < mcmc._current[1])
        self.assertTrue(mcmc._r > mcmc._current[1])

        # Check that interval I expansion steps are correct
        self.assertEqual(mcmc._k, 10)

        # We calculate the log pdf of the initial left interval edge
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)

        # Check that we have set the log_pdf of the initialised left edge correctly
        self.assertEqual(fx, mcmc._fx_l)
        
        # Ask for log_pdf of right edge
        self.assertFalse(mcmc._first_expansion)
        x = mcmc.ask()

        # We calculate the log pdf of the initial right interval edge
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)

        # Check that we have set the log_pdf of the initialised right edge correctly
        self.assertEqual(fx, mcmc._fx_r)
        
        # Check that flags for pdf of initial edges are False
        self.assertFalse(mcmc._init_left)
        self.assertFalse(mcmc._init_right)

        
        # THIRD RUN: begin expanding the interval
        x = mcmc.ask()
        
        # v < .5, therefore we expand the left edge
        self.assertEqual(x[1], mcmc._l)
        
        # Check that we are still within the slice or that k > 0
        self.assertTrue(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
        
        # The edges are inside the slice: return None and update the log pdf of the left edge
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(sample, None)
        self.assertEqual(fx, mcmc._fx_l)

        
        # SUBSEQUENT EXPANSIONS: expand left edge n-1 times and on the nth iteration we expand the right edge
        while mcmc._v < .5 or mcmc._k == 0:
            x = mcmc.ask()
            self.assertTrue(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
            self.assertEqual(sample, None)
        
        # COMPLETE INTERVAL EXPANSION: Check whether the edges are now outside the slice
        self.assertFalse(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
        
        # PROPOSE PARAMETER: now that we have estimated the interval, we sample a new parameter,
        # set the interval_found, _init_check and _continue_check flags to True 
        x = mcmc.ask()
        self.assertTrue(mcmc._interval_found)
        self.assertTrue(mcmc._init_check)
        self.assertTrue(mcmc._continue_check)

        # The log pdf of the proposed point is greater than the slice height.
        # The point has passed the "Threshold Check"
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertTrue(fx > mcmc._current_log_y)
        self.assertFalse(mcmc._init_left_hat)
        self.assertFalse(mcmc._init_right_hat)

        
        # START ACCEPTANCE CHECK: Since the new proposed point passed the threshold test (fx > log_y),
        # we mantain the flags for proceeding with the ``Acceptance Check``
        self.assertTrue(mcmc._init_check)
        self.assertTrue(mcmc._continue_check)     

        # Check whether we are initialising the ``Acceptance Check`` interval appropriately 
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)     
        self.assertEqual(fx, mcmc._fx_l_hat)
        self.assertEqual(None, mcmc._fx_r_hat)
        self.assertTrue(np.all(mcmc._temp_l == mcmc._temp_l_hat))

        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)     
        self.assertEqual(fx, mcmc._fx_r_hat)
        self.assertTrue(np.all(mcmc._temp_r == mcmc._temp_r_hat))

        # Since the intervals generated from the new point do not differ from the ones generated 
        # by the current sample, the following condition should be false, therefore d should remain
        # False
        self.assertFalse((mcmc._current[mcmc._i] < mcmc._m and mcmc._proposed[mcmc._i] >= mcmc._m) or (mcmc._current[mcmc._i] >= mcmc._m and mcmc._proposed[mcmc._i] < mcmc._m))
        
        # We proceed with the ``Acceptance Check``
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(fx, mcmc._fx_r_hat)

        # The rejection condition in the ``Acceptance Check`` procedure should be False
        self.assertFalse(mcmc._d == True and mcmc._current_log_y >= fx[0] and mcmc._current_log_y >= fx[1])

        # Since the point hasn't been rejected, we will continue the ``Acceptance Check`` process
        while (mcmc._r_hat - mcmc._l_hat) > 1.1 * mcmc._w[mcmc._i]:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)

        
        # The loop ends once the interval is smaller than ``1.1*w``
        self.assertFalse((mcmc._r_hat - mcmc._l_hat) > 1.1 * mcmc._w[mcmc._i])
        self.assertFalse((mcmc._d == True and mcmc._current_log_y >= fx[0] and mcmc._current_log_y >= fx[1]))

        # Since the Acceptance check is finished, the _continue_check has been set to False
        # and we accepted the point
        x = mcmc.ask()
        self.assertFalse(mcmc._continue_check)
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)

        # As we have accepted the new point, we reset the interval expansion flags
        self.assertTrue(mcmc._first_expansion)
        self.assertFalse(mcmc._interval_found)

        # All the parameters of the sample have been updates, so we reset the index to 0 
        self.assertEqual(mcmc._i, 0)

        # Now that we have generated the new sample, we set this to be the current sample
        self.assertTrue(np.all(mcmc._current == mcmc._proposed))

        # We generate a new log_y for the height of the new slice
        self.assertEqual(fx, mcmc._current_log_pdf)

        # Check whether the new slice has been generated correctly
        self.assertEqual(mcmc._current_log_y, mcmc._current_log_pdf - mcmc._e)

    
    def test_complete_run(self):
        """
        Test multiple MCMC iterations of the sample
        """
        # Set seed for monitoring
        np.random.seed(1)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([1,1])
        mcmc = pints.SliceDoublingMCMC(x0)

        # Run multiple iterations of the sampler
        chain = []
        while mcmc._mcmc_iteration < 10000:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
            if sample is not None:
                chain.append(sample)

        # Fit Multivariate Gaussian to chain samples
        mean = np.mean(chain, axis=0)
        cov = np.cov(chain, rowvar=0)

        print(mean) #[2.01 4.01]
        print(cov)  #[[1.00, 0.00][0.00, 2.94]]



    def test_basic(self):
        """
        Test basic methods of the class.
        """
        # Create mcmc
        x0 = np.array([1,1])
        mcmc = pints.SliceDoublingMCMC(x0)

        # Test name
        self.assertEqual(mcmc.name(), 'Slice Sampling - Doubling')

        # Test set_w
        mcmc.set_w(2)
        self.assertTrue(np.all(mcmc._w == np.array([2,2])))
        with self.assertRaises(ValueError):
            mcmc.set_w(-1)

        # Test set_p
        mcmc.set_p(3)
        self.assertEqual(mcmc._p, 3.)
        with self.assertRaises(ValueError):
            mcmc.set_p(-1)

        # Test get_w
        self.assertTrue(np.all(mcmc.get_w() == np.array([2,2])))

        # Test get_m
        self.assertEqual(mcmc.get_p(), 3.)

        # Test current_log_pdf
        self.assertEqual(mcmc.current_log_pdf(), mcmc._current_log_pdf)

        # Test current_slice height
        self.assertEqual(mcmc.current_slice_height(), mcmc._current_log_y)

        # Test number of hyperparameters
        self.assertEqual(mcmc.n_hyper_parameters(), 2)

        # Test setting hyperparameters
        mcmc.set_hyper_parameters([3, 100])
        self.assertTrue((np.all(mcmc._w == np.array([3,3]))))
        self.assertEqual(mcmc._p, 100)


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

        # Get properties of the noise sample
        noise_sample_mean = np.mean(values - org_values)
        noise_sample_std = np.std(values - org_values)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a log-likelihood function (adds an extra parameter!)
        log_likelihood = pints.GaussianLogLikelihood(problem)

        # Create a uniform prior over both the parameters and the new noise variable
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
            log_posterior, num_chains, xs, method=pints.SliceDoublingMCMC)

        for sampler in mcmc.samplers():
            sampler.set_w(0.1)

        # Add stopping criterion
        mcmc.set_max_iterations(1000)

        # Set up modest logging
        mcmc.set_log_to_screen(True)
        mcmc.set_log_interval(500)

        # Run!
        print('Running...')
        chains = mcmc.run()
        print('Done!')

if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
    