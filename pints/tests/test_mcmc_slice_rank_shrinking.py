# -*- coding: utf-8 -*-
#
# Tests the basic methods of the Covariance-Adaptive Slice Sampling:
# Rank Shrinking.
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


class TestSliceRankShrinking(unittest.TestCase):
    """
    Tests the basic methods of the Slice Sampling Rank Shrink
    routine.
    """
    def test_initialisation(self):
        """
        Tests whether all instance attributes are initialised correctly.
        """
        # Create mcmc
        x0 = np.array([2, 4])
        mcmc = pints.SliceRankShrinkingMCMC(x0)

        # Test attributes initialisation
        self.assertFalse(mcmc._running)
        self.assertFalse(mcmc._ready_for_tell)
        self.assertEqual(mcmc._current, None)
        self.assertEqual(mcmc._current_log_y, None)
        self.assertEqual(mcmc._proposed, None)
        self.assertEqual(mcmc._k, 0)
        self.assertEqual(mcmc._c_bar, 0)

    def test_first_run(self):
        """
        #Tests the very first run of the sampler.
        """
        # Create log pdf
        log_pdf = toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceRankShrinkingMCMC(x0)

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
        mcmc = pints.SliceRankShrinkingMCMC(x0)

        # Set initial crumb standard deviation
        mcmc.set_sigma_c(1)

        # First iteration
        x = mcmc.ask()
        fx, grad = log_pdf.evaluateS1(x)
        sample_0 = mcmc.tell((fx, grad))
        self.assertTrue(np.all(sample_0 == x0))

        # Second iteration
        x = mcmc.ask()
        self.assertEqual(mcmc._k, 1)
        fx, grad = log_pdf.evaluateS1(x)
        sample1 = mcmc.tell((fx, grad))
        self.assertTrue(sample1 is None)
        self.assertTrue(mcmc._c_bar is not True)

        # Third iteration
        x = mcmc.ask()
        self.assertEqual(mcmc._k, 2)
        fx, grad = log_pdf.evaluateS1(x)
        sample2 = mcmc.tell((fx, grad))
        self.assertTrue(np.all(sample2 == mcmc._current))
        self.assertEqual(mcmc._k, 0)
        self.assertEqual(mcmc._c_bar, 0)

        # Fourth iteration
        x = mcmc.ask()
        self.assertEqual(mcmc._k, 1)
        self.assertTrue(np.all(sample2 == mcmc._current))
        fx, grad = log_pdf.evaluateS1(x)
        sample3 = mcmc.tell((fx, grad))
        self.assertTrue(sample3 is None)

        # Fourth iteration
        x = mcmc.ask()
        self.assertEqual(mcmc._k, 2)
        fx, grad = log_pdf.evaluateS1(x)
        sample3 = mcmc.tell((fx, grad))
        self.assertEqual(mcmc._k, 0)
        self.assertEqual(mcmc._c_bar, 0)

    def test_run(self):
        # Set seed for monitoring
        np.random.seed(2)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceRankShrinkingMCMC(x0)

        # Run multiple iterations of the sampler
        chain = []
        while len(chain) < 100:
            x = mcmc.ask()
            fx, grad = log_pdf.evaluateS1(x)
            sample = mcmc.tell((fx, grad))
            if sample is not None:
                chain.append(np.copy(sample))

        # Fit Multivariate Gaussian to chain samples
        #print(np.mean(chain, axis=0))
        #print(np.cov(chain, rowvar=0))

    def test_basic(self):
        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceRankShrinkingMCMC(x0)

        # Test name
        self.assertEqual(mcmc.name(),
                         'Slice Sampling, Covariance Adaptive: Rank Shrinking')

        # Test set_w
        mcmc.set_sigma_c(3)
        self.assertEqual(mcmc._sigma_c, 3)
        with self.assertRaises(ValueError):
            mcmc.set_sigma_c(-1)

        # Test get_w
        self.assertEqual(mcmc.get_sigma_c(), 3)

        # Test current_slice height
        self.assertEqual(mcmc.get_current_slice_height(), mcmc._current_log_y)

        # Test number of hyperparameters
        self.assertEqual(mcmc.n_hyper_parameters(), 1)

        # Test setting hyperparameters
        mcmc.set_hyper_parameters([10])
        self.assertEqual(mcmc._sigma_c, 10)
