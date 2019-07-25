# -*- coding: utf-8 -*-
#
# Tests the basic methods of the Covariance-Adaptive Slice Sampling:
# Covariance Matching.
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


class TestSliceCovarianceMatching(unittest.TestCase):
    """
    Tests the basic methods of the Slice Sampling Covariance Matching
    routine.

    Please refer to the _slice_covariance_matching.py script in ..\_mcmc
    """
    def test_initialisation(self):
        """
        Tests whether all instance attributes are initialised correctly.
        """
        # Create mcmc
        x0 = np.array([2, 4])
        mcmc = pints.SliceCovarianceMatchingMCMC(x0)

        # Test attributes initialisation
        self.assertFalse(mcmc._running)
        self.assertFalse(mcmc._ready_for_tell)
        self.assertEqual(mcmc._current, None)
        self.assertEqual(mcmc._current_log_y, None)
        self.assertEqual(mcmc._proposed, None)
        self.assertEqual(mcmc._proposed_pdf, None)
        self.assertEqual(mcmc._proposed_pdf, None)
        self.assertTrue(np.all(mcmc._c_bar_star == np.zeros(2)))
        self.assertTrue(np.all(
            mcmc._F == mcmc._sigma_c ** (-1) * np.identity(2)))
        self.assertTrue(np.all(
            mcmc._R == mcmc._sigma_c ** (-1) * np.identity(2)))

    def test_first_run(self):
        # Create log pdf
        log_pdf = toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceCovarianceMatchingMCMC(x0)

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
        mcmc = pints.SliceCovarianceMatchingMCMC(x0)

        # Set initial crumb standard deviation
        mcmc.set_sigma_c(1)

        # First iteration - 2.95
        x = mcmc.ask()
        fx, grad = log_pdf.evaluateS1(x)
        sample_0 = mcmc.tell((fx, grad))
        self.assertTrue(np.all(sample_0 == x0))
        self.assertTrue(np.all(mcmc._current == x0))
        self.assertTrue(np.all(mcmc._proposed == x0))
        self.assertEqual(mcmc._current_log_pdf, fx)
        self.assertEqual(mcmc._M, fx)
        self.assertTrue(mcmc._current_log_pdf >= mcmc._current_log_y)
        self.assertFalse(mcmc._calculate_fx_u)
        self.assertFalse(mcmc._sent_proposal)
        self.assertTrue(np.all(mcmc._c_bar_star == 0))

        # Second iteration
        x = mcmc.ask()
        self.assertTrue(np.all(mcmc._c_bar_star == mcmc._c))
        self.assertTrue(mcmc._calculate_fx_u)
        self.assertTrue(mcmc._sent_proposal)
        self.assertTrue(np.all(mcmc._proposed == x))
        fx, grad = log_pdf.evaluateS1(x)
        sample_1 = mcmc.tell((fx, grad))
        self.assertEqual(sample_1, None)
        self.assertTrue(mcmc._calculate_fx_u)

        # Third iteration
        x = mcmc.ask()
        self.assertTrue(np.all(mcmc._u == x))
        self.assertFalse(mcmc._calculate_fx_u)
        fx, grad = log_pdf.evaluateS1(x)
        sample_2 = mcmc.tell((fx, grad))
        self.assertEqual(fx, mcmc._log_fx_u)
        self.assertTrue(np.all(mcmc._current == x0))
        self.assertEqual(sample_2, None)
        self.assertFalse(mcmc._calculate_fx_u)

        # Fourth iteration
        x = mcmc.ask()
        fx, grad = log_pdf.evaluateS1(x)
        self.assertTrue(mcmc._calculate_fx_u)
        self.assertTrue(mcmc._sent_proposal)
        sample_3 = mcmc.tell((fx, grad))
        self.assertEqual(sample_3, None)
        self.assertTrue(mcmc._calculate_fx_u)

        # Fifth iteration
        x = mcmc.ask()
        self.assertFalse(mcmc._calculate_fx_u)
        self.assertTrue(np.all(mcmc._u == x))
        fx, grad = log_pdf.evaluateS1(x)
        sample_4 = mcmc.tell((fx, grad))
        self.assertEqual(sample_4, None)

        # Sixth iteration
        x = mcmc.ask()
        fx, grad = log_pdf.evaluateS1(x)
        sample_5 = mcmc.tell((fx, grad))
        self.assertTrue(np.all(mcmc._current == sample_5))
        self.assertEqual(mcmc._M, fx)
        self.assertTrue((mcmc._current_log_y < mcmc._M))
        self.assertFalse(mcmc._calculate_fx_u)
        self.assertTrue(np.all(mcmc._c_bar_star == np.zeros(2)))

    def test_run(self):
        # Set seed for monitoring
        np.random.seed(2)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceCovarianceMatchingMCMC(x0)

        # Run multiple iterations of the sampler
        chain = []
        while len(chain) < 10000:
            x = mcmc.ask()
            fx, grad = log_pdf.evaluateS1(x)
            sample = mcmc.tell((fx, grad))
            if sample is not None:
                chain.append(np.copy(sample))

        # Fit Multivariate Gaussian to chain samples
        # (np.mean(chain, axis=0))
        # (np.cov(chain, rowvar=0))

    def test_basic(self):
        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceCovarianceMatchingMCMC(x0)

        # Test name
        self.assertEqual(
            mcmc.name(), 'Slice Sampling - Covariance Adaptive:' +
            ' ' + 'Covariance Matching')

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
        self.assertEqual(mcmc.n_hyper_parameters(), 2)

        # Test setting hyperparameters
        mcmc.set_hyper_parameters([10, 10])
        self.assertEqual(mcmc._sigma_c, 10)
        self.assertEqual(mcmc._theta, 10)

