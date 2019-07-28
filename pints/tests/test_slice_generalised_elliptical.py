# -*- coding: utf-8 -*-
#
# Tests the basic methods of the Generalised Elliptical Slice Sampling Routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

import unittest
import numpy as np

import pints
debug = False


class TestSliceGeneralisedElliptical(unittest.TestCase):
    """
    Tests the basic methods of the Generalised Elliptical Slice Sampling
    routine.

    Please refer to the _slice_generalised_elliptical.py script in ..\_mcmc
    """
    def test_initialisation(self):
        """
        Tests whether all instance attributes are initialised correctly.
        """
        # Create mcmc
        x0 = np.array([2, 4])
        mcmc = pints.SliceGeneralisedEllipticalMCMC(x0)

        # Test attributes initialisation
        self.assertFalse(mcmc._running)
        self.assertFalse(mcmc._ready_for_tell)
        self.assertEqual(mcmc._active_sample, None)
        self.assertEqual(mcmc._proposed_sample, None)

    def test_first_run(self):
        """
        Tests the very first run of the sampler.
        """

        # Set seed for testing
        np.random.seed(2)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceGeneralisedEllipticalMCMC(x0)

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
        self.assertTrue(np.all(mcmc._active_sample == x0))

    def test_full_run(self):
        """
        Tests a full run.
        """

        # Set seed for testing
        np.random.seed(2)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([0, 0])
        mcmc = pints.SliceGeneralisedEllipticalMCMC(x0)

        # First run
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertTrue(mcmc._prepare)
        self.assertTrue(np.all(sample == x0))
        self.assertEqual(len(mcmc._groups), 2)
        self.assertEqual(len(mcmc._groups[0]), 10)
        self.assertEqual(len(mcmc._groups[1]), 10)
        self.assertTrue(np.all(x0 == mcmc._active_sample))
        self.assertTrue(np.all(x0 == mcmc._groups[0][0]))
        self.assertEqual(len(mcmc._t_mu), 2)
        self.assertEqual(len(mcmc._t_Sigma), 2)
        self.assertEqual(len(mcmc._t_nu), 2)

        # Second run
        x = mcmc.ask()
        self.assertTrue(np.all(x == x0))
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(mcmc._active_sample_pi_log_pdf, fx)
        self.assertFalse(mcmc._prepare)

        # Third run
        x = mcmc.ask()
        self.assertFalse(mcmc._prepare)
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertTrue(sample is not None)
        self.assertTrue(np.all(mcmc._groups[0][0] == x))
        self.assertEqual(mcmc._index_active_sample, 1)
        self.assertTrue(np.all(mcmc._groups[0][1] == mcmc._active_sample))
        self.assertTrue(mcmc._prepare)

        # Fourth run
        x = mcmc.ask()
        self.assertTrue(np.all(mcmc._groups[0][1] == x))
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(fx, mcmc._active_sample_pi_log_pdf)
        self.assertTrue(np.all(mcmc._groups[0][1] == mcmc._active_sample))
        self.assertFalse(mcmc._prepare)
        self.assertEqual(sample, None)

        # Fifth run
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertTrue(sample is not None)
        self.assertTrue(np.all(mcmc._groups[0][1] == x))
        self.assertEqual(mcmc._index_active_sample, 2)
        self.assertTrue(np.all(mcmc._groups[0][2] == mcmc._active_sample))
        self.assertTrue(mcmc._prepare)

        # Test group transition
        while mcmc._index_active_sample != 9:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)

        x = mcmc.ask()
        self.assertTrue(np.all(mcmc._groups[0][9] == x))
        self.assertTrue(mcmc._prepare)
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(None, sample)

        while sample is None:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)

        self.assertTrue(np.all(mcmc._groups[0][9] == sample))
        self.assertEqual(mcmc._index_active_group, 1)
        self.assertEqual(mcmc._index_active_sample, 0)
        self.assertTrue(np.all(mcmc._groups[1][0] == mcmc._active_sample))

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
        mcmc = pints.SliceGeneralisedEllipticalMCMC(x0)

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
