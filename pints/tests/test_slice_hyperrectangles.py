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
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

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
        sample_0 = mcmc.tell(fx)
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
        fx = log_pdf.evaluateS1(x)[0]
        sample_0 = mcmc.tell(fx)
        self.assertTrue(np.all(sample_0 == x0))
        self.assertFalse(mcmc._hyperrectangle_positioned)

        # Next step
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample_1 = mcmc.tell(fx)
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
        mcmc.set_w(.5)
        print(mcmc._w)

        # Run multiple iterations of the sampler
        chain = []
        while len(chain) < 10000:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
            if sample is not None:
                chain.append(np.copy(sample))

        # Fit Multivariate Gaussian to chain samples
        print(np.mean(chain, axis=0))
        print(np.cov(chain, rowvar=0))