#!/usr/bin/env python3
#
# Tests the basic methods of the Covariance-Adaptive Slice Sampling:
# Rank Shrinking.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy


class TestSliceRankShrinking(unittest.TestCase):
    """
    Tests the basic methods of the Slice Sampling Rank Shrinking routine.
    """
    def test_first_run(self):
        # Tests the very first run of the sampler.

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceRankShrinkingMCMC(x0)

        # Ask() should fail if _ready_for_tell flag is True
        x = mcmc.ask()
        with self.assertRaises(RuntimeError):
            mcmc.ask()

        # Check whether first iteration of ask() returns x0
        self.assertTrue(np.all(x == x0))

        # Calculate log pdf for x0
        fx0, grad0 = log_pdf.evaluateS1(x0)

        # Test first iteration of tell(). The first point in the chain
        # should be x0
        x1, (fx1, grad1), ac = mcmc.tell((fx0, grad0))
        self.assertTrue(np.all(x1 == x0))

        # We update the _current_log_pdf value used to generate the new slice
        self.assertEqual(fx0, fx1)

        # Check that the new slice has been constructed appropriately
        self.assertTrue(mcmc.current_slice_height() < fx1)

        # Tell() should fail when fx is infinite
        mcmc = pints.SliceRankShrinkingMCMC(x0)
        mcmc.ask()
        with self.assertRaises(ValueError):
            mcmc.tell((np.inf, grad0))

        # Tell() should fail when _ready_for_tell is False
        with self.assertRaises(RuntimeError):
            mcmc.tell((fx0, grad0))

        # Test sensitivities
        self.assertTrue(mcmc.needs_sensitivities())

    def test_basic(self):
        # Test basic methods of the class.

        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceRankShrinkingMCMC(x0)

        # Test name
        self.assertEqual(
            mcmc.name(),
            'Slice Sampling - Covariance-Adaptive: Rank Shrinking.')

        # Test set_sigma_c(), sigma_c()
        mcmc.set_sigma_c(6)
        self.assertEqual(mcmc.sigma_c(), 6)
        with self.assertRaises(ValueError):
            mcmc.set_sigma_c(-1)

        # Test number of hyperparameters
        self.assertEqual(mcmc.n_hyper_parameters(), 1)

        # Test setting hyperparameters
        mcmc.set_hyper_parameters([33])
        self.assertEqual(mcmc.sigma_c(), 33)

    def test_run(self):
        # Tests complete run.

        # Create log pdf
        log_pdf = pints.toy.MultimodalGaussianLogPDF(
            modes=[[0, 2], [0, 7], [5, 0], [4, 4]])

        # Create non-adaptive mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceRankShrinkingMCMC(x0)
        mcmc.set_sigma_c(3)

        # Run multiple iterations of the sampler
        chain = []
        while len(chain) < 100:
            x = mcmc.ask()
            fx, grad = log_pdf.evaluateS1(x)
            reply = mcmc.tell((fx, grad))
            if reply is not None:
                y, fy, ac = reply
                chain.append(y)
                self.assertTrue(np.all(x == y))
                self.assertEqual(fx, fy[0])
                self.assertTrue(np.all(grad == fy[1]))
        self.assertEqual(np.shape(chain), (100, 2))


if __name__ == '__main__':
    unittest.main()
