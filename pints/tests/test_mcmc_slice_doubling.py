#!/usr/bin/env python3
#
# Tests the basic methods of the slice sampling with doubling routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

import unittest
import numpy as np

import pints
import pints.toy


class TestSliceDoubling(unittest.TestCase):
    """
    Tests the slice sampling with doubling routine.
    """

    def test_ask_tell_flow(self):
        # Tests the ask-and-tell pattern

        # Create problem
        x0 = np.array([2, 4])
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Ask must return initial point
        mcmc = pints.SliceDoublingMCMC(x0)
        self.assertTrue(np.all(mcmc.ask() == x0))

        # Ask can't be called twice
        self.assertRaises(RuntimeError, mcmc.ask)

        # Tell
        fx = log_pdf(x0)
        x1 = mcmc.tell(fx)
        self.assertTrue(np.all(x0 == x1))
        self.assertTrue(mcmc.current_log_pdf() == fx)

        # Check that the new slice has been constructed appropriately
        self.assertTrue(
            mcmc.current_slice_height() < mcmc.current_log_pdf())

        # Can't tell twice
        self.assertRaises(RuntimeError, mcmc.tell, log_pdf(x0))

        # First point must be finite
        mcmc = pints.SliceDoublingMCMC(x0)
        mcmc.ask()
        self.assertRaises(ValueError, mcmc.tell, np.inf)

    def test_name(self):
        # Tests the sampler's name
        mcmc = pints.SliceDoublingMCMC(np.array([1, 2]))
        self.assertEqual(mcmc.name(), 'Slice Sampling - Doubling')

    def test_width(self):
        # Test width methods
        mcmc = pints.SliceDoublingMCMC(np.array([1, 2]))

        # Test set_width(), width()
        mcmc.set_width(2)
        self.assertTrue(np.all(mcmc.width() == np.array([2, 2])))
        mcmc.set_width([5, 8])
        self.assertTrue(np.all(mcmc.width() == np.array([5, 8])))
        with self.assertRaises(ValueError):
            mcmc.set_width(-1)
        with self.assertRaises(ValueError):
            mcmc.set_width([3, 3, 3, 3])

    def test_expansion(self):
        # Test expansion step methods

        mcmc = pints.SliceDoublingMCMC(np.array([1, 2]))
        mcmc.set_expansion_steps(3)
        self.assertEqual(mcmc.expansion_steps(), 3.)
        with self.assertRaises(ValueError):
            mcmc.set_expansion_steps(-1)

    def test_hyper_parameters(self):
        # Test the hyper parameter interface

        mcmc = pints.SliceDoublingMCMC(np.array([1, 2]))
        self.assertEqual(mcmc.n_hyper_parameters(), 2)
        mcmc.set_hyper_parameters([3, 100])
        self.assertTrue((np.all(mcmc.width() == np.array([3, 3]))))
        self.assertEqual(mcmc.expansion_steps(), 100)

    def test_run(self):
        # Test a short run

        # Create log pdf
        log_pdf = pints.toy.MultimodalGaussianLogPDF(
            modes=[[1, 1], [1, 4], [5, 4], [1, 4]])

        # Create mcmc
        mcmc = pints.SliceDoublingMCMC(np.array([1, 2]))
        mcmc.set_width(2)

        # Run a few iterations of the sampler
        # But enough to hit all branches...
        np.random.seed(123)
        n = 42
        chain = []
        while len(chain) < n:
            sample = mcmc.tell(log_pdf(mcmc.ask()))
            if sample is not None:
                chain.append(np.copy(sample))
        self.assertEqual(np.shape(chain), (n, 2))


if __name__ == '__main__':
    unittest.main()
