#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Tests the basic methods of the Slice Sampling routine with Hyperrectangles.
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
    Tests the basic methods of the Hyperrectangles-based Slice Sampling
    routine.
    """

    def test_first_run(self):
        # Tests the very first run of the sampler.

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceHyperrectanglesMCMC(x0)

        # Ask() should fail if _ready_for_tell flag is True
        x = mcmc.ask()
        with self.assertRaises(RuntimeError):
            mcmc.ask()

        # Check whether first iteration of ask() returns x0
        self.assertTrue(np.all(x == x0))

        # Calculate log pdf for x0
        fx, grad = log_pdf.evaluateS1(x0)

        # Test first iteration of tell(). The first point in the chain
        # should be x0
        sample = mcmc.tell((fx, grad))
        self.assertTrue(np.all(sample == x0))

        # We update the _current_log_pdf value used to generate the new slice
        self.assertEqual(mcmc.current_log_pdf(), fx)

        # Check that the new slice has been constructed appropriately
        self.assertTrue(
            mcmc.current_slice_height() < mcmc.current_log_pdf())

        # Tell() should fail when fx is infinite
        mcmc = pints.SliceStepoutMCMC(x0)
        mcmc.ask()
        with self.assertRaises(ValueError):
            fx = np.inf
            mcmc.tell(fx)

        # Tell() should fail when _ready_for_tell is False
        with self.assertRaises(RuntimeError):
            mcmc.tell(fx)

    def test_basic(self):
        # Test basic methods of the class.

        # Create mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceHyperrectanglesMCMC(x0)

        # Test name
        self.assertEqual(mcmc.name(), 'Slice Sampling - Hyperrectangles')

        # Test set_width(), width()
        mcmc.set_width(2)
        self.assertTrue(np.all(mcmc.width() == np.array([2, 2])))
        mcmc.set_width([3, 5])
        self.assertTrue(np.all(mcmc.width() == np.array([3, 5])))
        with self.assertRaises(ValueError):
            mcmc.set_width(-1)
        with self.assertRaises(ValueError):
            mcmc.set_width([3, 3, 3, 3])

        # Test number of hyperparameters
        self.assertEqual(mcmc.n_hyper_parameters(), 2)

        # Test setting hyperparameters
        mcmc.set_hyper_parameters([3, 1])
        self.assertTrue((np.all(mcmc.width() == np.array([3, 3]))))
        self.assertEqual(mcmc.adaptive_shrinking(), True)

    def test_run(self):
        # Tests complete adaptive and non-adaptive runs.

        # Create log pdf
        log_pdf = pints.toy.MultimodalGaussianLogPDF(
            modes=[[0, 2], [0, 7], [5, 0], [4, 4]])

        # Create non-adaptive mcmc
        x0 = np.array([1, 1])
        mcmc = pints.SliceHyperrectanglesMCMC(x0)
        mcmc.set_width(30)
        self.assertEqual(mcmc.adaptive_shrinking(), False)

        # Run multiple iterations of the sampler
        chain = []
        while len(chain) < 50:
            x = mcmc.ask()
            fx, grad = log_pdf.evaluateS1(x)
            sample = mcmc.tell((fx, grad))
            if sample is not None:
                chain.append(np.copy(sample))
        self.assertEqual(np.shape(chain), (50, 2))

        # Create adaptive mcmc
        mcmc = pints.SliceHyperrectanglesMCMC(x0)
        mcmc.set_width(30)
        mcmc.set_adaptive_shrinking(True)
        self.assertEqual(mcmc.adaptive_shrinking(), True)

        # Run multiple iterations of the sampler
        chain = []
        while len(chain) < 50:
            x = mcmc.ask()
            fx, grad = log_pdf.evaluateS1(x)
            sample = mcmc.tell((fx, grad))
            if sample is not None:
                chain.append(np.copy(sample))
        self.assertEqual(np.shape(chain), (50, 2))


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
