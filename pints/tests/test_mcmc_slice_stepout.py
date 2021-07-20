#!/usr/bin/env python3
#
# Tests the basic methods of the slice sampling with stepout routine.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy


class TestSliceStepout(unittest.TestCase):
    """
    Tests the slice sampling with stepout routine.
    """

    def test_ask_tell_flow(self):
        # Tests the ask-and-tell pattern

        # Create problem
        x0 = np.array([2, 4])
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Ask must return initial point
        mcmc = pints.SliceStepoutMCMC(x0)
        self.assertTrue(np.all(mcmc.ask() == x0))

        # Ask can't be called twice
        self.assertRaises(RuntimeError, mcmc.ask)

        # Tell
        fx0 = log_pdf(x0)
        x1, fx1, ac = mcmc.tell(fx0)
        self.assertTrue(np.all(x0 == x1))
        self.assertTrue(fx0 == fx1)

        # Check that the new slice has been constructed appropriately
        self.assertTrue(mcmc.current_slice_height() < fx1)

        # Can't tell twice
        self.assertRaises(RuntimeError, mcmc.tell, log_pdf(x0))

        # First point must be finite
        mcmc = pints.SliceStepoutMCMC(x0)
        mcmc.ask()
        self.assertRaises(ValueError, mcmc.tell, np.inf)

    def test_name(self):
        # Tests the sampler's name
        mcmc = pints.SliceStepoutMCMC(np.array([1, 2]))
        self.assertEqual(mcmc.name(), 'Slice Sampling - Stepout')

    def test_width(self):
        # Test width methods
        mcmc = pints.SliceStepoutMCMC(np.array([1, 2]))

        # Test set_width(), width()
        mcmc.set_width(2)
        self.assertTrue(np.all(mcmc.width() == np.array([2, 2])))
        mcmc.set_width([3, 5])
        self.assertTrue(np.all(mcmc.width() == np.array([3, 5])))
        with self.assertRaises(ValueError):
            mcmc.set_width(-1)
        with self.assertRaises(ValueError):
            mcmc.set_width([3, 3, 3, 3])

    def test_expansion(self):
        # Test expansion step methods

        mcmc = pints.SliceStepoutMCMC(np.array([1, 2]))
        mcmc.set_expansion_steps(3)
        self.assertEqual(mcmc.expansion_steps(), 3.)
        with self.assertRaises(ValueError):
            mcmc.set_expansion_steps(-1)

    def test_overrelaxed(self):
        # Test overrelaxation methods

        mcmc = pints.SliceStepoutMCMC(np.array([1, 2]))
        mcmc.set_prob_overrelaxed(0.5)
        self.assertEqual(mcmc.prob_overrelaxed(), 0.5)
        with self.assertRaises(ValueError):
            mcmc.set_prob_overrelaxed(-1)
        with self.assertRaises(ValueError):
            mcmc.set_prob_overrelaxed(4)

    def test_bisection(self):
        # Test bisection methods

        mcmc = pints.SliceStepoutMCMC(np.array([1, 2]))
        mcmc.set_bisection_steps(40)
        self.assertEqual(mcmc.bisection_steps(), 40)
        with self.assertRaises(ValueError):
            mcmc.set_bisection_steps(-30)

    def test_hyper_parameters(self):
        # Test the hyper parameter interface

        mcmc = pints.SliceStepoutMCMC(np.array([1, 2]))
        self.assertEqual(mcmc.n_hyper_parameters(), 4)
        mcmc.set_hyper_parameters([3, 100, .7, 50])
        self.assertTrue((np.all(mcmc.width() == np.array([3, 3]))))
        self.assertEqual(mcmc.expansion_steps(), 100)
        self.assertEqual(mcmc.prob_overrelaxed(), 0.7)
        self.assertEqual(mcmc.bisection_steps(), 50)

    def test_run(self):
        # Test a short run

        # Create log pdf
        log_pdf = pints.toy.MultimodalGaussianLogPDF(
            modes=[[0, 2], [0, 7], [5, 0], [4, 4]])

        # Create and configure sampler
        mcmc = pints.SliceStepoutMCMC(np.array([1, 1]))
        mcmc.set_prob_overrelaxed(0.6)
        mcmc.set_width(30)
        mcmc.set_bisection_steps(2)

        # Run a few iterations of the sampler
        np.random.seed(456)
        n = 20
        chain = []
        while len(chain) < n:
            x = mcmc.ask()
            fx = log_pdf(x)
            reply = mcmc.tell(fx)
            if reply is not None:
                y, fy, ac = reply
                chain.append(y)
                self.assertEqual(fy, log_pdf(y))

        self.assertEqual(np.shape(chain), (n, 2))


if __name__ == '__main__':
    unittest.main()
