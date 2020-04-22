#!/usr/bin/env python3
#
# Tests the basic methods of the Hamiltonian MCMC routine.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import unittest
import sys

import pints
import pints.toy

from shared import StreamCapture


@unittest.skipIf(sys.hexversion < 0x03030000, 'No NUTS on Python < 3.3')
class TestNutsMCMC(unittest.TestCase):
    """
    Tests the basic methods of the No-U-Turn MCMC sampler.
    """

    def test_method(self):

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])

        # Create mcmc
        x0 = np.array([2, 2])
        sigma = [[3, 0], [0, 3]]
        mcmc = pints.NoUTurnMCMC(x0, sigma)

        # This method needs sensitivities
        self.assertTrue(mcmc.needs_sensitivities())

        # Perform short run
        chain = []
        for i in range(2 * mcmc.number_adaption_steps()):
            x = mcmc.ask()
            fx, gr = log_pdf.evaluateS1(x)
            sample = mcmc.tell((fx, gr))
            if sample is not None:
                chain.append(sample)
            if np.all(sample == x):
                self.assertEqual(mcmc.current_log_pdf(), fx)

        chain = np.array(chain)
        self.assertGreater(chain.shape[0], 1)
        self.assertEqual(chain.shape[1], len(x0))

    def test_logging(self):
        """
        Test logging includes name and custom fields.
        """
        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = [np.array([2, 2]), np.array([8, 8])]

        mcmc = pints.MCMCController(
            log_pdf, 2, x0, method=pints.NoUTurnMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()

        self.assertIn('No-U-Turn MCMC', text)

    def test_flow(self):

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = np.array([2, 2])

        # Test initial proposal is first point
        mcmc = pints.NoUTurnMCMC(x0)
        self.assertTrue(np.all(mcmc.ask() == mcmc._x0))

        # Test current log pdf is None (no tell yet)
        self.assertIsNone(mcmc.current_log_pdf())

        # Repeated asks
        self.assertRaises(RuntimeError, mcmc.ask)

        # Tell without ask
        mcmc = pints.NoUTurnMCMC(x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Repeated tells should fail
        x = mcmc.ask()
        mcmc.tell(log_pdf.evaluateS1(x))
        self.assertRaises(RuntimeError, mcmc.tell, log_pdf.evaluateS1(x))

        # Cannot set delta while running
        self.assertRaises(RuntimeError, mcmc.set_delta, 0.5)

        # Cannot set number of adpation steps while running
        self.assertRaises(RuntimeError, mcmc.set_number_adaption_steps, 500)

        # Bad starting point
        mcmc = pints.NoUTurnMCMC(x0)
        mcmc.ask()
        self.assertRaises(
            ValueError, mcmc.tell, (float('-inf'), np.array([1, 1])))

    def test_set_hyper_parameters(self):
        """
        Tests the parameter interface for this sampler.
        """
        x0 = np.array([2, 2])
        mcmc = pints.NoUTurnMCMC(x0)

        # Test delta parameter
        delta = mcmc.delta()
        self.assertIsInstance(delta, float)

        mcmc.set_delta(0.5)
        self.assertEqual(mcmc.delta(), 0.5)

        # delta between 0 and 1
        self.assertRaises(ValueError, mcmc.set_delta, -0.1)
        self.assertRaises(ValueError, mcmc.set_delta, 1.1)

        # Test number of adaption steps
        n = mcmc.number_adaption_steps()
        self.assertIsInstance(n, int)

        mcmc.set_number_adaption_steps(100)
        self.assertEqual(mcmc.number_adaption_steps(), 100)

        # should convert to int
        mcmc.set_number_adaption_steps(1.4)
        self.assertEqual(mcmc.number_adaption_steps(), 1)

        # number_adaption_steps is non-negative
        self.assertRaises(ValueError, mcmc.set_number_adaption_steps, -100)

        # test max tree depth
        mcmc.set_max_tree_depth(20)
        self.assertEqual(mcmc.max_tree_depth(), 20)
        self.assertRaises(ValueError, mcmc.set_max_tree_depth, -1)

        # test use_dense_mass_matrix
        mcmc.set_use_dense_mass_matrix(True)
        self.assertEqual(mcmc.use_dense_mass_matrix(), True)

        # hyper param interface
        self.assertEqual(mcmc.n_hyper_parameters(), 1)
        mcmc.set_hyper_parameters([2])
        self.assertEqual(mcmc.number_adaption_steps(), 2)

    def test_other_setters(self):
        # Tests other setters and getters.
        x0 = np.array([2, 2])
        mcmc = pints.NoUTurnMCMC(x0)
        self.assertRaises(ValueError, mcmc.set_hamiltonian_threshold, -0.3)
        threshold1 = mcmc.hamiltonian_threshold()
        self.assertEqual(threshold1, 10**3)
        threshold2 = 10
        mcmc.set_hamiltonian_threshold(threshold2)
        self.assertEqual(mcmc.hamiltonian_threshold(), threshold2)


if __name__ == '__main__':
    unittest.main()
