#!/usr/bin/env python3
#
# Tests the basic methods of the Hamiltonian MCMC routine.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture


class TestHamiltonianMCMC(unittest.TestCase):
    """
    Tests the basic methods of the Hamiltonian MCMC routine.
    """

    def test_method(self):

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])

        # Create mcmc
        x0 = np.array([2, 2])
        sigma = [[3, 0], [0, 3]]
        mcmc = pints.HamiltonianMCMC(x0, sigma)

        # This method needs sensitivities
        self.assertTrue(mcmc.needs_sensitivities())

        # Set number of leapfrog steps
        ifrog = 10
        mcmc.set_leapfrog_steps(ifrog)

        # Perform short run
        chain = []
        for i in range(100 * ifrog):
            x = mcmc.ask()
            fx, gr = log_pdf.evaluateS1(x)
            reply = mcmc.tell((fx, gr))
            if reply is not None:
                y, fy, ac = reply
                if i >= 50 * ifrog:
                    chain.append(y)
                self.assertTrue(isinstance(ac, bool))
                if ac:
                    self.assertTrue(np.all(x == y))
                    self.assertEqual(fx, fy[0])
                    self.assertTrue(np.all(gr == fy[1]))

        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))

    def test_logging(self):
        # Test logging includes name and custom fields.

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = [np.array([2, 2]), np.array([8, 8])]

        mcmc = pints.MCMCController(
            log_pdf, 2, x0, method=pints.HamiltonianMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()

        self.assertIn('Hamiltonian Monte Carlo', text)
        self.assertIn(' Accept.', text)

    def test_flow(self):

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = np.array([2, 2])

        # Test initial proposal is first point
        mcmc = pints.HamiltonianMCMC(x0)
        self.assertTrue(np.all(mcmc.ask() == mcmc._x0))

        # Repeated asks
        self.assertRaises(RuntimeError, mcmc.ask)

        # Tell without ask
        mcmc = pints.HamiltonianMCMC(x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Repeated tells should fail
        x = mcmc.ask()
        mcmc.tell(log_pdf.evaluateS1(x))
        self.assertRaises(RuntimeError, mcmc.tell, log_pdf.evaluateS1(x))

        # Bad starting point
        mcmc = pints.HamiltonianMCMC(x0)
        mcmc.ask()
        self.assertRaises(
            ValueError, mcmc.tell, (-np.inf, np.array([1, 1])))

    def test_set_hyper_parameters(self):
        # Tests the parameter interface for this sampler.

        x0 = np.array([2, 2])
        mcmc = pints.HamiltonianMCMC(x0)

        # Test leapfrog parameters
        n = mcmc.leapfrog_steps()
        d = mcmc.leapfrog_step_size()
        self.assertIsInstance(n, int)
        self.assertTrue(len(d) == mcmc._n_parameters)

        mcmc.set_leapfrog_steps(n + 1)
        self.assertEqual(mcmc.leapfrog_steps(), n + 1)
        self.assertRaises(ValueError, mcmc.set_leapfrog_steps, 0)

        mcmc.set_leapfrog_step_size(0.5)
        self.assertEqual(mcmc.leapfrog_step_size()[0], 0.5)
        self.assertRaises(ValueError, mcmc.set_leapfrog_step_size, -1)

        self.assertEqual(mcmc.n_hyper_parameters(), 2)
        mcmc.set_hyper_parameters([n + 2, 2])
        self.assertEqual(mcmc.leapfrog_steps(), n + 2)
        self.assertEqual(mcmc.leapfrog_step_size()[0], 2)

        mcmc.set_epsilon(0.4)
        self.assertEqual(mcmc.epsilon(), 0.4)
        self.assertRaises(ValueError, mcmc.set_epsilon, -0.1)
        mcmc.set_leapfrog_step_size(1)
        self.assertEqual(len(mcmc.scaled_epsilon()), 2)
        self.assertEqual(mcmc.scaled_epsilon()[0], 0.4)
        self.assertEqual(len(mcmc.divergent_iterations()), 0)
        self.assertRaises(ValueError, mcmc.set_leapfrog_step_size, [1, 2, 3])

        mcmc.set_leapfrog_step_size([1.5, 3])
        self.assertEqual(mcmc.leapfrog_step_size()[0], 1.5)
        self.assertEqual(mcmc.leapfrog_step_size()[1], 3)

    def test_other_setters(self):
        # Tests other setters and getters.
        x0 = np.array([2, 2])
        mcmc = pints.HamiltonianMCMC(x0)
        self.assertRaises(ValueError, mcmc.set_hamiltonian_threshold, -0.3)
        threshold1 = mcmc.hamiltonian_threshold()
        self.assertEqual(threshold1, 10**3)
        threshold2 = 10
        mcmc.set_hamiltonian_threshold(threshold2)
        self.assertEqual(mcmc.hamiltonian_threshold(), threshold2)


if __name__ == '__main__':
    unittest.main()
