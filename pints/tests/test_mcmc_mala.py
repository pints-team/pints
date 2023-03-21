#!/usr/bin/env python3
#
# Tests the basic methods of the MALA MCMC routine.
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


class TestMALAMCMC(unittest.TestCase):
    """
    Tests the basic methods of the MALA MCMC routine.
    """

    def test_method(self):

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])

        # Create mcmc
        x0 = np.array([2, 2])
        sigma = [[3, 0], [0, 3]]
        mcmc = pints.MALAMCMC(x0, sigma)

        # This method needs sensitivities
        self.assertTrue(mcmc.needs_sensitivities())

        # Perform short run
        chain = []
        for i in range(100):
            x = mcmc.ask()
            fx, gr = log_pdf.evaluateS1(x)
            reply = mcmc.tell((fx, gr))
            if reply is not None:
                y, fy, ac = reply
                if i >= 50:
                    chain.append(y)
                self.assertTrue(isinstance(ac, bool))
                if ac:
                    self.assertTrue(np.all(x == y))
                    self.assertEqual(fx, fy[0])
                    self.assertTrue(np.all(gr == fy[1]))

        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))
        self.assertTrue(mcmc.acceptance_rate() >= 0.0 and
                        mcmc.acceptance_rate() <= 1.0)

        mcmc._proposed = [1, 3]
        self.assertRaises(RuntimeError, mcmc.tell, (fx, gr))

    def test_logging(self):
        # Test logging includes name and custom fields.

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = [np.array([2, 2]), np.array([8, 8])]

        mcmc = pints.MCMCController(log_pdf, 2, x0, method=pints.MALAMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()

        self.assertIn('Metropolis-Adjusted Langevin Algorithm (MALA)',
                      text)
        self.assertIn(' Accept.', text)

    def test_flow(self):

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = np.array([2, 2])

        # Test initial proposal is first point
        mcmc = pints.MALAMCMC(x0)
        self.assertTrue(np.all(mcmc.ask() == mcmc._x0))

        # Repeated asks
        self.assertRaises(RuntimeError, mcmc.ask)

        # Tell without ask
        mcmc = pints.MALAMCMC(x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Repeated tells should fail
        x = mcmc.ask()
        mcmc.tell(log_pdf.evaluateS1(x))
        self.assertRaises(RuntimeError, mcmc.tell, log_pdf.evaluateS1(x))

        # Bad starting point
        mcmc = pints.MALAMCMC(x0)
        mcmc.ask()
        self.assertRaises(
            ValueError, mcmc.tell, (-np.inf, np.array([1, 1])))

        # Test initialisation twice
        mcmc = pints.MALAMCMC(x0)
        mcmc._running = True
        self.assertRaises(RuntimeError, mcmc._initialise)

    def test_set_hyper_parameters(self):
        # Tests the parameter interface for this sampler.

        x0 = np.array([2, 2])
        mcmc = pints.MALAMCMC(x0)
        self.assertTrue(np.array_equal(
                        mcmc._scale_vector,
                        np.diag(mcmc._sigma0))
                        )
        self.assertTrue(np.array_equal(mcmc.epsilon(),
                        0.2 * np.diag(mcmc._sigma0)))

        self.assertEqual(mcmc.n_hyper_parameters(), 1)
        mcmc.set_hyper_parameters([[3, 2]])
        self.assertTrue(np.array_equal(mcmc.epsilon(), [3, 2]))

        mcmc._step_size = 5
        mcmc._scale_vector = np.array([3, 7])
        mcmc._epsilon = None
        mcmc.set_epsilon()
        self.assertTrue(np.array_equal(mcmc.epsilon(), [15, 35]))
        mcmc.set_epsilon([0.4, 0.5])
        self.assertTrue(np.array_equal(mcmc.epsilon(), [0.4, 0.5]))
        self.assertRaises(ValueError, mcmc.set_epsilon, 3.0)
        self.assertRaises(ValueError, mcmc.set_epsilon, [-2.0, 1])


if __name__ == '__main__':
    unittest.main()
