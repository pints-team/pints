#!/usr/bin/env python3
#
# Tests the basic methods of the Hamiltonian MCMC routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture

debug = False


class TestHamiltonianMCMC(unittest.TestCase):
    """
    Tests the basic methods of the Hamiltonian MCMC routine.
    """

    def test_method(self):

        # Create log pdf
        log_pdf = pints.toy.NormalLogPDF([5, 5], [[4, -1], [1, 3]])

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
            sample = mcmc.tell((fx, gr))
            if i >= 50 * ifrog and sample is not None:
                chain.append(sample)
        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))

    def test_logging(self):
        """
        Test logging includes name and custom fields.
        """
        log_pdf = pints.toy.NormalLogPDF([5, 5], [[4, -1], [1, 3]])
        x0 = [np.array([2, 2]), np.array([8, 8])]

        mcmc = pints.MCMCSampling(log_pdf, 2, x0, method=pints.HamiltonianMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()

        self.assertIn('Hamiltonian MCMC', text)
        self.assertIn(' Accept.', text)

    def test_flow(self):

        log_pdf = pints.toy.NormalLogPDF([5, 5], [[4, -1], [1, 3]])
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
            ValueError, mcmc.tell, (float('-inf'), np.array([1, 1])))

    def test_set_hyper_parameters(self):
        """
        Tests the parameter interface for this sampler.
        """
        x0 = np.array([2, 2])
        mcmc = pints.HamiltonianMCMC(x0)

        # Test leapfrog parameters
        n = mcmc.leapfrog_steps()
        d = mcmc.leapfrog_step_size()
        self.assertIsInstance(n, int)
        self.assertIsInstance(d, float)

        mcmc.set_leapfrog_steps(n + 1)
        self.assertEqual(mcmc.leapfrog_steps(), n + 1)
        self.assertRaises(ValueError, mcmc.set_leapfrog_steps, 0)

        mcmc.set_leapfrog_step_size(d * 0.5)
        self.assertEqual(mcmc.leapfrog_step_size(), d * 0.5)
        self.assertRaises(ValueError, mcmc.set_leapfrog_step_size, -1)

        self.assertEqual(mcmc.n_hyper_parameters(), 2)
        mcmc.set_hyper_parameters([n + 2, d * 2])
        self.assertEqual(mcmc.leapfrog_steps(), n + 2)
        self.assertEqual(mcmc.leapfrog_step_size(), d * 2)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
