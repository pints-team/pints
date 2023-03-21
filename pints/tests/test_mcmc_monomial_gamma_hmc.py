#!/usr/bin/env python3
#
# Tests the basic methods of the Monomial-gamma HMC routine.
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


class TestMonomialGammaHamiltonianMCMC(unittest.TestCase):
    """
    Tests the basic methods of the Monomial-gamma HMC MCMC routine.
    """

    def test_method(self):

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])

        # Create mcmc
        x0 = np.array([2, 2])
        sigma = [[3, 0], [0, 3]]
        mcmc = pints.MonomialGammaHamiltonianMCMC(x0, sigma)

        # This method needs sensitivities
        self.assertTrue(mcmc.needs_sensitivities())

        # Set number of leapfrog steps
        ifrog = 4
        mcmc.set_leapfrog_steps(ifrog)

        # Perform short run
        chain = []
        for i in range(50 * ifrog):
            x = mcmc.ask()
            fx, gr = log_pdf.evaluateS1(x)
            reply = mcmc.tell((fx, gr))
            if reply is not None:
                y, fy, ac = reply
                if i >= 25 * ifrog:
                    chain.append(y)
                self.assertTrue(isinstance(ac, bool))
                if ac:
                    self.assertTrue(np.all(x == y))
                    self.assertEqual(fx, fy[0])
                    self.assertTrue(np.all(gr == fy[1]))

        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 25)
        self.assertEqual(chain.shape[1], len(x0))

    def test_logging(self):
        # Test logging includes name and custom fields.

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = [np.array([2, 2]), np.array([8, 8])]

        mcmc = pints.MCMCController(
            log_pdf, 2, x0, method=pints.MonomialGammaHamiltonianMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()

        self.assertIn('Monomial-Gamma Hamiltonian Monte Carlo', text)
        self.assertIn(' Accept.', text)

    def test_flow(self):

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = np.array([2, 2])

        # Test initial proposal is first point
        mcmc = pints.MonomialGammaHamiltonianMCMC(x0)
        self.assertTrue(np.all(mcmc.ask() == mcmc._x0))

        # Repeated asks
        self.assertRaises(RuntimeError, mcmc.ask)

        # Tell without ask
        mcmc = pints.MonomialGammaHamiltonianMCMC(x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Repeated tells should fail
        x = mcmc.ask()
        mcmc.tell(log_pdf.evaluateS1(x))
        self.assertRaises(RuntimeError, mcmc.tell, log_pdf.evaluateS1(x))

        # Bad starting point
        mcmc = pints.MonomialGammaHamiltonianMCMC(x0)
        mcmc.ask()
        self.assertRaises(
            ValueError, mcmc.tell, (-np.inf, np.array([1, 1])))

    def test_kinetic_energy(self):
        # Tests kinetic energy values and derivatives

        x0 = np.array([2, 2])
        model = pints.MonomialGammaHamiltonianMCMC(x0)
        model.ask()

        # kinetic energy
        ke = model._K([-3.5, 0.5], 1.5, 1.7, 0.7)
        self.assertAlmostEqual(4.4280785731741243, ke)

        # derivatives
        self.assertAlmostEqual(model._K_deriv_indiv(1.3, 0.5, 0.3, 1.3),
                               0.38513079718715132)
        derivs = model._K_deriv_indiv([-3.5, 0.5], 1.5, 1.7, 0.7)
        self.assertAlmostEqual(derivs[0], -0.62264078154277391)
        self.assertAlmostEqual(derivs[1], 0.77273306758997307)
        self.assertAlmostEqual(len(derivs), 2)

        # sample momentum from ke function
        mom = model._sample_momentum()
        self.assertEqual(len(mom), 2)
        self.assertTrue(mom[0] != mom[1])

    def test_set_hyper_parameters(self):
        # Tests the parameter interface for this sampler.

        x0 = np.array([2, 2])
        mcmc = pints.MonomialGammaHamiltonianMCMC(x0)

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

        self.assertEqual(mcmc.n_hyper_parameters(), 5)
        mcmc.set_hyper_parameters([n + 2, 2, 0.4, 2.3, 1.3])
        self.assertEqual(mcmc.leapfrog_steps(), n + 2)
        self.assertEqual(mcmc.leapfrog_step_size()[0], 2)
        self.assertEqual(mcmc.a(), 0.4)
        self.assertEqual(mcmc.c(), 2.3)
        self.assertEqual(mcmc.mass(), 1.3)

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

        a = 1.9
        mcmc.set_a(a)
        self.assertEqual(mcmc.a(), a)
        self.assertRaises(ValueError, mcmc.set_a, -0.9)

        c = 3.5
        mcmc.set_c(c)
        self.assertEqual(mcmc.c(), c)
        self.assertRaises(ValueError, mcmc.set_c, -0.1)

        m = 2.9
        mcmc.set_mass(m)
        self.assertEqual(mcmc.mass(), m)
        self.assertRaises(ValueError, mcmc.set_mass, -1.8)

    def test_other_setters(self):
        # Tests other setters and getters.
        x0 = np.array([2, 2])
        mcmc = pints.MonomialGammaHamiltonianMCMC(x0)
        self.assertRaises(ValueError, mcmc.set_hamiltonian_threshold, -0.3)
        threshold1 = mcmc.hamiltonian_threshold()
        self.assertEqual(threshold1, 10**3)
        threshold2 = 10
        mcmc.set_hamiltonian_threshold(threshold2)
        self.assertEqual(mcmc.hamiltonian_threshold(), threshold2)


if __name__ == '__main__':
    unittest.main()
