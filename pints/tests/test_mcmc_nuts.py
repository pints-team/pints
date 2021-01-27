#!/usr/bin/env python3
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

        # Perform short run, test logging while we are at it
        logger = pints.Logger()
        logger.set_stream(None)
        mcmc._log_init(logger)
        chain = []
        for i in range(2 * mcmc.number_adaption_steps()):
            x = mcmc.ask()
            fx, gr = log_pdf.evaluateS1(x)
            sample = mcmc.tell((fx, gr))
            mcmc._log_write(logger)

            if sample is not None:
                chain.append(sample)
            if np.all(sample == x):
                self.assertEqual(mcmc.current_log_pdf(), fx)

        chain = np.array(chain)
        self.assertGreater(chain.shape[0], 1)
        self.assertEqual(chain.shape[1], len(x0))

    def test_method_with_dense_mass(self):

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])

        # Create mcmc
        x0 = np.array([2, 2])
        sigma = [[3, 0], [0, 3]]
        mcmc = pints.NoUTurnMCMC(x0, sigma)
        mcmc.set_use_dense_mass_matrix(True)

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

    def test_model_that_gives_nan(self):
        # This model will return a nan in the gradient evaluation, which
        # originally tripped up the find_reasonable_epsilon function in nuts.
        # Run it for a bit so that we get coverage on the if statement!

        model = pints.toy.LogisticModel()
        real_parameters = model.suggested_parameters()
        times = model.suggested_parameters()
        org_values = model.simulate(real_parameters, times)
        np.random.seed(1)
        noise = 0.2
        values = org_values + np.random.normal(0, noise, org_values.shape)
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, noise)

        log_prior = pints.UniformLogPrior(
            [0.01, 40],
            [0.2, 60]
        )

        log_posterior = pints.LogPosterior(log_likelihood, log_prior)

        xs = [real_parameters * 1.1]
        nuts_mcmc = pints.MCMCController(log_posterior,
                                         len(xs), xs,
                                         method=pints.NoUTurnMCMC)

        nuts_mcmc.set_max_iterations(10)
        nuts_mcmc.set_log_to_screen(False)
        nuts_chains = nuts_mcmc.run()

        self.assertFalse(np.isnan(np.sum(nuts_chains)))

    def test_method_near_boundary(self):

        # Create log pdf
        log_pdf = pints.UniformLogPrior([0, 0], [1, 1])

        # Create mcmc
        x0 = np.array([0.999, 0.999])
        sigma = [[1, 0], [0, 1]]
        mcmc = pints.NoUTurnMCMC(x0, sigma)

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
        self.assertGreater(mcmc.divergent_iterations().shape[0], 0)

    def test_logging(self):
        # Test logging includes name and custom fields.
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
        # Tests the parameter interface for this sampler.
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

    def test_build_tree_nan(self):
        # This method gives nan in the hamiltonian_dash in the build_tree function
        # Needed for coverage

        model = pints.toy.LogisticModel()
        real_parameters = np.array([0.015, 20])
        times = np.linspace(0, 1000, 50)
        org_values = model.simulate(real_parameters, times)
        np.random.seed(1)
        noise = 0.1
        values = org_values + np.random.normal(0, noise, org_values.shape)
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, noise)

        log_prior = pints.UniformLogPrior(
            [0.0001, 1],
            [1, 500]
        )

        log_posterior = pints.LogPosterior(log_likelihood, log_prior)

        xs = [[0.36083914, 1.99013825]]
        nuts_mcmc = pints.MCMCController(log_posterior,
                                         len(xs), xs,
                                         method=pints.NoUTurnMCMC)

        nuts_mcmc.set_max_iterations(50)
        nuts_mcmc.set_log_to_screen(False)
        np.seed(5)
        nuts_chains = nuts_mcmc.run()

        self.assertFalse(np.isnan(np.sum(nuts_chains)))


if __name__ == '__main__':
    unittest.main()
