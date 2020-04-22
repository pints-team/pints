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

import pints


class TestDualAveraging(unittest.TestCase):
    """
    Tests the dual averaging adaption used in the NUTS sampler.
    """

    def test_dual_averaging(self):

        num_warmup_steps = 200
        target_accept_prob = 0.5
        init_epsilon = 1.0
        init_inv_mass_matrix = np.array([[1, 0], [0, 1]])
        target_mass_matrix = np.array([[10, 0], [0, 10]])

        with self.assertRaises(ValueError):
            averager = pints.DualAveragingAdaption(10,
                                                   target_accept_prob,
                                                   init_epsilon,
                                                   init_inv_mass_matrix)

        averager = pints.DualAveragingAdaption(num_warmup_steps,
                                               target_accept_prob,
                                               init_epsilon,
                                               init_inv_mass_matrix)

        self.assertEqual(averager._epsilon, init_epsilon)
        np.testing.assert_array_equal(
            averager.get_inv_mass_matrix(), init_inv_mass_matrix)
        self.assertEqual(averager._counter, 0)

        initial_window = 75
        base_window = 25
        terminal_window = 50

        self.assertEqual(averager._next_window, initial_window + base_window)
        self.assertEqual(averager._adapting, True)

        def fake_accept_prob(epsilon):
            return 1.0 / (10.0 * epsilon)

        stored_x = np.empty((2, base_window))
        for i in range(averager._next_window - 1):
            x = np.random.multivariate_normal(np.zeros(2) + 123,
                                              target_mass_matrix)
            restart = averager.step(x, fake_accept_prob(averager._epsilon))
            self.assertFalse(restart)
            if i >= averager._initial_window:
                stored_x[:, i - averager._initial_window] = x

        np.testing.assert_array_equal(
            averager.get_inv_mass_matrix(), init_inv_mass_matrix)
        x = np.random.multivariate_normal(np.zeros(2) + 123,
                                          target_mass_matrix)

        np.testing.assert_array_equal(averager._samples[:, :-1],
                                      stored_x[:, :-1])
        restart = averager.step(x, fake_accept_prob(averager._epsilon))
        self.assertTrue(restart)
        stored_x[:, -1] = x

        cov = np.cov(stored_x)
        n = base_window
        p = 2
        adapted_cov = (n / (n + 5.0)) * cov + \
            1e-3 * (5.0 / (n + 5.0)) * np.eye(p)
        np.testing.assert_array_equal(averager.get_inv_mass_matrix(),
                                      adapted_cov)
        np.testing.assert_array_equal(averager.get_mass_matrix(),
                                      np.linalg.inv(adapted_cov))
        self.assertAlmostEqual(fake_accept_prob(averager._epsilon),
                               target_accept_prob, 1)

        self.assertEqual(averager._counter, initial_window + base_window)
        self.assertEqual(averager._next_window,
                         num_warmup_steps - terminal_window)

        for i in range(averager._next_window - averager._counter):
            x = np.random.multivariate_normal(np.zeros(2) + 123,
                                              target_mass_matrix)
            averager.step(x, fake_accept_prob(averager._epsilon))

        self.assertEqual(averager._counter, num_warmup_steps - terminal_window)
        self.assertEqual(averager._next_window, num_warmup_steps)

        for i in range(averager._next_window - averager._counter):
            x = np.random.multivariate_normal(np.zeros(2) + 123,
                                              target_mass_matrix)
            averager.step(x, fake_accept_prob(averager._epsilon))

        self.assertEqual(averager._counter, num_warmup_steps)
        self.assertEqual(averager._adapting, False)


if __name__ == '__main__':
    unittest.main()
