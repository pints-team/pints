#!/usr/bin/env python3
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import unittest

import pints
from shared import StreamCapture


class TestDualAveraging(unittest.TestCase):
    """
    Tests the dual averaging adaption used in the NUTS sampler.
    """

    def test_accept_prob_of_greater_then_one(self):
        num_warmup_steps = 200
        target_accept_prob = 1.0
        init_epsilon = 1.0
        init_inv_mass_matrix = np.array([[1, 0], [0, 1]])

        averager = pints.DualAveragingAdaption(num_warmup_steps,
                                               target_accept_prob,
                                               init_epsilon,
                                               init_inv_mass_matrix)

        # should adjust the input 2.0 to 1.0, which is the target accept
        # probability
        averager.adapt_epsilon(2.0)
        self.assertEqual(averager._h_bar, 0.0)

    def test_use_dense_mass_matrix(self):
        num_warmup_steps = 200
        target_accept_prob = 1.0
        init_epsilon = 1.0
        init_inv_mass_matrix = np.array([[1, 0], [0, 1]])

        averager = pints.DualAveragingAdaption(num_warmup_steps,
                                               target_accept_prob,
                                               init_epsilon,
                                               init_inv_mass_matrix)

        self.assertTrue(averager.use_dense_mass_matrix())

        init_inv_mass_matrix = np.array([1, 1])
        averager = pints.DualAveragingAdaption(num_warmup_steps,
                                               target_accept_prob,
                                               init_epsilon,
                                               init_inv_mass_matrix)

        self.assertFalse(averager.use_dense_mass_matrix())

    def test_set_inv_mass(self):
        num_warmup_steps = 200
        target_accept_prob = 1.0
        init_epsilon = 1.0
        init_inv_mass_matrix = np.array([[1, 0], [0, 0]])

        with StreamCapture() as c:
            with self.assertRaises(AttributeError):
                pints.DualAveragingAdaption(
                    num_warmup_steps,
                    target_accept_prob,
                    init_epsilon,
                    init_inv_mass_matrix
                )
            self.assertIn("WARNING", c.text())

    def test_dual_averaging(self):

        num_warmup_steps = 200
        target_accept_prob = 0.5
        init_epsilon = 1.0
        init_inv_mass_matrix = np.array([[1, 0], [0, 1]])
        target_mass_matrix = np.array([[10, 0], [0, 10]])

        # raises an exception if the requested number of warm-up steps is
        # too low
        with self.assertRaises(ValueError):
            averager = pints.DualAveragingAdaption(10,
                                                   target_accept_prob,
                                                   init_epsilon,
                                                   init_inv_mass_matrix)

        averager = pints.DualAveragingAdaption(num_warmup_steps,
                                               target_accept_prob,
                                               init_epsilon,
                                               init_inv_mass_matrix)

        # test initialisation
        self.assertEqual(averager._epsilon, init_epsilon)
        np.testing.assert_array_equal(
            averager.get_inv_mass_matrix(), init_inv_mass_matrix)
        self.assertEqual(averager._counter, 0)

        # these are the default window sizes for the algorithm
        initial_window = 75
        base_window = 25
        terminal_window = 50

        self.assertEqual(averager._next_window, initial_window + base_window)
        self.assertEqual(averager._adapting, True)

        # dummy function to generate acceptance probabilities
        # dual averaging will attempt to set epsilon so this function
        # returns `target_accept_prob`
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

        # before the end of the window the mass matrix should not have been
        # updated
        np.testing.assert_array_equal(
            averager.get_inv_mass_matrix(), init_inv_mass_matrix)
        x = np.random.multivariate_normal(np.zeros(2) + 123,
                                          target_mass_matrix)

        np.testing.assert_array_equal(averager._samples[:, :-1],
                                      stored_x[:, :-1])
        restart = averager.step(x, fake_accept_prob(averager._epsilon))

        # end of window triggers a restart
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

        # test that we have adapted epsilon correctly
        self.assertAlmostEqual(fake_accept_prob(averager._epsilon),
                               target_accept_prob, 1)

        # test the counters
        self.assertEqual(averager._counter, initial_window + base_window)
        self.assertEqual(averager._next_window,
                         num_warmup_steps - terminal_window)

        # test counters for two more windows
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

        # check that subsequent steps do nothing
        old_counter = averager._counter
        averager.step(x, fake_accept_prob(averager._epsilon))
        self.assertEqual(old_counter, averager._counter)


if __name__ == '__main__':
    unittest.main()
