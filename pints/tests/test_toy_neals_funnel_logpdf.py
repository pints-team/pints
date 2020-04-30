#!/usr/bin/env python3
#
# Tests the Neal's funnel log-pdf toy problem.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestNealsFunnelLogPDF(unittest.TestCase):
    """
    Tests the Neal's funnel log pdf toy problem.
    """
    def test_default(self):
        # Tests instantiation and calls.

        # test default instantiation
        f = pints.toy.NealsFunnelLogPDF()
        self.assertEqual(f.n_parameters(), 10)
        x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
        self.assertAlmostEqual(f(x), -24.512039428370223)
        l, dl = f.evaluateS1(x)
        self.assertAlmostEqual(l, -24.512039428370223)
        dnu = np.sum(np.repeat(0.5 * (np.exp(-3) - 1), 9)) - 3.0 / 9.0
        y = np.concatenate((np.repeat(-np.exp(-3), 9), [dnu]))
        self.assertTrue(np.array_equal(y, dl))

        # test sampling and KL divergence
        samples = f.sample(10000)
        self.assertTrue(np.array_equal(samples.shape,
                                       [10000, 10]))
        self.assertTrue(f.kl_divergence(samples) < 0.1)
        self.assertEqual(f.kl_divergence(samples), f.distance(samples))

        # test mean
        self.assertTrue(np.array_equal(np.zeros(10), f.mean()))
        self.assertTrue(np.array_equal(
            np.concatenate((np.repeat(90, f._n_parameters - 1), [9])),
            f.var()))

        # test marginal_log_pdf
        log_prob = f.marginal_log_pdf(0.5, -0.5)
        self.assertAlmostEqual(log_prob, -2.9064684028038599)

    def test_bad_calls(self):
        # Tests bad calls.
        f = pints.toy.NealsFunnelLogPDF()
        self.assertRaises(ValueError, f.__call__, [1, 2])
        self.assertRaises(ValueError, pints.toy.NealsFunnelLogPDF, 1)

        n = 10
        d = 10
        x = np.ones((n, d + 1))
        self.assertRaises(ValueError, f.kl_divergence, x)
        x = np.ones((n, d, 2))
        self.assertRaises(ValueError, f.kl_divergence, x)

    def test_bespoke(self):
        # Tests non-default function behaviour
        f = pints.toy.NealsFunnelLogPDF(20)
        self.assertEqual(f.n_parameters(), 20)
        x = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, 8, 4]
        self.assertAlmostEqual(f(x), -59.117213036088565)
        l, dl = f.evaluateS1(x)
        self.assertAlmostEqual(l, -59.117213036088565)
        self.assertTrue(dl[0], 0.018315638888734179)
        self.assertTrue(dl[18], -0.14652511110987343)
        self.assertTrue(dl[19], -9.1935032500063443)

        # test sampling and KL divergence
        samples = f.sample(10000)
        self.assertTrue(np.array_equal(samples.shape,
                                       [10000, 20]))
        self.assertTrue(f.kl_divergence(samples) < 0.1)
        self.assertEqual(f.kl_divergence(samples), f.distance(samples))

        # test mean
        self.assertTrue(np.array_equal(np.zeros(20), f.mean()))
        self.assertTrue(np.array_equal(
            np.concatenate((np.repeat(90, f._n_parameters - 1), [9])),
            f.var()))

        n = 10
        d = 20
        x = np.ones((n, d + 1))
        self.assertRaises(ValueError, f.kl_divergence, x)
        x = np.ones((n, d, 2))
        self.assertRaises(ValueError, f.kl_divergence, x)

    def test_suggested_bounds(self):
        # Tests suggested_bounds().
        # default
        f = pints.toy.NealsFunnelLogPDF()
        bounds = f.suggested_bounds()
        magnitude = 30
        bounds1 = np.tile([-magnitude, magnitude],
                          (f._n_parameters, 1))
        bounds1 = np.transpose(bounds1).tolist()
        self.assertTrue(np.array_equal(bounds, bounds1))

        # non-default
        f = pints.toy.NealsFunnelLogPDF(20)
        bounds = f.suggested_bounds()
        magnitude = 30
        bounds1 = np.tile([-magnitude, magnitude],
                          (f._n_parameters, 1))
        bounds1 = np.transpose(bounds1).tolist()
        self.assertTrue(np.array_equal(bounds, bounds1))


if __name__ == '__main__':
    unittest.main()
