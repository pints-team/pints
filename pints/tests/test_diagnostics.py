#!/usr/bin/env python3
#
# Tests the basic methods of diagnostics.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import numpy as np
import pints._diagnostics


class TestDiagnostics(unittest.TestCase):
    """
    Tests various diagnostic measures available in Pints
    """
    def test_autocorrelation(self):
        # Tests that autocorrelation measure is correct
        x = np.array([1, 2, 3, 4, -1, -1])
        y = pints._diagnostics.autocorrelation(x)
        y_true = np.array(
            [1., 0.21354167, -0.41666667, -0.296875, -0.03645833, 0.03645833])

        for i in range(0, len(x)):
            self.assertAlmostEqual(y[i], y_true[i])

    def test_autocorrelation_negative(self):
        # Tests autocorrelation_negative yields the correct result
        # under both possibilities

        # Test for case where there is a negative element
        x = np.array([1, 2, 3, 4, -1, -1])
        self.assertEqual(pints._diagnostics._autocorrelate_negative(x), 4)

        # Test for case with no negative elements
        x = np.array([1, 2, 3, 4, 1, 1])
        self.assertEqual(pints._diagnostics._autocorrelate_negative(x), 6)

    def test_effective_sample_size_single_parameter(self):
        # Tests that ESS for a single parameter is correct

        # For case with negative elements in x
        x = np.array([1, 2, 3, 4, -1, -1])
        y = pints._diagnostics.effective_sample_size_single_parameter(x)
        self.assertAlmostEqual(y, 1.75076, 5)

        # Case with positive elements only in x
        x = np.array([1, 2, 3, 4, 1, 1])
        self.assertAlmostEqual(
            pints._diagnostics.effective_sample_size_single_parameter(x),
            1.846154, 6)

    def test_effective_sample_size(self):
        # Tests ess for a matrix of parameters

        # matrix with two columns of samples
        x = np.transpose(np.array([[1.0, 1.1, 1.4, 1.3, 1.3],
                                   [1.0, 2.0, 3.0, 4.0, 5.0]]))
        y = pints._diagnostics.effective_sample_size(x)
        self.assertAlmostEqual(y[0], 1.439232, 6)
        self.assertAlmostEqual(y[1], 1.315789, 6)

        # Bad calls
        self.assertRaisesRegex(
            ValueError, '2d array', pints.effective_sample_size, x[0])
        self.assertRaisesRegex(
            ValueError, 'At least two', pints.effective_sample_size, x[:1])

    def test_within(self):
        # Tests within chain variance calculation

        # matrix with two columns of samples
        x = np.array([[1.0, 1.1, 1.4, 1.3, 1.3],
                      [1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertAlmostEqual(pints._diagnostics._within(x), 1.2635, 5)

    def test_between(self):
        # Tests between chain variance calculation

        # matrix with two columns of samples
        x = np.array([[1.0, 1.1, 1.4, 1.3, 1.3],
                      [1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertAlmostEqual(pints._diagnostics._between(x), 7.921, 4)

    def test_rhat(self):
        # Tests that rhat works

        # Tests Rhat computation for one parameter, chains.shape=(2, 4)
        chains = np.array([[1.0, 1.1, 1.4, 1.3],
                          [1.0, 2.0, 3.0, 4.0]])
        self.assertAlmostEqual(
            pints._diagnostics.rhat(chains), 2.3303847470550716, 6)

        # Test Rhat computation for two parameters, chains.shape=(3, 4, 2)
        chains = np.array([
            [
                [-1.10580535, 2.26589882],
                [0.35604827, 1.03523364],
                [-1.62581126, 0.47308597],
                [1.03999619, 0.58203464]
            ],
            [
                [-1.04755457, -2.28410098],
                [0.17577692, -0.79433186],
                [-0.07979098, -1.87816551],
                [-1.39836319, 0.95119085]
            ],
            [
                [-1.1182588, -0.34647435],
                [1.36928142, -1.4079284],
                [0.92272047, -1.49997615],
                [0.89531238, 0.63207977]
            ]])

        y = pints._diagnostics.rhat(chains)
        d = np.array(y) - np.array([0.84735944450487122, 1.1712652416950846])
        self.assertLess(np.linalg.norm(d), 0.01)

    def test_bad_rhat_inputs(self):
        # Tests whether exceptions are thrown should the input to rhat not be
        # valid

        # Pass chain of dimension 1
        chains = np.empty(shape=1)
        message = (
            'Dimension of chains is 1. '
            + 'Method computes Rhat for one '
            'or multiple parameters and therefore only accepts 2 or 3 '
            'dimensional arrays.')
        self.assertRaisesRegex(
            ValueError, message[0], pints.rhat, chains)

        # Pass chain of dimension 4
        chains = np.empty(shape=(1, 1, 1, 1))
        message = (
            'Dimension of chains is 4. '
            + 'Method computes Rhat for one '
            'or multiple parameters and therefore only accepts 2 or 3 '
            'dimensional arrays.')
        self.assertRaisesRegex(
            ValueError, message[0], pints.rhat, chains)

        # Pass bad warm-up arguments
        chains = np.empty(shape=(2, 4))

        # warm-up greater than 100%
        warm_up = 1.1
        message = (
            '`warm_up` is set to 1.1. `warm_up` only takes values in [0,1].')
        self.assertRaisesRegex(
            ValueError, message[0], pints.rhat, chains, warm_up)

        # Negative warm-up
        warm_up = -0.1
        message = (
            '`warm_up` is set to -0.1. `warm_up` only takes values in [0,1].')
        self.assertRaisesRegex(
            ValueError, message[0], pints.rhat, chains, warm_up)

        # Pass chains with too little samples (n<4)
        chains = np.empty(shape=(1, 4))
        warm_up = 0.9
        message = (
            'Number of samples per chain after warm-up and chain splitting is '
            '1. Method needs at least 2 samples per chain.')
        self.assertRaisesRegex(
            ValueError, message[0], pints.rhat, chains, warm_up)

    def test_rhat_all_params(self):
        # Tests that rhat_all works

        x = np.array([[[-1.10580535, 2.26589882],
                       [0.35604827, 1.03523364],
                       [-1.62581126, 0.47308597],
                       [1.03999619, 0.58203464]],
                      [[-1.04755457, -2.28410098],
                       [0.17577692, -0.79433186],
                       [-0.07979098, -1.87816551],
                       [-1.39836319, 0.95119085]],
                      [[-1.1182588, -0.34647435],
                       [1.36928142, -1.4079284],
                       [0.92272047, -1.49997615],
                       [0.89531238, 0.63207977]]])

        y = pints._diagnostics.rhat_all_params(x)
        d = np.array(y) - np.array([0.84735944450487122, 1.1712652416950846])
        self.assertLess(np.linalg.norm(d), 0.01)


if __name__ == '__main__':
    unittest.main()
