#!/usr/bin/env python
#
# Tests the basic methods of diagnostics.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import numpy as np
import pints._diagnostics

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


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
        self.assertEqual(pints._diagnostics.autocorrelate_negative(x), 4)

        # Test for case with no negative elements
        x = np.array([1, 2, 3, 4, 1, 1])
        self.assertEqual(pints._diagnostics.autocorrelate_negative(x), 7)

    def test_ess_single_param(self):
        # Tests that ESS for a single parameter is correct

        # For case with negative elements in x
        x = np.array([1, 2, 3, 4, -1, -1])
        y = pints._diagnostics.ess_single_param(x)
        self.assertAlmostEqual(y, 1.75076, 5)

        # Case with positive elements only in x
        x = np.array([1, 2, 3, 4, 1, 1])
        self.assertAlmostEqual(
            pints._diagnostics.ess_single_param(x), 1.846154, 6)

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
        self.assertAlmostEqual(pints._diagnostics.within(x), 1.2635, 5)

    def test_between(self):
        # Tests between chain variance calculation

        # matrix with two columns of samples
        x = np.array([[1.0, 1.1, 1.4, 1.3, 1.3],
                      [1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertAlmostEqual(pints._diagnostics.between(x), 7.921, 4)

    def test_reorder(self):
        # Tests that reorder function reshapes correctly

        test = np.random.normal(loc=0, scale=1, size=(4, 10, 3))
        y = np.array(pints._diagnostics.reorder(0, test))
        self.assertEqual(y.shape, (4, 10))

    def test_reorder_all(self):
        # Tests that reorder_all function reshapes correctly

        test = np.random.normal(loc=0, scale=1, size=(4, 10, 3))
        y = np.array(pints._diagnostics.reorder_all_params(test))
        self.assertEqual(y.shape, (3, 4, 10))

    def test_rhat(self):
        # Tests that rhat works

        x = np.array([[1.0, 1.1, 1.4, 1.3, 1.3],
                      [1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertAlmostEqual(pints._diagnostics.rhat(x), 1.433115, 6)

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
        d = np.array(y) - np.array([1.0246953961614296, 1.3219816558533388])
        self.assertLess(np.linalg.norm(d), 0.01)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
