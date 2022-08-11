#!/usr/bin/env python3
#
# Tests the Boundaries classes.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import numpy as np


class TestRectangularBoundaries(unittest.TestCase):
    """
    Tests the RectangularBoundaries class.
    """

    def test_creation(self):
        # Tests creation and input checking

        # Create boundaries
        pints.RectangularBoundaries([1, 2], [3, 4])
        pints.RectangularBoundaries([1], [2])
        pints.RectangularBoundaries(np.array([1, 2, 3]), [4, 5, 6])
        pints.RectangularBoundaries(1, 2)

        # Create invalid boundaries
        self.assertRaises(ValueError, pints.RectangularBoundaries, [1, 2], [1])
        self.assertRaises(ValueError, pints.RectangularBoundaries, [], [])
        self.assertRaises(ValueError, pints.RectangularBoundaries, [2], [1])
        self.assertRaises(
            ValueError, pints.RectangularBoundaries, [1, 1], [10, 1])

    def test_boundary_checking(self):

        # Check methods
        lower = [1, -2]
        upper = [3, 4]
        b = pints.RectangularBoundaries(lower, upper)
        self.assertEqual(b.n_parameters(), len(lower))
        self.assertTrue(np.all(b.lower() == np.array(lower)))
        self.assertTrue(np.all(b.upper() == np.array(upper)))
        self.assertTrue(np.all(b.range() == np.array(upper) - np.array(lower)))

        # Check checking
        # Within bounds
        self.assertTrue(b.check([2, 3]))
        # On a lower bound
        self.assertTrue(b.check([1, 3]))
        # Below a lower bound
        self.assertFalse(b.check([1 - 1e16, 4]))
        # On an upper bound
        self.assertFalse(b.check([3, 0]))
        # Above an upper bound
        self.assertFalse(b.check([2, 14]))
        # Wrong in every way
        self.assertFalse(b.check([-20, 20]))
        self.assertFalse(b.check([20, -20]))
        # Negative number
        self.assertFalse(b.check([2, -3]))

    def test_sampling(self):
        # Tests sampling from within rectangular boundaries

        lower = np.array([1, -1])
        upper = np.array([2, 1])
        d = 2
        b = pints.RectangularBoundaries(lower, upper)
        self.assertTrue(b.check(b.sample()))

        n = 1
        x = b.sample()
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = b.sample(n)
        self.assertEqual(x.shape, (n, d))

        for p in b.sample(50):
            self.assertTrue(b.check(p))


class TestLogPDFBoundaries(unittest.TestCase):
    """
    Tests boundaries based on a LogPDF.
    """
    def test_basic(self):

        # Create a custom LogPDF for testing
        class Gradient(pints.LogPDF):

            def n_parameters(self):
                return 1

            def __call__(self, x):
                return x

        # Create boundaries based on gradient
        b = pints.LogPDFBoundaries(Gradient(), 0.75)

        # Test n_parameters
        self.assertEqual(b.n_parameters(), 1)

        # Test
        self.assertFalse(b.check(0))
        self.assertFalse(b.check(-1))
        self.assertTrue(b.check(2))
        self.assertTrue(b.check(1))
        self.assertFalse(b.check(0.75))

        # Test bad creation
        self.assertRaisesRegex(
            ValueError, 'must be a pints.LogPDF', pints.LogPDFBoundaries, 5, 5)

        # Can't sample from this log pdf!
        self.assertRaises(NotImplementedError, b.sample, 1)

        # Can sample if we have a prior that supports it
        b = pints.RectangularBoundaries([1, 1], [2, 2])
        p = pints.UniformLogPrior(b)
        p.sample(2)
        b = pints.LogPDFBoundaries(p)
        b.sample(2)


if __name__ == '__main__':
    unittest.main()
