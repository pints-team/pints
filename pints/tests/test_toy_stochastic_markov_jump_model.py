#!/usr/bin/env python3
#
# Tests if the markov jump model works.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
from pints.toy.stochastic import DegradationModel


class TestMarkovJumpModel(unittest.TestCase):
    """
    Tests if the markov jump model works using
    the degradation model.
    """
    def test_start_with_zero(self):
        # Test the special case where the initial molecule count is zero
        model = DegradationModel(0)
        times = [0, 1, 2, 100, 1000]
        parameters = [0.1]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertTrue(np.all(values == np.zeros(5)))

    def test_start_with_twenty(self):
        # Run small simulation
        model = DegradationModel(20)
        times = [0, 1, 2, 100, 1000, 10000]
        parameters = [0.1]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 20)
        self.assertEqual(values[-1], 0)
        self.assertTrue(np.all(values[1:] <= values[:-1]))

    def test_simulate(self):
        times = np.linspace(0, 100, 101)
        model = DegradationModel(20)
        time, mol_count = model.simulate_raw([0.1], 100)
        values = model.interpolate_mol_counts(time, mol_count, times)
        self.assertTrue(len(time), len(mol_count))
        # Test output of Gillespie algorithm
        expected = np.array([[x] for x in range(20, -1, -1)])
        self.assertTrue(np.all(mol_count == expected))

        # Check simulate function returns expected values
        self.assertTrue(np.all(values[np.where(times < time[1])] == 20))

        # Check interpolation function works as expected
        temp_time = np.array([np.random.uniform(time[0], time[1])])
        self.assertEqual(
            model.interpolate_mol_counts(time, mol_count, temp_time)[0], 20)
        temp_time = np.array([np.random.uniform(time[1], time[2])])
        self.assertEqual(
            model.interpolate_mol_counts(time, mol_count, temp_time)[0], 19)

        # Check interpolation works if 1 time is given
        time, mol, out = [1], [7], [0, 1, 2, 3]
        interpolated = model.interpolate_mol_counts(time, mol, out)
        self.assertEqual(list(interpolated), [7, 7, 7, 7])

        # Check if no times are given
        self.assertRaisesRegex(ValueError, 'At least one time',
                               model.interpolate_mol_counts, [], [], out)

        # Check if times and count don't match
        self.assertRaisesRegex(ValueError, 'must match',
                               model.interpolate_mol_counts, [1], [2, 3], out)

        # Decreasing output times
        self.assertRaisesRegex(
            ValueError, 'must be non-decreasing',
            model.interpolate_mol_counts, [1, 2], [2, 3], [1, 2, 3, 4, 0])

        # Check extrapolation outside of output times
        # Note: step-wise "interpolation", no actual interpolation!
        time = [10, 11]
        mol = [5, 6]
        out = [0, 10, 10.5, 11, 20]
        val = model.interpolate_mol_counts(time, mol, out)
        self.assertEqual(list(val), [5, 5, 5, 6, 6])

        time = [10]
        mol = [5]
        out = [0, 10, 10.5, 11, 20]
        val = model.interpolate_mol_counts(time, mol, out)
        self.assertEqual(list(val), [5, 5, 5, 5, 5])

    def test_errors(self):
        model = DegradationModel(20)
        # times cannot be negative
        times_2 = np.linspace(-10, 10, 21)
        parameters_2 = [0.1]
        self.assertRaisesRegex(ValueError, 'Negative times',
                               model.simulate, parameters_2, times_2)

        # this model should have 1 parameter
        times = np.linspace(0, 100, 101)
        parameters_3 = [0.1, 1]
        self.assertRaisesRegex(ValueError, 'should have 1 parameter',
                               model.simulate, parameters_3, times)

        # Initial value can't be negative
        self.assertRaisesRegex(ValueError, 'Initial molecule count',
                               DegradationModel, -1)


if __name__ == '__main__':
    unittest.main()
