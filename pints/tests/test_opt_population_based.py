#!/usr/bin/env python3
#
# Tests the shared methods of the PopulationBasedOptimiser
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest

import numpy as np

import pints
import pints.toy


class TestPopulationBasedOptimiser(unittest.TestCase):
    """
    Tests the  shared methods of the PopulationBasedOptimiser.
    """
    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

    def test_population_size(self):

        r = pints.toy.RosenbrockError()
        x = np.array([1.01, 1.01])
        opt = pints.OptimisationController(r, x, method=pints.XNES)
        m = opt.optimiser()
        n = m.population_size()
        m.set_population_size(n + 1)
        self.assertEqual(m.population_size(), n + 1)

        # Test invalid size
        self.assertRaisesRegex(
            ValueError, 'at least 1', m.set_population_size, 0)

        # test hyper parameter interface
        self.assertEqual(m.n_hyper_parameters(), 1)
        m.set_hyper_parameters([n + 2])
        self.assertEqual(m.population_size(), n + 2)
        self.assertRaisesRegex(
            ValueError, 'at least 1', m.set_hyper_parameters, [0])

        # Test changing during run
        m.ask()
        self.assertRaises(Exception, m.set_population_size, 2)


if __name__ == '__main__':
    unittest.main()

