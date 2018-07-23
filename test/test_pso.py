#!/usr/bin/env python
#
# Tests the basic methods of the PSO optimiser.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import re
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture, TemporaryDirectory

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

debug = False
method = pints.PSO


class TestPSO(unittest.TestCase):
    """
    Tests the basic methods of the PSO optimiser.
    """

    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

    def test_unbounded(self):
        """ Runs an optimisation without boundaries. """
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        opt = pints.Optimisation(r, x, method=method)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_bounded(self):
        """ Runs an optimisation with boundaries. """
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.Boundaries([-0.01, 0.95], [0.01, 1.05])
        opt = pints.Optimisation(r, x, boundaries=b, method=method)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_bounded_and_sigma(self):
        """ Runs an optimisation without boundaries and sigma. """
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.Boundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.Optimisation(r, x, s, b, method)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_ask_tell(self):
        """ Tests ask-and-tell related error handling. """
        opt = method(np.array([1.1, 1.1]))

        # Stop called when not running
        self.assertFalse(opt.stop())

        # Tell before ask
        self.assertRaisesRegex(
            Exception, 'ask\(\) not called before tell\(\)', opt.tell, 5)

    def test_logging(self):
        """ Tests logging for PSO and other optimisers. """
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.Boundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.Optimisation(r, x, s, b, method)

        # No logging
        opt = pints.Optimisation(r, x, s, b, method)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(False)
        opt.set_log_to_file(False)
        with StreamCapture() as c:
            opt.run()
        self.assertEqual(c.text(), '')

        # Log to screen
        opt = pints.Optimisation(r, x, s, b, method)
        opt.set_parallel(2)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(True)
        opt.set_log_to_file(False)
        with StreamCapture() as c:
            opt.run()
        lines = c.text().splitlines()

        self.assertEqual(len(lines), 11)
        self.assertEqual(lines[0], 'Maximising LogPDF')
        self.assertEqual(lines[1], 'using Particle Swarm Optimisation (PSO)')
        self.assertEqual(
            lines[2], 'Running in parallel with 2 worker processes.')
        self.assertEqual(lines[3], 'Population size: 6')
        self.assertEqual(lines[4], 'Iter. Eval. Best      Time m:s')

        pint = '[0-9]+[ ]+'
        pflt = '[0-9.-]+[ ]+'
        ptim = '[0-9]{1}:[0-9]{2}.[0-9]{1}'
        pattern = re.compile(pint * 2 + pflt + ptim)
        for line in lines[5:-1]:
            self.assertTrue(pattern.match(line))
        self.assertEqual(
            lines[-1], 'Halting: Maximum number of iterations (10) reached.')

        # Log to file
        opt = pints.Optimisation(r, x, s, b, method=method)
        opt.set_max_iterations(10)
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                opt.set_log_to_screen(False)
                opt.set_log_to_file(filename)
                opt.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
            self.assertEqual(c.text(), '')

        self.assertEqual(len(lines), 6)
        self.assertEqual(
            lines[0],
            'Iter. Eval. Best      f0        f1        f2        f3       '
            ' f4        f5        Time m:s'
        )

        pattern = re.compile(pint * 2 + pflt * 7 + ptim)
        for line in lines[1:]:
            self.assertTrue(pattern.match(line))

    def test_suggest_population_size(self):
        """
        Tests the suggested_population_size() method for population based
        optimisers.
        """
        r = pints.toy.RosenbrockError(1, 100)
        x0 = np.array([1.1, 1.1])
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])
        opt = pints.Optimisation(r, x0, boundaries=b, method=method)
        opt = opt.optimiser()

        # Test basic usage
        self.assertEqual(type(opt.suggested_population_size()), int)
        self.assertTrue(opt.suggested_population_size() > 0)

        # Test rounding
        n = opt.suggested_population_size() + 1
        self.assertEquals(opt.suggested_population_size(n), n)
        self.assertEquals(opt.suggested_population_size(2) % 2, 0)
        self.assertEquals(opt.suggested_population_size(3) % 3, 0)
        self.assertEquals(opt.suggested_population_size(5) % 5, 0)
        self.assertEquals(opt.suggested_population_size(7) % 7, 0)
        self.assertEquals(opt.suggested_population_size(11) % 11, 0)

    def test_creation(self):
        """ Test optimiser creation. """
        # Test basic creation
        x0 = [1, 2, 3]
        pints.PSO(x0)
        self.assertRaisesRegex(
            ValueError, 'greater than zero', pints.PSO, [])

        # Test with boundaries
        x0 = [1, 2]
        b = pints.Boundaries([0, 0], [3, 3])
        pints.PSO(x0, boundaries=b)
        self.assertRaisesRegex(
            ValueError, 'within given boundaries', pints.PSO, [4, 4],
            boundaries=b)
        self.assertRaisesRegex(
            ValueError, 'same dimension', pints.PSO, [4, 4, 4],
            boundaries=b)

        # Test with scalar sigma
        pints.PSO(x0, 3)
        self.assertRaisesRegex(
            ValueError, 'greater than zero', pints.PSO, x0, -1)

        # Test with vector sigma
        pints.PSO(x0, [3, 3])
        self.assertRaisesRegex(
            ValueError, 'greater than zero', pints.PSO, x0, [3, -1])
        self.assertRaisesRegex(
            ValueError, 'have dimension 2', pints.PSO, x0, [3, 3, 3])

    def test_optimisation_creation(self):
        """
        Test optimisation creation.
        """
        # Test invalid dimensions
        r = pints.toy.RosenbrockError(1, 100)
        self.assertRaisesRegex(
            ValueError, 'same dimension', pints.Optimisation, r, [1])

        # Test invalid method
        self.assertRaisesRegex(
            ValueError, 'subclass', pints.Optimisation, r, [1, 1],
            method=pints.AdaptiveCovarianceMCMC)

    def test_set_hyper_parameters(self):
        """
        Tests the hyper-parameter interface for this optimiser.
        """
        r = pints.toy.RosenbrockError(1, 100)
        x0 = np.array([1.1, 1.1])
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])
        opt = pints.Optimisation(r, x0, boundaries=b, method=method)
        m = opt.optimiser()
        self.assertEqual(m.n_hyper_parameters(), 2)
        n = m.population_size()

        m.set_hyper_parameters([n + 1, 0.5])
        self.assertEqual(m.population_size(), n + 1)

        # Test invalid size
        self.assertRaisesRegex(
            ValueError, 'at least 1', m.set_hyper_parameters, [0, 0.5])
        self.assertRaisesRegex(
            ValueError, 'in the range 0-1', m.set_hyper_parameters, [n, 1.5])

    def test_name(self):
        """ Test the name() method. """
        opt = pints.PSO(np.array([0, 1.01]))
        self.assertIn('PSO', opt.name())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
