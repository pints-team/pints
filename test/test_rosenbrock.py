#!/usr/bin/env python3
#
# Tests the Rosenbrock toy problems.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestRosenbrock(unittest.TestCase):
    """
    Tests the Rosenbrock toy problems.
    """
    def test_error(self):
        f = pints.toy.RosenbrockError()
        self.assertEqual(f.dimension(), 2)
        fx = f([10, 10])
        self.assertTrue(fx > 0)

        xopt = f.optimum()
        fopt = f(xopt)
        self.assertEqual(fopt, 0)

        np.random.seed(1)
        for x in np.random.uniform(-5, 5, size=(10, 2)):
            self.assertTrue(f(x) > fopt)


        f = pints.toy.RosenbrockError(10, 10)
        self.assertEqual(f.dimension(), 2)
        fx = f([11, 11])
        self.assertTrue(fx > 0)

        xopt = f.optimum()
        fopt = f(xopt)
        self.assertEqual(fopt, 0)

        np.random.seed(1)
        for x in np.random.uniform(0, 20, size=(10, 2)):
            self.assertTrue(f(x) > fopt)

    def test_log_pdf(self):
        f = pints.toy.RosenbrockLogPDF()
        self.assertEqual(f.dimension(), 2)
        fx = f([10, 10])
        self.assertTrue(fx < 0)

        xopt = f.optimum()
        fopt = f(xopt)
        self.assertEqual(fopt, float('inf'))

        np.random.seed(1)
        for x in np.random.uniform(-5, 5, size=(10, 2)):
            self.assertTrue(f(x) < fopt)

        f = pints.toy.RosenbrockLogPDF(10, 10)
        self.assertEqual(f.dimension(), 2)
        fx = f([11, 11])
        self.assertTrue(fx < 0)

        xopt = f.optimum()
        fopt = f(xopt)
        self.assertEqual(fopt, float('inf'))

        np.random.seed(1)
        for x in np.random.uniform(0, 20, size=(10, 2)):
            self.assertTrue(f(x) < fopt)






if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
