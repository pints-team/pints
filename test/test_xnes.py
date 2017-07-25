#!/usr/bin/env python
#
# Tests the basic methods of the xNES optimiser.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import pints.toy as toy
import numpy as np
import matplotlib.pyplot as pl

class TestXNES(unittest.TestCase):
    """
    Tests the basic methods of the xNES optimiser.
    """
    def __init__(self, name):
        super(TestXNES, self).__init__(name)
    
        # Create toy model
        self.model = toy.LogisticModel()
        self.real_parameters = [0.015, 500]
        self.times = np.linspace(0, 1000, 1000)
        self.values = self.model.simulate(self.real_parameters, self.times)

        # Create an object with links to the model and time series
        self.problem = pints.SingleSeriesProblem(
            self.model, self.times, self.values)

        # Select a score function
        self.score = pints.SumOfSquaresError(self.problem)

        # Select some boundaries
        self.boundaries = pints.Boundaries([0, 400], [0.03, 600])
        
        # Set a hint
        self.hint = 0.014, 499
        
        # Minimum score function value to obtain
        self.cutoff = 1e-9
        
        # Maximum tries before it counts as failed
        self.max_tries = 3

    def test_unbounded_no_hint(self):
        
        xnes = pints.XNES(self.score)
        xnes.set_verbose(False)
        for i in xrange(self.max_tries):        
            found_parameters, found_solution = xnes.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)
        
    def test_bounded_no_hint(self):
    
        xnes = pints.XNES(self.score, self.boundaries)
        xnes.set_verbose(False)
        for i in xrange(self.max_tries):        
            found_parameters, found_solution = xnes.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)
        
    def test_unbounded_with_hint(self):
    
        xnes = pints.XNES(self.score, hint=self.hint)
        xnes.set_verbose(False)
        for i in xrange(self.max_tries):        
            found_parameters, found_solution = xnes.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)
        
    def test_bounded_with_hint(self):
    
        xnes = pints.XNES(self.score, self.boundaries, self.hint)
        xnes.set_verbose(False)
        for i in xrange(self.max_tries):        
            found_parameters, found_solution = xnes.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)

if __name__ == '__main__':
    unittest.main()
