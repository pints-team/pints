#!/usr/bin/env python
#
# Simply tests if the pints module can be imported
#
import unittest
import numpy as np
import pints
import pints.toy
class TestLogistic(unittest.TestCase):
    def test_start_with_zero(self):
        model = pints.toy.LogisticModel(0)
        times = [0, 1, 2, 10000]
        parameters = [1, 5]        
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        for v in values:
            self.assertEqual(v, 0)
    def test_start_with_two(self):
        # Run small simulation
        model = pints.toy.LogisticModel(2)
        times = [0, 1, 2, 10000]
        parameters = [1, 5]        
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 2)
        self.assertEqual(values[-1], parameters[-1])
        # Run large simulation
        times = np.arange(0, 1000)
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 2)
        self.assertEqual(values[-1], parameters[-1])
        self.assertTrue(np.all(values[1:] >= values[:-1]))
        
        

if __name__ == '__main__':
    unittest.main()
