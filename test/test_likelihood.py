#!/usr/bin/env python
#
# Tests the log likelihood methods
#
import unittest
class TestLogLikelihood(unittest.TestCase):
    def test_scaled_log_likelihood(self):
        import pints
        import pints.toy as toy
        import numpy as np

        model = toy.LogisticModel()
        real_parameters = [0.015, 500]
        test_parameters = [0.014, 501]
        sigma = 0.001
        times = np.linspace(0, 1000, 1000)
        values = model.simulate(real_parameters, times)

        # Create an object with links to the model and time series
        problem = pints.SingleSeriesProblem(model, times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.KnownNoiseLogLikelihood(problem,sigma)
        log_likelihood_scaled = pints.ScaledLogLikelihood(log_likelihood_not_scaled)

        eval_not_scaled = log_likelihood_not_scaled(test_parameters)
        eval_scaled = log_likelihood_scaled(test_parameters)

        self.assertEqual(int(eval_not_scaled),-211495080219)
        self.assertAlmostEqual(eval_scaled*len(times),eval_not_scaled)
    #TODO Add tests for the remaining likelihood classes and methods
if __name__ == '__main__':
    unittest.main()
