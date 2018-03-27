#!/usr/bin/env python
import numpy as np
import pints

# Define boring 1-output and 2-output models
class NullModel1(pints.ForwardModel):
    def dimension(self):
        return 1

    def simulate(self, x, times):
        return np.zeros(times.shape)

class NullModel2(pints.ForwardModel):
    def dimension(self):
        return 1

    def simulate(self, x, times):
        return np.zeros(times.shape).reshape((len(times), 1))

# Create single output problem
times = np.arange(10)
np.random.seed(1)
sigma = 3
values1 = np.random.uniform(0, sigma, times.shape)
problem1 = pints.SingleSeriesProblem(NullModel1(), times, values1)
log1 = pints.KnownNoiseLogLikelihood(problem1, sigma)


# Create one multi output problem
values2 = values1.reshape((10, 1))
problem2 = pints.MultiSeriesProblem(NullModel2(), times, values2)
log2 = pints.KnownMultivariateNoiseLogLikelihood(problem2, sigma)

t = pints.Timer()

t.reset()
for i in xrange(10):
    log1(0)
print(t.time())

t.reset()
for i in xrange(10):
    log2(0)
print(t.time())

t.reset()
for i in xrange(10000):
    log1(0)
print(t.time())

t.reset()
for i in xrange(10000):
    log2(0)
print(t.time())
