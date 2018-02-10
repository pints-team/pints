#!/usr/bin/env python
from __future__ import print_function
import pints
import pints.toy as toy
import pints.plot
import numpy as np
import matplotlib.pyplot as plt

# Load a forward model
model = toy.LogisticModel()

# Create some toy data
real_parameters = [0.015, 500]
times = np.linspace(0, 1000, 1000)
org_values = model.simulate(real_parameters, times)

# Add noise
noise = 10
values = org_values + np.random.normal(0, noise, org_values.shape)
real_parameters = np.array(real_parameters + [noise])

# Get properties of the noise sample
noise_sample_mean = np.mean(values - org_values)
noise_sample_std = np.std(values - org_values)

# Create an object with links to the model and time series
problem = pints.SingleSeriesProblem(model, times, values)

# Create a log-likelihood function (adds an extra parameter!)
log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

# Create a uniform prior over both the parameters and the new noise variable
log_prior = pints.UniformLogPrior(
    [0.01, 400, noise * 0.1],
    [0.02, 600, noise * 100]
)

# Create a posterior log-likelihood (log(prior * likelihood))
log_posterior = pints.LogPosterior(log_prior, log_likelihood)

# Create an adaptive covariance MCMC routine
x0 = real_parameters * 1.1
x1 = real_parameters * 1.2
x2 = real_parameters * 0.9
x3 = real_parameters * 0.8

e = pints.SequentialEvaluator(log_posterior)

mcmc = pints.DifferentialEvolutionMCMC(4, [x0, x1, x2, x3])

chains = []
for i in range(10):
    xs = mcmc.ask()
    fxs = e.evaluate(xs)
    samples = mcmc.tell(fxs)
    if i > 0:
        chains.append(samples)
    print(i)
chains = np.array(chains)

i = 0

a = np.array(chains[:,i,:], copy=True)
print(a.shape)

#s = chains.shape
#chains = chains.reshape(s[1], s[0], s[2])
chains = np.swapaxes(chains, 0, 1)

b = chains[0,:,:]
print(b.shape)

print(a == b)


#pints.plot.trace(chain)
plt.show()

