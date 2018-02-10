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

mcmc = pints.AdaptiveCovarianceMCMC(x0)
mcmc.set_target_acceptance_rate(0.3)
mcmc.set_adaptation(False)

rate = []
chain = []
for i in range(10000):
    x = mcmc.ask()
    fx = log_posterior(x)
    sample = mcmc.tell(fx)
    if i == 1000:
        mcmc.set_adaptation(True)
    if i >= 2000:
        chain.append(sample)
    rate.append(mcmc.acceptance_rate())

    print(i, rate[-1])
chain = np.array(chain)
rate = np.array(rate)

print(chain.shape)

pints.plot.trace(chain)

plt.figure()
plt.plot(rate)
plt.show()

