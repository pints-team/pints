#!/usr/bin/env python2
from __future__ import print_function
import pints
import pints.toy as toy
import numpy as np
import matplotlib.pyplot as pl

np.random.seed(1)

# Load a forward model
model = toy.LogisticModel()

# Create some toy data
real_parameters = [0.015, 500]
times = np.linspace(0, 1000, 1000)
org_values = model.simulate(real_parameters, times)

# Add noise
noise = 50
values = org_values + np.random.normal(0, noise, org_values.shape)
real_parameters = np.array(real_parameters + [noise])

# Create an object with links to the model and time series
problem = pints.SingleSeriesProblem(model, times, values)

# Create a log-likelihood function (adds an extra parameter!)
log_likelihood = pints.GaussianLogLikelihood(problem)

# Create a uniform prior over both the parameters and the new noise variable
prior = pints.UniformPrior(
    [0.01, 400, noise*0.1],
    [0.02, 600, noise*100]
    )

# Create a Bayesian log-likelihood (prior * likelihood)
log_likelihood = pints.BayesianLogLikelihood(prior, log_likelihood)

# Run a simple adaptive mcmc routine
x0 = np.array([0.014, 450, 70])
sigma0 = x0 * 1e-2
chain = pints.adaptive_covariance_mcmc(log_likelihood, x0, sigma0)

# Plot input
pl.figure()
pl.plot(times, values)
pl.plot(times, org_values, color='tab:green')

# Plot output
pl.figure()
for i, real in enumerate(real_parameters):
    pl.subplot(len(real_parameters), 1, 1+i)
    pl.axvline(real)
    pl.hist(chain[:,i], label='p' + str(i + 1), bins=40, color='tab:green',
    alpha=0.5)
pl.show()


