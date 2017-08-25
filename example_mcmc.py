#!/usr/bin/env python2
from __future__ import print_function
import pints
import pints.toy as toy
import numpy as np
import matplotlib.pyplot as pl

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
log_likelihood = pints.GaussianLogLikelihood(problem)

# Create a uniform prior over both the parameters and the new noise variable
prior = pints.UniformPrior(
    [0.01, 400, noise*0.1],
    [0.02, 600, noise*100]
    )

# Create a Bayesian log-likelihood (prior * likelihood)
log_likelihood = pints.BayesianLogLikelihood(prior, log_likelihood)

# Run a simple adaptive mcmc routine
x0 = real_parameters * 1.1
chain = pints.adaptive_covariance_mcmc(log_likelihood, x0)

# Plot input
pl.figure()
pl.plot(times, values)
pl.plot(times, org_values, color='tab:green')

# Plot log-likelihood function (noise causes bias!)
pl.figure()
for i, p in enumerate(real_parameters):
    # Add subplot
    pl.subplot(len(real_parameters), 1, 1 + i)
    pl.xlabel('Parameter ' + str(i + 1))
    # Generate some x-values near the true parameter
    if i + 1 == len(real_parameters):
        # Noise plot: special case
        # First, add a line showing the sample standard deviation of the
        # generated noise
        pl.axvline(noise_sample_std, color='tab:orange', label='Sample std.')
        # Next, choose a wide range of parameters so we can see the peak of the
        # log-likelihood curve
        xmin = min([p*0.95, noise_sample_std*0.95])
        xmax = max([p*1.05, noise_sample_std*1.05])
    else:
        # Choose same limits as histogram (see below)
        mu = np.mean(chain[:,i])
        sigma = np.std(chain[:,i])
        xmin = mu - 3 * sigma
        xmax = mu + 3 * sigma
    x = np.linspace(xmin, xmax, 100)
    # Calculate log-likelihood with other parameters fixed
    y = [log_likelihood(list(real_parameters[:i]) + [j]
        + list(real_parameters[1+i:])) for j in x]
    # Plot
    pl.plot(x, y, color='tab:green', label='Log-likelihood')
    pl.axvline(p, color='tab:blue', label='True value')
    pl.legend()
pl.tight_layout()

# Plot output
pl.figure()
for i, real in enumerate(real_parameters):
    # Add subplot
    pl.subplot(len(real_parameters), 1, 1+i)
    pl.xlabel('Parameter ' + str(i + 1))
    # Show true value
    pl.axvline(real)
    # Show histogram of chain
    pl.hist(chain[:,i], label='p' + str(i + 1), bins=40, color='tab:green',
        alpha=0.5)
    # Center plot around mean of chain
    mu = np.mean(chain[:,i])
    sigma = np.std(chain[:,i])
    pl.xlim(mu - 3 * sigma, mu + 3 * sigma)
pl.axvline(noise_sample_std, color='tab:orange')
pl.tight_layout()
pl.show()

