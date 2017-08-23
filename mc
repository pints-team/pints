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

if False:
    pl.figure()
    pl.plot(times, values)
    pl.plot(times, org_values)
    pl.show()

# Create an object with links to the model and time series
problem = pints.SingleSeriesProblem(model, times, values)

# Create a log-likelihood function (adds an extra parameter!)
log_likelihood = pints.GaussianLogLikelihood(problem)

# Select some boundaries
boundaries = pints.Boundaries([
    # Lower
    0.014,
    490,
    noise*0.1,
    ], [
    # Upper
    0.016,
    510,
    noise*100,
    ])

# Create a prior
prior = pints.UniformPrior(boundaries)

# Run a simple adaptive mcmc routine

# Initial guess
mu = real_parameters * 1.0  #TODO
sigma = np.diag(mu * 0.01)

# Target acceptance rate
acceptance_target = 0.25

# Total number of iterations
iterations = 2000 * prior.dimension()

# Number of iterations to use adaptation in
adaptation = int(iterations / 2)

# Number of iterations to discard as burn-in
burn_in = int(iterations / 2)

# Thinning: Store only one sample per X
thinning = 1

# First point
current = mu
current_likelihood = log_likelihood(current)

# Chain of stored samples
chain = []

# Initial acceptance rate (value doesn't matter)
loga = 0
acceptance = 0

# Go!
for i in xrange(iterations):
    # Propose new point
    proposed = np.random.multivariate_normal(current, np.exp(loga) * sigma)
    
    # Check if the point can be accepted
    accepted = 0.0
    if prior(proposed) > 0:
        # Accept based on likelihood estimate
        proposed_likelihood = log_likelihood(proposed)
        u = np.log(np.random.rand())
        if u < proposed_likelihood - current_likelihood:
            accepted = 1.0
            current = proposed
            current_likelihood = proposed_likelihood
    
    # Adapt covariance matrix
    if i > adaptation:
        gamma = 1 / (i - adaptation + 1) ** 0.6
        dsigm = np.reshape(current - mu, (len(current), 1))
        sigma = (1 - gamma) * sigma + gamma * np.dot(dsigm, dsigm.T)
        mu = (1 - gamma) * mu + gamma * current
        loga += gamma * (accepted - acceptance_target)

    # Update acceptance rate
    acceptance = (i * acceptance + accepted) / (i + 1)
    
    # Add point to chain
    if i > burn_in and i % thinning == 0:
        chain.append(current)

chain = np.array(chain)

pl.figure()
for i, real in enumerate(real_parameters):
    pl.subplot(len(real_parameters), 1, 1+i)
    pl.hist(chain[:,i], label='p' + str(i + 1), bins=40)
pl.show()


