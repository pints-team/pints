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
cprior = 1.0 / np.product(boundaries.range())
def prior(x):
    return boundaries.check(x) * cprior

# Create a log-likelihood function
def log_likelihood(x):
    error = values - model.simulate(x[:-1], times)
    return -len(values) * np.log(x[-1]) - np.sum(error**2) / (2 * x[-1]**2)
    
if False:
    pl.figure()
    ip = 0
    x = np.linspace(boundaries.lower()[ip], boundaries.upper()[ip], 100)
    y = []
    for xx in x:
        p = np.copy(real_parameters)
        p[ip] = xx
        #y.append(np.exp(log_likelihood(p)))
        y.append(log_likelihood(p))

    pl.plot(x, y)
    pl.show()


# Run a simple adaptive mcmc 

# Initial guess

theta = real_parameters * 1.0
sigma = np.diag(boundaries.range() / 100)

sigma = np.diag(np.abs(theta * 0.001))

loga = 0
t = 0

last = theta
ll_last = log_likelihood(last)

# Empty chain
chain = []

# Initial acceptance rate (value doesn't matter)
acceptance = 1

# Acceptance target
acceptance_target = 0.25

mu = np.copy(theta)

n1 = 1000 * (1 + model.dimension())
n2 = 1000git st0 * (1 + model.dimension())

for t in xrange(n1 + n2):
    # Guess new point
    next = np.random.multivariate_normal(theta, np.exp(loga) * sigma)
    # Optimisation: Don't evaluate when prior is zero
    accepted = 0.0
    if prior(next) > 0:
        # Optimisation: Don't evaluate when proposal is last point
        # (this happens a lot!)
        if np.all(next == last):
            accepted = 1.0
        else:
            # Accept based on likelihood estimate
            ll_next = log_likelihood(next)
            u = np.log(np.random.uniform())
            if u < ll_next - ll_last:
                accepted = 1.0
                # Move to new proposal
                last = next
                ll_last = ll_next
    else:
        next = last
    # Adapt covariance matrix
    if t > n1:
        gamma = (t - n1 + 1) ** -0.6

        dsigm = np.reshape(next - mu, (len(next), 1))
        sigma = (1 - gamma) * sigma + gamma * np.dot(dsigm, dsigm.T)
        
        mu = (1 - gamma) * mu + gamma * next
        
        loga += gamma * (accepted - acceptance_target)
                
    # Update acceptance rate
    acceptance = (t * acceptance + accepted) / (t + 1)
    # Add point to chain (#TODO thinning)
    chain.append(next)

    if accepted:
        print('Acceptance: ' + str(acceptance))

burn = n1

chain = np.array(chain[burn:])


for i, real in enumerate(real_parameters):
    pl.figure()
    pl.hist(chain[:,i], label='p' + str(i + 1), bins=60)
pl.show()

'''
# Select a score function
score = pints.SumOfSquaresError(problem)

# Perform an optimization with boundaries and hints
x0 = 0.015, 500
sigma0 = [0.0001, 0.01]
found_parameters, found_solution = pints.cmaes(
    score,
    boundaries,
    x0,
    sigma0,
    )

print('Score at true solution: ')
print(score(real_parameters))

print('Found solution:          True parameters:' )
for k, x in enumerate(found_parameters):
    print(pints.strfloat(x) + '    ' + pints.strfloat(real_parameters[k]))
'''
