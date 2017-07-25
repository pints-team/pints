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
values = model.simulate(real_parameters, times)

# Create an object with links to the model and time series
problem = pints.SingleSeriesProblem(model, times, values)

# Select a score function
score = pints.SumOfSquaresError(problem)

# Select some boundaries
boundaries = pints.Boundaries([0, 400], [0.03, 600])

print('Score at true solution: ')
print(score(real_parameters))

if True:
    # Perform an optimization to find the true parameters
    found_parameters, found_solution = pints.xnes(score)

    print('Found solution:          True parameters:' )
    for k, x in enumerate(found_parameters):
        print(pints.strfloat(x) + '    ' + pints.strfloat(real_parameters[k]))
    print(np.linalg.norm(found_parameters - real_parameters))

if False:
    # Perform an optimization with boundaries
    found_parameters, found_solution = pints.xnes(score, boundaries)

    print('Found solution:          True parameters:' )
    for k, x in enumerate(found_parameters):
        print(pints.strfloat(x) + '    ' + pints.strfloat(real_parameters[k]))

if False:
    # Perform an optimization with a hint
    hint = 0.0145, 499
    found_parameters, found_solution = pints.xnes(score, hint=hint)

    print('Found solution:          True parameters:' )
    for k, x in enumerate(found_parameters):
        print(pints.strfloat(x) + '    ' + pints.strfloat(real_parameters[k]))

if False:
    # Perform an optimization with boundaries and a hint
    hint = 0.015, 500
    found_parameters, found_solution = pints.xnes(score, boundaries, hint)

    print('Found solution:          True parameters:' )
    for k, x in enumerate(found_parameters):
        print(pints.strfloat(x) + '    ' + pints.strfloat(real_parameters[k]))

