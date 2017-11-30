#
# Tests the electrochemistry models (have to be compiled first!)
#
import pints
import sys

# Steps to run this example:
#   1. make a folder to build the C++ source code
#         $ mkdir build
#         $ cd build
#   2. compile the source code using optimisations
#         $ cmake -DCMAKE_BUILD_TYPE=Release ..
#         $ make
#   3. go back to the `electrochemistry` folder
#         $ cd ..
#   4. put your data files in the current folder
#         $ cp /path/to/data/files/GC01_FeIII-1mM_1M-KCl_02_009Hz.txt .
#   5. run the example
#         $ python example.py

# put path to compiled files here
sys.path.insert(0, 'build')

import electrochemistry
import numpy as np
import matplotlib.pyplot as plt

DEFAULT = {
    'reversed': True,
    'Estart': 0.5,
    'Ereverse': -0.1,
    'omega': 9.0152,
    'phase': 0,
    'dE': 0.08,
    'v': -0.08941,
    't_0': 0.001,
    'T': 297.0,
    'a': 0.07,
    'c_inf': 1*1e-3*1e-3,
    'D': 7.2e-6,
    'Ru': 8.0,
    'Cdl': 20.0*1e-6,
    'E0': 0.214,
    'k0': 0.0101,
    'alpha': 0.53,
    }

# Create the model
ecmodel = electrochemistry.ECModel(DEFAULT)

# Read in data file
filename = 'GC01_FeIII-1mM_1M-KCl_02_009Hz.txt'
data = electrochemistry.ECTimeData(filename,ecmodel,ignore_begin_samples=5,ignore_end_samples=0)

# Wrap electrochemistry model for use in Pints, expose three of the
# parameters for fitting
parameters = ['E0', 'k0', 'Cdl']
pints_model = electrochemistry.PintsModelAdaptor(ecmodel,parameters)

# Get non dimensional parameters from ecmodel
default_parameters = [ecmodel.params[i] for i in parameters]

# simulate model at time points in data file
values = pints_model.simulate(default_parameters,data.time)

# Create an object with links to the model and time series
problem = pints.SingleSeriesProblem(pints_model, data.time, data.current)

# Select a score function
score = pints.SumOfSquaresError(problem)
score_eval = score(default_parameters)
print 'sum of squared error is ',score_eval

plt.plot(data.time,data.current,label='experiment')
plt.plot(data.time,values,label='experiment')
plt.show()

