[![TravisCI](https://travis-ci.org/martinjrobins/pints.svg?branch=master)](https://travis-ci.org/martinjrobins/pints)

# What is Pints?

We are working on problems in the area of inverse modelling and baysian 
inference as applied to problems in electrochemistry and cardiac 
electrophysiology.
Pints (Probabilistic Inference on Noisy Time-Series) 
is a collaborative repository to organise our efforts and share tools.

## How do you use pints

To use a model in Pints, you need to make sure it implements two methods:

```
dimension() --> Returns the dimension of the parameter space.
        
simulate(parameters, times) --> Returns a vector of model evaluations at
                                the given times, using the given parameters
```

If your model implements these methods - or you can write a wrapper
class that does - you can start using Pints for optimisation or MCMC.

## Examples

Examples are given in the `example_x.py` files, in the root of the
project.

## Installing

You'll need the following requirements:

- Python 2.7
- Python libraries: cma numpy matplotlib scipy

Then, install using ```python setup.py install```

Or, if you plan to make changes to pints: `python setup.py develop` (this will
 make Python find the local pints files when you use `import pints`).

### Testing pints

To run all tests, clone the repository, navigate to the pints directory
and type:

```
python -m unittest discover -v test
```

Or use the bash script `run-tests.sh`


