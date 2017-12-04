[![TravisCI](https://travis-ci.org/pints-team/pints.svg?branch=master)](https://travis-ci.org/pints-team/pints)

# What is Pints??

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

Here is an [example](examples/writing-a-model.ipynb) showing how to implement 
your own model with Pints.

### Examples

Pints comes with a number of [detailed examples](examples/EXAMPLES.md).

## How do you install Pints

You'll need the following requirements:

- Python 2.7
- Python libraries: `cma` `numpy` `matplotlib` `scipy`

Then, install using 

```
python setup.py install
```

### For developers:

If you plan to make changes to pints, install with:

```
python setup.py develop
```

(this will make Python find the local pints files when you use `import pints`).

### Testing:

To run quick tests, use `run-tests.sh`.


