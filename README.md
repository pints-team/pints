[![TravisCI](https://travis-ci.org/pints-team/pints.svg?branch=master)](https://travis-ci.org/pints-team/pints)

# What is Pints?

We are working on problems in the area of inverse modelling and baysian 
inference as applied to problems in electrochemistry and cardiac 
electrophysiology.
Pints (Probabilistic Inference on Noisy Time-Series) 
is a collaborative repository to organise our efforts and share tools.

## How do I use pints?

To use a model in Pints, you need to make sure it extends the [ForwardModel](http://pints.readthedocs.io/en/latest/core_classes_and_methods.html#forward-model) interface, which has just two methods:

```
dimension() --> Returns the dimension of the parameter space.
        
simulate(parameters, times) --> Returns a vector of model evaluations at
                                the given times, using the given parameters
```

If your model implements these methods - [or you can write a wrapper class that does](examples/writing-a-model.ipynb) - you can start using Pints for [optimisation](examples/optimisation-first-example.ipynb) or [MCMC](examples/inference-first-example.ipynb).

### Examples

Pints comes with a number of [detailed examples](examples/EXAMPLES.md).

## How can I install Pints?

You'll need the following requirements:

- Python 2.7
- Python libraries: `cma` `numpy` `matplotlib` `scipy`

Then, install using 

```
python setup.py install
```

## How can I contribute to Pints?

If you'd like to contribute to Pints, please create an [issue](https://guides.github.com/features/issues/) where new proposals can be discusssed. Then, make any changes on a [fork](https://help.github.com/articles/fork-a-repo/) of this repo on github, and send us your modifications via a [pull request](https://help.github.com/articles/about-pull-requests/).

### Installation

If you plan to make changes to Pints, install with:

```
python setup.py develop
```

(this will make Python find the local pints files when you use `import pints`).

### Testing:

To run quick tests, use `run-tests.sh`.


