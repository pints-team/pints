[![TravisCI](https://travis-ci.org/pints-team/pints.svg?branch=master)](https://travis-ci.org/pints-team/pints)
[![Documentation Status](https://readthedocs.org/projects/pints/badge/?version=latest)](http://pints.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/pints-team/pints/master?filepath=examples)
 [![codecov](https://codecov.io/gh/pints-team/pints/branch/master/graph/badge.svg)](https://codecov.io/gh/pints-team/pints)

# What is Pints?

Pints (Probabilistic Inference on Noisy Time-Series) is a framework for optimisation and bayesian inference problems on noisy time-series, such as arise in electrochemistry and cardiac electrophysiology.


## How do I use pints?

To use a model in Pints, you need to make sure it extends the [ForwardModel](http://pints.readthedocs.io/en/latest/core_classes_and_methods.html#forward-model) interface, which has just two methods:

```
dimension() --> Returns the dimension of the parameter space.
        
simulate(parameters, times) --> Returns a vector of model evaluations at
                                the given times, using the given parameters
```

If your model implements these methods - [or you can write a wrapper class that does](examples/writing-a-model.ipynb) - you can start using Pints for [optimisation](examples/optimisation-first-example.ipynb) or [MCMC](examples/inference-first-example.ipynb).

### Examples and documentation

Pints comes with a number of [detailed examples](examples/EXAMPLES.md), hosted here on github. In addition, there is a [full API documentation](http://pints.readthedocs.io/en/latest/), hosted on readthedocs.io.

## How can I install Pints?

You'll need the following requirements:

- Python 2.7 or Python 3.4+
- Python libraries: `cma` `numpy` `matplotlib` `scipy`

These will be installed automatically if you go to the directory you downloaded pints to, and run

```
$ pip install .
```

To uninstall again, type

```
$ pip uninstall pints
```

## How can I contribute to Pints?

If you'd like to contribute to Pints, please create an [issue](https://guides.github.com/features/issues/) where new proposals can be discusssed. Then, make any changes on a [fork](https://help.github.com/articles/fork-a-repo/) of this repo on github, and send us your modifications via a [pull request](https://help.github.com/articles/about-pull-requests/).

### Installation

If you plan to make changes to Pints, install with:

```
$ pip install -e .[dev,docs]
```

This will

1. Install all the dependencies for Pints, including the ones for documentation (docs) and development (dev).
2. Tell Python to use your local pints files when you use `import pints` anywhere on your system.

### Testing:

To check your coding style, use

```
$ flake8
```

To run quick tests, use

```
$ python run-tests.py --unit
```


