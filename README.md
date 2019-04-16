[![travis](https://travis-ci.org/pints-team/pints.svg?branch=master)](https://travis-ci.org/pints-team/pints)
[![appveyor](https://ci.appveyor.com/api/projects/status/k8xvn7md0pte2gsi/branch/master?svg=true)](https://ci.appveyor.com/project/MichaelClerx/pints/branch/master)
[![codecov](https://codecov.io/gh/pints-team/pints/branch/master/graph/badge.svg)](https://codecov.io/gh/pints-team/pints)
[![functional](https://github.com/pints-team/functional-testing-results/blob/master/badge.svg)](https://github.com/pints-team/functional-testing-results)
[![binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/pints-team/pints/master?filepath=examples)
[![readthedocs](https://readthedocs.org/projects/pints/badge/?version=latest)](http://pints.readthedocs.io/en/latest/?badge=latest)
[![BCH compliance](https://bettercodehub.com/edge/badge/pints-team/pints?branch=master)](https://bettercodehub.com/results/pints-team/pints)

# What is Pints?

Pints (Probabilistic Inference on Noisy Time-Series) is a framework for optimisation and bayesian inference problems with ODE models of noisy time-series, such as arise in electrochemistry and cardiac electrophysiology.

## Using Pints

Pints can work with any model that implements the [pints.ForwardModel](http://pints.readthedocs.io/en/latest/core_classes_and_methods.html#forward-model) interface. This has just two methods:

```
n_parameters() --> Returns the dimension of the parameter space.

simulate(parameters, times) --> Returns a vector of model evaluations at
                                the given times, using the given parameters
```

If your model implements these methods - [or you can write a wrapper class that does](examples/writing-a-model.ipynb) - you can start using Pints for [optimisation](examples/optimisation-first-example.ipynb) or [sampling](examples/sampling-first-example.ipynb).

A simple example could look like this:

```
# Create a model
class MyModel(pints.ForwardModel):
    def n_parameters():
        return 2
    def simulate(parameters, times):
        ... Run a simulation implemented in Python
        ... or in a Python C/C++ extension
        ... or even call a binary in a subprocess
        return simulated_data

# Load your experimental data
times, values = ... Load your data as lists or numpy arrays

# Combine model and data into a Problem
problem = pints.SingleOutputProblem(model, times, values)

# Select an ErrorMeasure, or a LogLikelihood
# Here we'll use a log likelihood that assumes Gaussian noise with sigma=3
log_likelihood = pints.GaussianKnownNoiseLogLikelihood(problem, 3)

# Run an optimisation to find parameters that give a good fit
guess = [10, 2]
obtained_parameters, obtained_log_likelihood = pints.optimise(log_likelihood, guess)

# Use MCMC to sample the log_likelihood near the found solutions
initial_points = [obtained_parameters, obtained_parameters]
samples = pints.mcmc_sample(log_likelihood, 2, initial_points)
```

For details, please see [our long list of detailed examples](examples/README.md)

### Examples and documentation

Pints comes with a number of [detailed examples](examples/README.md), hosted here on github. In addition, there is a [full API documentation](http://pints.readthedocs.io/en/latest/), hosted on readthedocs.io.

### Citing Pints

If you use PINTS in any scientific work, please [credit our work with a citation](./CITATION).


## Installing Pints

You'll need the following requirements:

- Python 2.7 or Python 3.4+
- Python libraries: `cma` `numpy` `matplotlib` `scipy`

These can easily be installed using `pip`. To do this, first make sure you have the latest version of pip installed:

```
$ pip install --upgrade pip
```

Then navigate to the path where you downloaded Pints to, and install both Pints and its dependencies by typing:

```
$ pip install .
```

Or, if you want to install Pints as a [developer](CONTRIBUTING.md), use

```
$ pip install -e .[dev,docs]
```

To uninstall again, type

```
$ pip uninstall pints
```

## Contributing to Pints

If you'd like to help us develop Pints by adding new methods, writing documentation, or fixing embarassing bugs, please have a look at these [guidelines](CONTRIBUTING.md) first.

## License

Pints is fully open source. For more information about its license, see [LICENSE](./LICENSE).

