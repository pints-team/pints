[![Unit tests on multiple python versions](https://github.com/pints-team/pints/actions/workflows/unit-test-python-coverage.yml/badge.svg)](https://github.com/pints-team/pints/actions/workflows/unit-test-python-coverage.yml)
[![Unit tests on multiple operating systems](https://github.com/pints-team/pints/actions/workflows/unit-test-os-coverage.yml/badge.svg)](https://github.com/pints-team/pints/actions/workflows/unit-test-os-coverage.yml)
[![codecov](https://codecov.io/gh/pints-team/pints/branch/main/graph/badge.svg)](https://codecov.io/gh/pints-team/pints)
[![Change-point testing code](https://raw.githubusercontent.com/pints-team/change-point-testing/main/badge-code.svg)](https://github.com/pints-team/change-point-testing)
[![Change-point testing results](https://raw.githubusercontent.com/pints-team/change-point-testing/main/badge-results.svg)](https://www.cs.ox.ac.uk/projects/PINTS/functional-testing)
[![binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/pints-team/pints/main?filepath=examples)
[![readthedocs](https://readthedocs.org/projects/pints/badge/?version=latest)](http://pints.readthedocs.io/en/latest/?badge=latest)

# What is Pints?

PINTS (Probabilistic Inference on Noisy Time-Series) is a framework for optimisation and Bayesian inference on ODE models of noisy time-series, such as arise in electrochemistry and cardiac electrophysiology.

PINTS is described in [this publication in JORS](http://doi.org/10.5334/jors.252), and can be cited using the information given in our [CITATION file](https://github.com/pints-team/pints/blob/main/CITATION).
More information about PINTS papers can be found in the [papers directory](https://github.com/pints-team/pints/tree/main/papers).


## Using PINTS

PINTS can work with any model that implements the [pints.ForwardModel](http://pints.readthedocs.io/en/latest/core_classes_and_methods.html#forward-model) interface.
This has just two methods:

```
n_parameters() --> Returns the dimension of the parameter space.

simulate(parameters, times) --> Returns a vector of model evaluations at
                                the given times, using the given parameters
```

Experimental data sets in PINTS are defined simply as lists (or arrays) of `times` and corresponding experimental `values`.
If you have this kind of data, and if [your model (or model wrapper)](https://github.com/pints-team/pints/blob/main/examples/stats/custom-model.ipynb) implements the two methods above, then you are ready to start using PINTS to infer parameter values using [optimisation](https://github.com/pints-team/pints/blob/main/examples/optimisation/first-example.ipynb) or [sampling](https://github.com/pints-team/pints/blob/main/examples/sampling/first-example.ipynb).

A brief example is shown below:
![An example of using PINTS in an optimisation](https://raw.githubusercontent.com/pints-team/pints/main/example.svg)
_(Left)_ A noisy experimental time series and a computational forward model.
_(Right)_ Example code for an optimisation problem.
The full code can be [viewed here](https://github.com/pints-team/pints/blob/main/examples/sampling/readme-example.ipynb) but a friendlier, more elaborate, introduction can be found on the [examples page](https://github.com/pints-team/pints/blob/main/examples/README.md).

A graphical overview of the methods included in PINTS can be [viewed here](https://pints-team.github.io/pints-methods-overview/).

### Examples and documentation

PINTS comes with a number of [detailed examples](https://github.com/pints-team/pints/blob/main/examples/README.md), hosted here on github.
In addition, there is a [full API documentation](http://pints.readthedocs.io/en/latest/), hosted on readthedocs.io.


## Installing PINTS

The latest release of PINTS can be installed without downloading (cloning) the git repository, by opening a console and typing

```
$ pip install --upgrade pip
$ pip install pints
```

Note that you'll need Python 3.6 or newer.

If you prefer to have the latest cutting-edge version, you can instead install from the repository, by typing

```
$ git clone https://github.com/pints-team/pints.git
$ cd pints
$ pip install -e .[dev,docs]
```

To uninstall again, type:

```
$ pip uninstall pints
```


## What's new in this version of PINTS?

To see what's changed in the latest release, see the [CHANGELOG](https://github.com/pints-team/pints/blob/main/CHANGELOG.md).


## Contributing to PINTS

If you'd like to help us develop PINTS by adding new methods, writing documentation, or fixing embarassing bugs, please have a look at these [guidelines](https://github.com/pints-team/pints/blob/main/CONTRIBUTING.md) first.


## License

PINTS is fully open source. For more information about its license, see [LICENSE](https://github.com/pints-team/pints/blob/main/LICENSE.md).


## Get in touch

Questions, suggestions, or bug reports? [Open an issue](https://github.com/pints-team/pints/issues) and let us know.

Alternatively, feel free to email us at `pints at maillist.ox.ac.uk`.
