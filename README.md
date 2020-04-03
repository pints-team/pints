[![Unit tests on multiple python versions](https://github.com/pints-team/pints/workflows/Unit%20tests%20on%20multiple%20python%20versions/badge.svg)](https://github.com/pints-team/pints/actions)
[![Unit tests on multiple operating systems](https://github.com/pints-team/pints/workflows/Unit%20tests%20on%20multiple%20operating%20systems/badge.svg)](https://github.com/pints-team/pints/actions)
[![codecov](https://codecov.io/gh/pints-team/pints/branch/master/graph/badge.svg)](https://codecov.io/gh/pints-team/pints)
[![Functional testing code](https://github.com/pints-team/functional-testing/blob/master/badge-code.svg)](https://github.com/pints-team/functional-testing)
[![Functional testing results](https://github.com/pints-team/functional-testing/blob/master/badge-results.svg)](https://www.cs.ox.ac.uk/projects/PINTS/functional-testing)
[![binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/pints-team/pints/master?filepath=examples)
[![readthedocs](https://readthedocs.org/projects/pints/badge/?version=latest)](http://pints.readthedocs.io/en/latest/?badge=latest)
[![BCH compliance](https://bettercodehub.com/edge/badge/pints-team/pints?branch=master)](https://bettercodehub.com/results/pints-team/pints)

# What is Pints?

PINTS (Probabilistic Inference on Noisy Time-Series) is a framework for optimisation and Bayesian inference on ODE models of noisy time-series, such as arise in electrochemistry and cardiac electrophysiology.

PINTS is described in [this publication in JORS](http://doi.org/10.5334/jors.252), and can be cited using the information given in our [CITATION file](./CITATION).
More information about PINTS papers can be found in the [papers directory](./papers/).


## Using PINTS

PINTS can work with any model that implements the [pints.ForwardModel](http://pints.readthedocs.io/en/latest/core_classes_and_methods.html#forward-model) interface.
This has just two methods:

```
n_parameters() --> Returns the dimension of the parameter space.

simulate(parameters, times) --> Returns a vector of model evaluations at
                                the given times, using the given parameters
```

Experimental data sets in PINTS are defined simply as lists (or arrays) of `times` and corresponding experimental `values`.
If you have this kind of data, and if [your model (or model wrapper)](examples/writing-a-model.ipynb) implements the two methods above, then you are ready to start using PINTS to infer parameter values using [optimisation](examples/optimisation-first-example.ipynb) or [sampling](examples/sampling-first-example.ipynb).

A brief example is shown below:  
![An example of using PINTS in an optimisation](example.svg)  
_(Left)_ A noisy experimental time series and a computational forward model.
_(Right)_ Example code for an optimisation problem.
The full code can be [viewed here](examples/readme-example.ipynb) but a friendlier, more elaborate, introduction can be found on the [examples page](examples/README.md).

A graphical overview of the methods included in PINTS can be [viewed here](https://pints-team.github.io/pints-methods-overview/).

### Examples and documentation

PINTS comes with a number of [detailed examples](examples/README.md), hosted here on github.
In addition, there is a [full API documentation](http://pints.readthedocs.io/en/latest/), hosted on readthedocs.io.


## Installing PINTS

You'll need the following requirements:

- Python 2.7 or Python 3.5+
- Python libraries: `cma matplotlib numpy scipy tabulate`

These can easily be installed using `pip`. To do this, first make sure you have the latest version of pip installed:

```
$ pip install --upgrade pip
```

Then navigate to the path where you downloaded PINTS to, and install both PINTS and its dependencies by typing:

```
$ pip install .
```

To install PINTS as a [developer](CONTRIBUTING.md), use

```
$ pip install -e .[dev,docs]
```

To uninstall again, type

```
$ pip uninstall pints
```

## Contributing to PINTS

If you'd like to help us develop PINTS by adding new methods, writing documentation, or fixing embarassing bugs, please have a look at these [guidelines](CONTRIBUTING.md) first.

## License

PINTS is fully open source. For more information about its license, see [LICENSE](LICENSE.md).

## Get in touch

Questions, suggestions, or bug reports? [Open an issue](https://github.com/pints-team/pints/issues) and let us know.

Alternatively, feel free to email us at `pints at maillist.ox.ac.uk`.
