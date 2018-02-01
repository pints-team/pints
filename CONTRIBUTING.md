# Contributing to Pints

If you'd like to contribute to Pints (thanks!), please have a look at the [guidelines below](#workflow).

If you're already familiar with our workflow, maybe have a quick look at the [pre-commit checks](#pre-commit-checks) directly below.

## Pre-commit checks

Before you commit any code, please perform the following checks:

- [No style issues](#coding-style-guidelines): `$ flake8`
- [All tests pass](#testing): `$ python run-tests.py --unit2 --unit3`
- [The documentation builds](#building-the-documentation): `$ cd docs` and then `$ make clean; make html`



## Workflow

We use [GIT](https://en.wikipedia.org/wiki/Git) and [GitHub](https://en.wikipedia.org/wiki/GitHub) to coordinate our work. When making any kind of update, we try to follow the procedure below.

### A. Before you begin

1. Create an [issue](https://guides.github.com/features/issues/) where new proposals can be discusssed before any coding is done.
2. Create a [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) of this repo (ideally on your own [fork](https://help.github.com/articles/fork-a-repo/)), where all changes will be made
3. Download the source code onto your local system, by [cloning](https://help.github.com/articles/cloning-a-repository/) the repository (or your fork of the repository).
4. [Install](#installation) Pints with the developer options.
5. [Test](#testing) if your installation worked, using the test script `run-tests.py`.

You now have everything you need to start making changes!

### B. Writing your code

5. Pints is developed in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), and makes heavy use of [NumPy](https://en.wikipedia.org/wiki/NumPy) (see also [NumPy for MatLab users](https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html) and [Python for R users](http://blog.hackerearth.com/how-can-r-users-learn-python-for-data-science)).
6. Make sure to follow our [coding style guidelines](#coding-style-guidelines).
7. Commit your changes to your branch with useful, descriptive commit messages: Remember these are publically visible and should still make sense a few months ahead in time. While developing, you can keep using the github issue you're working on as a place for discussion. [Refer to your commits](https://stackoverflow.com/questions/8910271/how-can-i-reference-a-commit-in-an-issue-comment-on-github) when discussing specific lines of code.
8. If you want to add a dependency on another library, or re-use code you found somewhere else, have a look at [these guidelines](#dependencies-and-reusing-code).

### C. Merging your changes with Pints

8. Pints has online documentation at http://pints.readthedocs.io/. To make sure any new methods or classes you added show up there, please read the [documentation](#documentation) section.
9. If you added a major new feature, perhaps it should be showcased in an [example notebook](#example-notebooks).
10. [Test your code!](#testing)
11. When you feel your code is finished, or at least warrants serious discussion, run the [pre-commit checks](#pre-commit-checks) and then create a [pull request](https://help.github.com/articles/about-pull-requests/) (PR) on [Pints' GitHub page](https://github.com/pints-team/pints).
12. Once a PR has been created, it will be reviewed by any member of the community. Changes might be suggested which you can make by simply adding new commits to the branch. When everything's finished, someone with the right GitHub permissions will merge your changes into Pints master repository.

Finally, if you really, really, _really_ love developing Pints, have a look at the current [project infrastructure](#infrastructure).



## Installation

To install Pints with all developer options, use:

```
$ pip install -e .[dev,docs]
```

This will

1. Install all the dependencies for Pints, including the ones for documentation (docs) and development (dev).
2. Tell Python to use your local pints files when you use `import pints` anywhere on your system.



## Coding style guidelines

Pints follows the [PEP8 recommendations](https://www.python.org/dev/peps/pep-0008/) for coding style. These are very common guidelines, and community tools have been developed to check how well projects implement them.

We use [flake8](http://flake8.pycqa.org/en/latest/) to check our PEP8 adherence. To try this on your system, navigate to the Pints directory in a console and type

```
$ flake8
```

When you commit your changes they will be checked against flake8 automatically (see [infrastructure](#infrastructure)).

### Naming

Naming is hard. In general, we aim for descriptive class, method, and argument names. Avoid abbreviations when possible without making names overly long, so `mean` is better than `mu`, but a class name like `AdaptiveMCMC` is fine.

Class names are CamelCase, and start with an upper case letter, for example `SuperDuperMCMC`. Method and variable names are lower case, and use underscores for word separation, for example `x` or `iteration_count`.

### Python 2 and 3

Python is currently in a long, long transition phase from Python 2 to Python 3. Pints supports both Python 2 (version 2.7 and upwards) and Python 3 (version 3.4 and upwards).
The easiest way to write code that works on both versions is to write for Python 3, (avoiding the more spectacular new features) and [then test on both versions](#testing).

In addition, most scripts start with these lines:

```
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
```

These [future imports](https://docs.python.org/2/library/__future__.html) are ignored by Python 3, but tell Python 2 to mimmick some of its features. Notably, the ``division`` package changes the result of ``3 / 2`` from ``1`` to ``1.5`` (this means you can write ``1 / x`` instead of ``1.0 / x``).


## Dependencies and reusing code

While it's a bad idea for developers to "reinvent the wheel", it's important for users to get a _reasonably sized download and an easy install_. In addition, external libraries can sometimes cease to be supported, and when they contain bugs it might take a while before fixes become available as automatic downloads to Pints users.
For these reasons, all dependencies in Pints should be thought about carefully, and discussed on GitHub.

Direct inclusion of code from other packages is possible, as long as their license permits it and is compatible with ours, but again should be considered carefully and discussed in the group. Snippets from blogs and stackoverflow can often be included without attribution, but if they solve a particularly nasty problem (or are very hard to read) it's often a good idea to attribute them via a link in our source code.




## Testing

All code requires testing. We use the [unittest](https://docs.python.org/3.3/library/unittest.html) package for our tests.

To run quick tests, use

```
$ python run-tests.py --unit
```

### Writing tests

Every new feature should have its own test. To create ones, have a look at the `test` directory and see if there's a test for a similar method. Copy-pasting this is a good way to start.

Next, add some simple (and speedy!) tests of your main features. If these run without exceptions that's a good start! Next, check the output of your methods using any of these [assert methods](https://docs.python.org/3.3/library/unittest.html#assert-methods).

### Running more tests

If you want to check your tests on Python 2 and 3, use

```
$ python run-tests.py --unit2 --unit3
```

When you commit anything to Pints, these checks will also be run automatically (see [infrastructure](#infrastructure)).

### Testing notebooks

To test all example notebooks, use

```
$ python run-tests.py --books
```

If notebooks fail because of changes to pints, it can be a bit of a hassle to debug. In these cases, you can create a temporary export of a notebook's Python content using

```
$ python run-tests.py --debook examples/notebook-name.ipynb script.py
```


## Documentation

Pints is documented in several ways.

First and foremost, every method and every class should have a [docstring](https://www.python.org/dev/peps/pep-0257/) that describes in plain terms what it does, and what the expected input and output is.

These docstrings can be fairly simple, but can also make use of [reStructuredText](http://docutils.sourceforge.net/docs/user/rst/quickref.html), a markup language designed specifically for writing [technical documentation](https://en.wikipedia.org/wiki/ReStructuredText). For example, you can link to other Pints classes by writing `:class:``pints.ForwardModel`` ` or another method using `:meth:``run()`` `.

In addition, we write a (very) small bit of documentation in separate reStructuredText files in the `docs` directory. Most of what these files do is simply import docstrings from the source code. But they also do things like add tables and indexes. If you've added a new class to a module, search the `docs` directory for that modules `.rst` file and add your class (in alphabetical order) to its index. If you've added a whole new module, copy-paste another module's file and add a link to your new file in the appropriate `index.rst` file.

Using [Sphinx](http://www.sphinx-doc.org/en/stable/) the documentation in `docs` can be converted to HTML, PDF, and other formats. In particular, we use it to generate the documentation on http://pints.readthedocs.io/

### Building the documentation

To test and debug the documentation, it's best to build it locally. To do this, make sure you have the relevant dependencies installed (see [installation](#installation)), navigate to your pints directory in a console, and then type:

```
cd docs
make clean
make html
```

Next, open a browser, and navigate to your local pints directory (by typing the path, or part of the path into your location bar). Then have a look at `<your pints path>/docs/build/html/index.html`.


### Example notebooks

Major Pints features are showcased in [Jupyter notebooks](https://jupyter.org/) stored in the [examples] directory. Which features are "major" is of course wholy subjective, so please discuss on GitHub first!

All example notebooks should be listed in [examples/EXAMPLES.md]. Please follow the (naming and writing) style of existing notebooks where possible.




## Infrastructure

### Setuptools

Installation of Pints _and dependencies_ is handled via [setuptools](http://setuptools.readthedocs.io/)

Configuration files:

```
setup.py
```

Note that this file must be kept in sync with

- The version number in `pints/__init__.py`
- The dependencies and dependency structure of the [pip requirements files](#pip).

### PIP

Installation of Pints dependencies can also be handled by [PIP](http://pip.readthedocs.io/). This is needed for certain other tools too.

Configuration files ([format](https://pip.readthedocs.io/en/1.1/requirements.html)):

```
requirements.txt
requirements-dev.txt
requirements-docs.txt
```

Note that these files must be kept in sync with

- The dependencies and dependency structure of setuptools `setup.py`.

The requirements files link to each other, so that calling `$ pip install -r requirements-dev.txt` will install everything listed in `requirements.txt` and `requirements-docs.txt` as well.

### Travis CI

All committed code is tested using [Travis CI](https://travis-ci.org/), tests are published on [https://travis-ci.org/pints-team/pints].

Configuration files:

```
.travis.yaml
```

Unit tests and flake8 testing is done for every commit. A nightly cronjob also tests the notebooks.

### Codecov

Code coverage (how much of our code is actually seen by the unit tests) is tested using [Codecov](https://docs.codecov.io/), a report is visible on [https://codecov.io/gh/pints-team/pints].

Configuration files:

```
.coveragerc
```

### Read the Docs

Documentation is built using [https://readthedocs.org/] and published on [http://pints.readthedocs.io/].

### Binder

Editable notebooks are made available using [Binder](mybinder.readthedocs.io) at [https://mybinder.org/v2/gh/pints-team/pints/master?filepath=examples].

Configuration files:

```
postBuild
```

### Flake8

[Style checking](#coding-style-guidelines) is performed using [flake8](http://flake8.pycqa.org/en/latest/).

Configuration files:

```
.flake8
```

### GitHub

GitHub does some magic with particular filesnames. In particular:

- The first page people see when they go to [our GitHub page](https://github.com/pints-team/pints) displays the contents of [README.md], which is written in the [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) format. Some guidelines can be found [here](https://help.github.com/articles/about-readmes/).
- The license for using Pints is stored in [LICENSE], and [automatically](https://help.github.com/articles/adding-a-license-to-a-repository/) linked to by GitHub.
- This file, [CONTRIBUTING.md] is recognised as the contribution guidelines and a link is [automatically](https://github.com/blog/1184-contributing-guidelines) displayed when new issues or pull requests are created.
