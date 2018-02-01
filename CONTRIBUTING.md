# Contributing to Pints

If you'd like to contribute to Pints (thanks!), please have a look at the [guidelines below](#workflow).

If you're already familiar with our workflow, maybe have a quick look at the [pre-commit checks](#pre-commit-checks):



## Pre-commit checks

Before you commit any code, please perform the following checks:

- [No style issues](#coding-style-guideliens): `$ flake8`
- [All tests pass](#testing): `$ python run-tests.py --unit2 --unit3`
- [The documentation builds](#building-the-documentation): `$ cd docs` and then `$ make clean; make html`



## Workflow

We use [GIT](https://en.wikipedia.org/wiki/Git) and [GitHub](https://en.wikipedia.org/wiki/GitHub) to coordinate our work. When making any kind of update, we try to follow the procedure below.

### Before you begin

1. Create an [issue](https://guides.github.com/features/issues/) where new proposals can be discusssed before any coding is done.
2. Create a [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) of this repo (ideally on your own [fork](https://help.github.com/articles/fork-a-repo/)), where all changes will be made
3. Download the source code onto your local system, by [cloning](https://help.github.com/articles/cloning-a-repository/) the repository (or your fork of the repository).
4. [Install](#installation) Pints with the developer options.
5. [Test](#testing) if your installation worked, using the test script `run-tests.py`.

You now have everything you need to start making changes!

### When coding

5. Pints is developed in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), and makes heavy use of [NumPy](https://en.wikipedia.org/wiki/NumPy) (see also [NumPy for MatLab users](https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html) and [Python for R users](http://blog.hackerearth.com/how-can-r-users-learn-python-for-data-science)).
6. Make sure to follow our [coding style guidelines](#coding-style-guidelines).
7. Commit your changes to your branch with useful, descriptive commit messages: Remember these are publically visible and should still make sense a few months ahead in time. While developing, you can keep using the github issue you're working on as a place for discussion. [Refer to your commits](https://stackoverflow.com/questions/8910271/how-can-i-reference-a-commit-in-an-issue-comment-on-github) when discussing specific lines of code.

### Merging your changes with Pints

8. Pints has online documentation at http://pints.readthedocs.io/. To make sure any new methods or classes you added show up there, please read the [documentation](#documentation) section.
9. If you added a major new feature, perhaps it should be showcased in an [example notebook](#example-notebooks).
10. [Test your code!](#testing)
11. When you feel your code is finished, or at least warrants serious discussion
    - [Run these pre-commit checks](#pre-commit-checks)
    - Create a [pull request](https://help.github.com/articles/about-pull-requests/) (PR) on [Pints' GitHub page](https://github.com/pints-team/pints).
12. Once a PR has been created, it will be reviewed by any member of the community. Changes might be suggested which you can make by simply adding new commits to the branch. When everything's finished, someone with the right GitHub permissions will merge your changes into Pints master repository.

Finally, if you really, really, _really_, love developing Pints, have a look at the current [project infrastructure](#infrastructure).



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

When you commit your changes they will be checked against flake8 automatically (see [infrastructure](#infrastructure).

### Python 2 and 3

Python is currently in a long, long transition phase from Python 2 to Python 3. Pints supports both Python 2 (version 2.7 and upwards) and Python 3 (version 3.4 and upwards).
The easiest way to write code that works on both versions is to write for Python 3, (avoiding the more spectacular new features) and [then test on both versions](#testing).

In addition, most scripts start with these lines:

```
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
```

These are ignored by Python 3, but tell Python 2 to mimmick some of its features. Notably, the ``division`` package changes the result of ``3 / 2`` from ``1`` to ``1.5`` (this means you can write ``1 / x`` instead of ``1.0 / x``).



## Testing

All code requires testing. We use the [unittest](https://docs.python.org/3.3/library/unittest.html) package for our tests.

To run quick tests, use

```
$ python run-tests.py --unit
```

### Writing tests

Every new feature should have its own test. To create ones, have a look at the `test` directory and see if there's a test for a similar method. Copy-pasting this is a good way to start.

Next, add some simple (and speedy!) tests of your main features. If these run without exceptions that's a good start! Next, check the output of your methods using any of these [assert methods](https://docs.python.org/3.3/library/unittest.html#assert-methods).



## Documentation

Pints is documented in several ways.

First and foremost, every method and every class should have a [docstring](https://www.python.org/dev/peps/pep-0257/) that describes in plain terms what it does, and what the expected input and output is.

These docstrings can be fairly simple, but can also make use of [reStructuredText](http://docutils.sourceforge.net/docs/user/rst/quickref.html), a markup language designed specifically for writing [technical documentation](https://en.wikipedia.org/wiki/ReStructuredText). For example, you can link to other Pints classes by writing `:class:``pints.ForwardModel`` ` or another method using `:meth:``run()`` `.

In addition, we write a (very) small bit of documentation in separate reStructuredText files in the `docs` directory. Most of what these files do is simply import docstrings from the source code. But they also do things like add tables and indexes. If you've added a new class to a module, search the `docs` directory for that modules `.rst` file and add your class (in alphabetical order) to its index. If you've added a whole new module, copy-paste another module's file and add a link to your new file in the appropriate `index.rst` file.

Using [Sphinx](http://www.sphinx-doc.org/en/stable/) the documentation in `docs` can be converted to HTML, PDF, and other formats. In particular, we use it to generate the documentation on http://pints.readthedocs.io/

### Building the documentation

TODO


### Example notebooks

Jupyter notebooks

TODO







## Infrastructure

Installation

setup.py (setuptools)
requirements.txt (pip)
requirements-dev.txt (pip)

Testing: travis

coverage:

binder
