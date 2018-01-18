from setuptools import setup, find_packages

# Load text for description and license
with open('README.md') as f:
    readme = f.read()
with open('LICENSE') as f:
    license = f.read()

# Go!
setup(
    # Module name (lowercase)
    name='pints',
    # Remember to keep this in sync with pints/__init__.py
    version='0.0.1',
    description='Probabilistic Inference in Noisy Time-Series',
    long_description=readme,
    license=license,
    # author='',
    # author_email='',
    maintainer='Michael Clerx',
    maintainer_email='michael.clerx@cs.ox.ac.uk',
    url='https://github.com/pints-team/pints',
    # Packages to include
    packages=find_packages(include=('pints', 'pints.*')),
    # List of dependencies
    install_requires=[
        'cma>=2',
        'numpy>=1.8',
        'scipy>=0.13',
        # Note: Matplotlib is loaded for debug plots, but to ensure pints runs
        # on systems without an attached display, it should never be imported
        # outside of plot() methods.
        # Should not be imported
        'matplotlib>=1.5',
    ],
    extras_require={
        'docs': [
            'guzzle-sphinx-theme',  # Nice theme for docs
            'sphinx>=1.5',          # For doc generation
        ],
        'dev': [
            'flake8>=3'             # For code style checking
        ],
    },
)
