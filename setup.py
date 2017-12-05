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
    #author='',
    #author_email='',
    maintainer='Michael Clerx',
    maintainer_email='michael.clerx@cs.ox.ac.uk',
    url='https://github.com/pints-team/pints',
    # Packages to include
    packages=find_packages(include=('pints', 'pints.*')),
    # List of dependencies
    install_requires=[
        'cma>=2',
        'guzzle-sphinx-theme',
        'numpy>=1.8',
        'scipy>=0.13',
        ],
)
