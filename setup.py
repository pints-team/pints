#
# Pints setuptools script
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from setuptools import setup, find_packages

# Load text for description and license
with open('README.md') as f:
    readme = f.read()


# Read version number from file
def load_version():
    try:
        import os
        root = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(root, 'pints', 'version'), 'r') as f:
            version = f.read().strip().split(',')
        return '.'.join([str(int(x)) for x in version])
    except Exception as e:
        raise RuntimeError('Unable to read version number (' + str(e) + ').')


# Go!
setup(
    # Module name (lowercase)
    name='pints',
    version=load_version(),

    # Description
    description='Probabilistic Inference in Noisy Time-Series',
    long_description=readme,
    long_description_content_type='text/markdown',

    # License name
    license='BSD 3-clause license',

    # Maintainer information
    # author='',
    # author_email='',
    maintainer='PINTS Team',
    maintainer_email='pints@maillist.ox.ac.uk',
    url='https://github.com/pints-team/pints',

    # Project URLs
    project_urls={
        'Bug Tracker': 'https://github.com/pints-team/pints/issues',
        'Documentation': 'https://pints.readthedocs.io',
        'Source Code': 'https://github.com/pints-team/pints',
    },

    # Packages to include
    packages=find_packages(include=('pints', 'pints.*')),

    # Include non-python files (via MANIFEST.in)
    include_package_data=True,

    # List of dependencies
    install_requires=[
        'cma>=2',
        'numpy>=1.8',
        'scipy>=0.14',
        # Note: Matplotlib is loaded for debug plots, but to ensure pints runs
        # on systems without an attached display, it should never be imported
        # outside of plot() methods.
        'matplotlib>=1.5',
        'tabulate',
        'threadpoolctl',
    ],
    extras_require={
        'docs': [
            'sphinx>=1.5, !=1.7.3',     # For doc generation
        ],
        'dev': [
            'flake8>=3',            # For code style checking
            'jupyter',              # For documentation and testing
            'nbconvert',
            'traitlets',
        ],
        'stan': [
            'pystan>=3',
        ]
    },
    python_requires='>=3.7',
)
