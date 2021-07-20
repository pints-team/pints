#
# I/O helper classes for Pints
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#


def load_samples(filename, n=None):
    """
    Loads samples from the given ``filename`` and returns a 2d NumPy array
    containing them.

    If the optional argument ``n`` is given, the method assumes there are ``n``
    files, with names based on ``filename`` such that e.g. ``test.csv`` would
    become ``test_0.csv``, ``test_1.csv``, ..., ``test_n.csv``. In this case
    a list of 2d NumPy arrays is returned.

    Assumes the first line in each file is a header.

    See also :meth:`save_samples()`.
    """
    import numpy as np
    import os

    # Define data loading method
    def load(filename):
        with open(filename, 'r') as f:
            lines = iter(f)
            next(lines)  # Skip header
            return np.asarray(
                [[float(x) for x in line.split(',')] for line in lines])

    # Load from filename directly
    if n is None:
        return load(filename)

    # Load from systematically named files
    n = int(n)
    if n < 1:
        raise ValueError(
            'Argument `n` must be `None` or an integer greater than zero.')
    parts = os.path.splitext(filename)
    filenames = [parts[0] + '_' + str(i) + parts[1] for i in range(n)]

    # Check if files exist before loading (saves times)
    for filename in filenames:
        if not os.path.isfile(filename):
            raise FileNotFoundError('File not found: ' + filename)

    # Load and return
    return [load(filename) for filename in filenames]


def save_samples(filename, *sample_lists):
    """
    Stores one or multiple lists of samples at the path given by ``filename``.

    If one list of samples is given, the filename is used as is. If multiple
    lists are given, the filenames are updated to include ``_0``, ``_1``,
    ``_2``, etc.

    For example, ``save_samples('test.csv', samples)`` will store information
    from ``samples`` in ``test.csv``. Using
    ``save_samples('test.csv', samples_0, samples_1)`` will store the samples
    from ``samples_0`` to ``test_0.csv`` and ``samples_1`` to ``test_1.csv``.

    See also: :meth:`load_samples()`.
    """
    import numpy as np
    import os
    import pints

    # Get filenames
    k = len(sample_lists)
    if k < 1:
        raise ValueError('At least one set of samples must be given.')
    elif k == 1:
        filenames = [filename]
    else:
        parts = os.path.splitext(filename)
        filenames = [parts[0] + '_' + str(i) + parts[1] for i in range(k)]

    # Check shapes
    try:
        sample_lists = np.array(sample_lists, dtype=float)
    except ValueError:
        raise ValueError(
            'Sample lists must contain only floats and be of same length.')
    shape = sample_lists[0].shape
    if len(shape) != 2:
        raise ValueError(
            'Samples must be given as 2d arrays (e.g. lists of lists).')

    # Store
    filename = iter(filenames)
    header = ','.join(['"p' + str(j) + '"' for j in range(shape[1])])
    for samples in sample_lists:
        with open(next(filename), 'w') as f:
            f.write(header + '\n')
            for sample in samples:
                f.write(','.join([pints.strfloat(x) for x in sample]) + '\n')

