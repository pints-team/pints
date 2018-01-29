#!/usr/bin/env python
#
# Runs all unit tests included in Pints.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
import sys
import argparse
import tempfile
import subprocess


def run_unit_tests(executable='python'):
    """
    Runs unit tests in subprocess, exits if they don't finish.
    """
    print('Running with executable `' + executable + '`')
    cmd = [executable] + [
        '-m',
        'unittest',
        'discover',
        '-v',
        'test',
    ]
    try:
        p = subprocess.Popen(cmd)
        ret = p.wait()
        if ret != 0:
            sys.exit(ret)
    except KeyboardInterrupt:
        p.terminate()
        print('')
        sys.exit(1)


def scan_for_notebooks(root, recursive=True):
    """
    Scans for, and tests, all notebooks in a directory.
    """
    ok = True
    debug = False

    # Scan path
    for filename in os.listdir(root):
        path = os.path.join(root, filename)

        # Recurse into subdirectories
        if recursive and os.path.isdir(path):
            # Ignore hidden directories
            if filename[:1] == '.':
                continue
            ok &= scan_for_notebooks(path, recursive)

        # Test notebooks
        if os.path.splitext(path)[1] == '.ipynb':
            if debug:
                print(path)
            else:
                ok &= test_notebook(path)

    # Return True if every notebook is ok
    return ok


def test_notebook(path):
    """
    Tests a single notebook, exists if it doesn't finish.
    """
    print('Testing ' + path, end='')
    with tempfile.NamedTemporaryFile() as output_file:
        cmd = [
            'jupyter',
            'nbconvert',
            '--to',
            'notebook',
            '--execute',
            '--ExecutePreprocessor.timeout=3600',
            '--output',
            output_file.name,
            path
        ]
        with open(os.devnull, 'w') as stdout:
            with open(os.devnull, 'w') as stderr:
                p = subprocess.Popen(
                    cmd,
                    stdout=stdout,
                    stderr=stderr,
                )
                print(' ... ', end='')
                sys.stdout.flush()
                try:
                    ret = p.wait()
                except KeyboardInterrupt:
                    print('\nNotebook test aborted')
                    sys.exit(1)
    if ret == 0:
        print('ok')
    if ret != 0:
        print('FAILED')
    return ret == 0


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run unit tests for Pints.',
        epilog='To run individual unit tests, use e.g.'
               ' $ test/test_logistic_model.py',
    )
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run all unit tests using the default Python interpreter.',
    )
    parser.add_argument(
        '--unit2',
        action='store_true',
        help='Run all unit tests using the `python2` interpreter.',
    )
    parser.add_argument(
        '--unit3',
        action='store_true',
        help='Run all unit tests using the `python3` interpreter.',
    )
    parser.add_argument(
        '--books',
        action='store_true',
        help='Test Jupyter notebooks (using the default jupyter interpreter).',
    )
    args = parser.parse_args()

    # Run tests
    has_run = False
    if args.unit:
        has_run = True
        run_unit_tests('python')
    if args.unit2:
        has_run = True
        run_unit_tests('python2')
    if args.unit3:
        has_run = True
        run_unit_tests('python3')
    if args.books:
        has_run = True
        scan_for_notebooks('examples')
    if not has_run:
        parser.print_help()
