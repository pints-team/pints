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
import nbconvert
import subprocess
import unittest

def run_coverage_tests():
    """
    Runs unit tests using unittest framework, exits if they don't finish.
    """
    suite = unittest.defaultTestLoader.discover('test',pattern='test*.py')
    unittest.TextTestRunner(verbosity=2).run(suite)

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


def run_notebook_tests():
    """
    Runs Jupyter notebook tests. Exits if they fail.
    """
    if not scan_for_notebooks('examples'):
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


def test_notebook(path, executable='python'):
    """
    Tests a single notebook, exists if it doesn't finish.
    """
    print('Testing ' + path + ' ... ', end='')
    sys.stdout.flush()

    # Load notebook, convert to python
    e = nbconvert.exporters.PythonExporter()
    code, __ = e.from_filename(path)

    # Remove coding statement, if present
    code = '\n'.join([x for x in code.splitlines() if x[:9] != '# coding'])

    # Tell matplotlib not to produce any figures
    env = dict(os.environ)
    env['MPLBACKEND'] = 'Template'

    # Run in subprocess
    cmd = [executable] + ['-c', code]
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        stdout, stderr = p.communicate()
        # TODO: Use p.communicate(timeout=3600) if Python3 only
        if p.returncode != 0:
            # Show failing code, output and errors before returning
            print('ERROR')
            print('-- script ' + '-' * (79 - 10))
            for i, line in enumerate(code.splitlines()):
                j = str(1 + i)
                print(j + ' ' * (5 - len(j)) + line)
            print('-- stdout ' + '-' * (79 - 10))
            print(stdout)
            print('-- stderr ' + '-' * (79 - 10))
            print(stderr)
            print('-' * 79)
            return False
    except KeyboardInterrupt:
        p.terminate()
        print('ABORTED')
        sys.exit(1)

    # Sucessfully run
    print('ok')
    return True


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
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='''Run all unit tests using the default Python interpreter while \
                collecting coverage.''',
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
        run_notebook_tests()
    if args.coverage:
        has_run = True
        run_coverage_tests()
    if not has_run:
        parser.print_help()
