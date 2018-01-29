#!/usr/bin/env python
#
# Runs all unit tests included in Pints.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import sys
import argparse
import subprocess

# Parse input arguments
parser = argparse.ArgumentParser(
    description = 'Run tests for Pints.',
    epilog =
        'To run individual tests, use e.g. $ test/test_logistic_model.py',
    )
parser.add_argument(
    '--dual',
    action = 'store_true',
    help = 'Run tests in both python2 and python3.',
    )
args = parser.parse_args()

# Call tests as subprocess(es)
def test(executable='python'):
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

if args.dual:
    # Run both versions
    test('python3')
    test('python2')
else:
    # Run single, without default interpreter
    test('python')

