#!/usr/bin/env bash
#
# Runs all unit tests included in PINTS.
#
# To run all tests:
#   $ ./run-test.sh
#
# To run a specific test:
#   $ test/test_logistic_model.py
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
python3 -m unittest discover -v test
python2 -m unittest discover -v test
