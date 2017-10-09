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
pints_dir=`pwd`
electrochemistry_dir=${pints_dir}/problems/electrochemistry

export set PYTHONPATH=$PYTHONPATH:${pints_dir}
cd $electrochemistry_dir
cmake -DCMAKE_BUILD_TYPE=$PINTS_BUILD_TYPE .
make
exit_code=`python -m unittest discover -v test`
cd $pints_dir
exit $exit_code
