#!/usr/bin/env bash
#
# Runs the electrochemistry tests
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
PINTS=$(pwd)/../../
BUILD=$(pwd)/build/
PPATH=$PYTHONPATH:$PINTS:$BUILD
env PYTHONPATH=$PPATH python -m unittest discover -v test
