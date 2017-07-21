PINTS=$(pwd)/../../
BUILD=$(pwd)/build/
env PYTHONPATH=$PYTHONPATH:$PINTS:$BUILD python2 tests/tests.py
