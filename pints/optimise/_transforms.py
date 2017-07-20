#
# Parameter transformation classes for implementing boundaries
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
class _TriangleWaveTransform(object):
    """
    Transforms parameters from an unbounded space to a bounded space, using a
    triangle waveform (``/\/\/\...``).
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.range = upper - lower
        self.range2 = 2 * self.range
    def __call__(self, x, *args):
        y = np.remainder(x - self.lower, self.range2)
        z = np.remainder(y, self.range)
        return ((self.lower + z) * (y < self.range)
            + (self.upper - z) * (y >= self.range))

