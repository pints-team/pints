#
# Utility classes for Pints
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints

def strfloat(x):
    """
    Converts a float to a string, with maximum precision.
    """
    return pints.FLOAT_FORMAT.format(float(x))
    
