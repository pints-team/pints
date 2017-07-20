#
# Defines an abstract forward model
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
class ForwardModel(object):
    """
    Defines an interface for forward models.
    """
    
    def __init__(self):
        super(ForwardModel, self).__init__()

    def dimension(self):
        """
        Returns the dimension of the parameter space.
        """
        raise NotImplementedError
        
    def simulate(self, parameters, times):
        """
        Runs a forward simulation with the given ``parameters`` and returns a
        time-series with data points corresponding to the given ``times``.
        
        Arguments:
        
        ``parameters``
            An ordered list of parameter values.
        ``times``
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.

        Note: For efficiency, neither ``parameters`` or ``times`` should be
        copied when `simulate` is called.
        
        
        copied: No
        changes to either should be made by processes running in parallel while
        a simulation is being run!
        """
        raise NotImplementedError
