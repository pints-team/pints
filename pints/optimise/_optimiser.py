#
# Base class for optimisers.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
class Optimiser(object):
    """
    Takes a model and recorded data as input and attempts to find the model
    parameters that best reproduce the recordings.
    
    Arguments:
    
    ``model``
        The model to fit.
    ``times``
        The times to evaluate at.
    ``values``
        The recorded values at the given times.
    ``lower``
        Lower bounds for the model parameters.
    ``upper`
        Upper bounds for the model parameters.
    ``hint=None``
        An optional starting point for the search.
        
    """
    def __init__(model, times, values, lower, upper, hint=None):
        # Store model, get dimension
        self._model = model
        d = model.dimension()
        # Check times and values
        if len(times) != len(values):
            raise ValueError('Times and values arrays must have same length.')
        self._times = np.array(times, copy=True)
        self._values = np.array(values, copy=True)
        # Check boundaries
        if len(lower) != d:
            raise ValueError('Lower bounds must have same dimension as model.')
        if len(upper) != d:
            raise ValueError('Upper bounds must have same dimension as model.')
        self._lower = np.array(lower, copy=True)
        self._upper = np.array(upper, copy=True)
        # Check hint
        if hint is None:
            self._hint = 0.5 * (self._lower + self._upper)
        else:
            if len(hint) != d:
                raise ValueError('Hint must have same dimension as model.')
            self._hint = np.array(hint, copy=True)
            if np.any(self._hint < self._lower):
                raise ValueError('Hint is outside of lower boundaries.')
            if np.any(self._hint > self._upper):
                raise ValueError('Hint is outside of upper boundaries.')
        
    def fit(self):
        #TODO: Write doc
        raise NotImplementedError
