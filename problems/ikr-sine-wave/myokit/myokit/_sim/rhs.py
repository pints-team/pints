#
# Right-hand-side equation benchmarking tool.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import os
import sys
import timeit
import myokit
# Location of C source file
SOURCE_FILE = 'rhs.c'
class RhsBenchmarker(myokit.CModule):
    """
    Benchmarks a model's right hand side (rhs).

    To enable partial benchmarking the argument ``variables`` can be used to
    pass in a list of variables whose combined evaluation time can be obtained
    by using :meth:`bench_part()`. Only non-constant, non-bound variables can
    be selected.

    By default, the given list of variables are the only variables included in
    a partial simulation. This behaviour can be inverted by setting
    ``exclude_selected=True``. With this setting, all variables except those in
    the given list will be tested.

    A valid myokit model should be provided as the ``model`` argument.
    """
    _index = 0 # Unique id
    def __init__(self, model, variables=None, exclude_selected=False):
        super(RhsBenchmarker, self).__init__()
        # Require a valid model
        model.validate()
        # Clone model
        self._model = model.clone()
        # Check given variables
        self._check_variables(variables)
        # Extension module id
        RhsBenchmarker._index += 1
        module_name = 'myokit_RhsBenchmarker_' + str(RhsBenchmarker._index)
        # Distutils arguments
        args = {
            'module_name'      : module_name,
            'model'            : self._model,
            'variables'        : self._variables,
            'exclude_selected' : exclude_selected,
            }
        fname = os.path.join(myokit.DIR_CFUNC, SOURCE_FILE)
        # Debug
        if myokit.DEBUG:
            print(self._code(fname, args, 
                line_numbers=myokit.DEBUG_LINE_NUMBERS))
            import sys
            sys.exit(1)
        # Create extension
        self._ext = self._compile(module_name, fname, args, ['m'])
    def bench_full(self, log, repeats=40000, fastest=False):
        """
        Benchmarks the entire RHS for every position found in the given log
        file.

        The argument ``log`` should point to a :class:`myokit.DataLog` or
        similar containing the values of all state variables and any bound
        variables used in the model.
        
        The benchmark can be performed in two ways:
        
        ``fastest=False (default)``
            Each point in the given log is calculated ``repeats`` number of
            times. For each point, the average time it took to do a single
            calculation is returned.
        ``fastest=True``
            Each point in the given log is calculated ``repeats`` number of
            times. For each point, the fastest evaluation time is returned.
            
        """
        self._check_log(log)
        # Get first and last+1 position to check
        start = 0
        stop  = len(log[self._model.states().next().qname()])
        # Check repeats
        repeats = int(repeats)
        # Create benchmarking function
        bench = timeit.default_timer
        # Run
        times = self._ext.bench_full(bench, log, start, stop, repeats, fastest)
        return times
    def bench_part(self, log, repeats=40000, fastest=False):
        """
        This method benchmarks the combined evaluation time of the RHS for all
        variables passed in when this RhsBenchmarker was created. The combined
        RHS is benchmarked ``repeats`` number of times for every point in the
        given log.

        The argument ``log`` should point to a :class:`myokit.DataLog` or
        similar containing the values of all state variables and any bound
        variables used in the model.

        Only the selected variables RHS will be evaluated, not their
        dependencies or nested children variables. Bound and constant variables
        cannot be selected. If state variables are chosen, their derivative is
        evaluated.
        
        The benchmark can be performed in two ways:
        
        ``fastest=False (default)``
            Each point in the given log is calculated ``repeats`` number of
            times. For each point, the average time it took to do a single
            calculation is returned.
        ``fastest=True``
            Each point in the given log is calculated ``repeats`` number of
            times. For each point, the fastest evaluation time is returned.
            
        """
        self._check_log(log)
        # Get first and last+1 position to check
        start = 0
        stop  = len(log[self._model.states().next().qname()])
        # Check repeats
        repeats = int(repeats)
        # Create benchmarking function
        bench = timeit.default_timer
        # Run
        times = self._ext.bench_part(bench, log, start, stop, repeats, fastest)
        return times
    def mean(self, times):
        """
        Like meth:`mean_std()` but returns only the final mean.
        """
        return self.mean_std(times)[0]
    def mean_std(self, times):
        """
        Removes outliers and then calculates the mean and standard deviation of
        the given benchmarked times.
        """
        import numpy as np
        times = np.array(times, copy=False)
        # Remove outliers twice
        for i in xrange(0, 2):
            avg = np.mean(times)
            std = np.std(times)
            s3 = std*3.0
            times = times[(times > (avg - s3)) * (times < (avg + s3))]
        return np.mean(times), np.std(times)
    def _check_log(self, log):
        """
        Checks the given log for compatibility with this benchmarker.
        """
        n = None
        for state in self._model.states():
            state = state.qname()
            if not state in log:
                raise ValueError('State variable <' + str(state) + '> not'
                    ' found in simulation log.')
            m = len(log[state])
            if n is None:
                n = m
            elif n != m:
                raise ValueError('All lists in the simulation log must'
                    ' have the same length. Found entries with size ' + str(n)
                    + ' and ' + str(m) + '.')
    def _check_variables(self, variables):
        """
        Checks the given variable list for compatibility with this benchmarker
        and stores a result in self._variables.
        """
        if variables is None:
            self._variables = []
        else:
            self._variables = set()
            for var in variables:
                if type(var) in (str, unicode):
                    # String? Then get variable from model
                    var = self._model.get(var)
                else:
                    # Variable object? Then replace by equivalent form internal
                    # cloned model.
                    var = self._model.get(var.qname())
                if var.is_constant():
                    raise ValueError('Given variables can\'t be constant.')
                if var.is_bound():
                    raise ValueError('Given variables can\'t be bound.')
                # Get same variable from cloned internal version, add to set
                self._variables.add(self._model.get(var.qname()))
            # Create list from set
            self._variables = list(self._variables)
