#
# Tools for working with Markov models
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
from __future__ import division
import numpy as np
import myokit
class LinearModel(object):
    """
    Represents a linear Markov model of an ion channel extracted from a
    :class:`myokit.Model`.
    
    The class assumes the markov model can be written as::
    
        dot(x) = A(V,p) * x
        I = B(V,p) * x
        
    where ``V`` is the membrane potential, ``p`` is a set of parameters and
    ``A`` and ``B`` are the matrices that relate the state ``x`` to its
    derivative ``dot(x)`` and a current ``I``.
    
    ``A`` and ``B`` can contain non-linear functions, but should be simple
    scalar matrices when evaluated for a fixed ``V`` and ``p``.
    
    The model variables to treat as parameter are specified by the user when
    the model is created. Any other variables, for example state variables such
    as intercellular calcium or constants such as temperature, are fixed when
    the markov model is created and can no longer be changed.
    
    To create a :class:`Markov`, pass in a :class:`myokit.Model` and select a
    list of states. All other states will be fixed at their current value and
    an attempt will be made to write the remaining state equations as linear
    combinations of the states. If this is not possible, a :class:`ValueError`
    is raised. The membrane potential must be indicated using the label
    ``membrane_potential`` or by passing it in as ``vm``.
   
    The current variable is optional, if no current is specified by the user
    the relation ``I = B * x`` is dropped and no ``B`` is calculated.
    
    Example::
    
        import myokit
        import myokit.lib.markov as markov
        
        # Load a model from disk
        model = myokit.load_model('some-model.mmt')
        
        # Select the relevant states and parameters
        states = [
            'ina.C1',
            'ina.C2',
            'ina.O',
            ...
            ]
        parameters = [
            'ina.p1',
            'ina.p2',
            ...
            ]
        current = 'ina.INa'
            
        # Extract a markov model
        mm = markov.LinearModel(model, states, parameters, current)

        # Get the matrices A and B such that dot(x) = A * x and I = B * x
        # where ``x`` is the state vector and ``I`` is the current.
        A, B = mm.matrices(membrane_potential=-40)
        print(A)
    
    Alternatively, a LinearModel can be constructed from a single component
    using the method :meth:`from_component() <LinearModel.from_component()>`::
    
        import myokit
        import myokit.lib.markov as markov
        
        # Load a model from disk
        model = myokit.load_model('some-model.mmt')
        
        # Extract a markov model
        mm = markov.LinearModel.from_component(model.get('ina'))
        
    Arguments:
    
    ``model``
        The model to work with.
    ``states``
        An ordered list of state variables (or state variable names) from
        ``model``. All remaining state variables will be frozen in place. Each
        state's derivative must be a linear combination of the other states.
    ``parameters``
        A list of parameters to maintain in their symbolic form.
    ``current``
        The markov model's current variable. The current must be a linear
        combination of the states (for example ``g * (V - E) * (O1 + O2)``)
        where ``O1`` and ``O2`` are states. If no current variable is specified
        ``None`` can be used instead.
    ``vm``
        The variable indicating membrane potential. If set to ``None``
        (default) the method will search for a variable with the label
        ``membrane_potential``.

    *Note: this class requires Sympy to be installed.*
    """
    # NOTE: A LinearModel must be immutable!
    # This ensures that the simulations don't have to clone it (which would be
    # costly and difficult).
    # A LinearModel has a method "matrices()" which can evaluate A and B
    # for its default values or newly passed in values, but it never updates
    # its internal state in any way!
    def __init__(self, model, states, parameters=None, current=None, vm=None):
        super(LinearModel, self).__init__()
        # Import sympy, or raise an ImportError if we can't.
        from myokit.formats import sympy
        #
        # Check input
        #
        # Clone model
        self._model = model.clone()
        del(model)
        # Check and collect state variables
        self._states = []
        for state in states:
            if isinstance(state, myokit.Variable):
                state = state.qname()
            try:
                state = self._model.get(str(state), myokit.Variable)
            except KeyError:
                raise ValueError('Unknown state: <' + str(state) + '>.')
            if not state.is_state():
                raise ValueError('Variable <' + state.qname() + '> is not a'
                    ' state.')
            if state in self._states:
                raise ValueError('State <' + state.qname() + '> was added'
                    ' twice.')
            self._states.append(state)
        del(states)
        # Check and collect parameter variables
        unique = set()
        self._parameters = []
        if parameters is None:
            parameters = []
        for parameter in parameters:
            if isinstance(parameter, myokit.Variable):
                parameter = parameter.qname()
            try:
                parameter = self._model.get(parameter, myokit.Variable)
            except KeyError:
                raise ValueError('Unknown parameter: <' + str(parameter)
                    + '>.')
            if not parameter.is_literal():
                raise ValueError('Unsuitable parameter: <' + str(parameter)
                    + '>.')
            if parameter in unique:
                raise ValueError('Parameter listed twice: <' + str(parameter)
                    + '>.')
            unique.add(parameter)
            self._parameters.append(parameter)
        del(unique)
        del(parameters)
        # Check current variable
        if current is not None:
            if isinstance(current, myokit.Variable):
                current = current.qname()
            current = self._model.get(current, myokit.Variable)
            if current.is_state():
                raise ValueError('Current variable can not be a state.')
        self._current = current
        del(current)
        # Check membrane potential variable
        if vm is None:
            vm = self._model.label('membrane_potential')
        if vm is None:
            raise ValueError('A membrane potential must be specified as'
                ' `vm` or using the label `membrane_potential`.')
        if isinstance(vm, myokit.Variable):
            vm = vm.qname()
        self._membrane_potential = self._model.get(vm)
        if self._membrane_potential in self._parameters:
            raise ValueError('Membrane potential should not be included in the'
                ' list of parameters.')
        if self._membrane_potential in self._states:
            raise ValueError('The membrane potential should not be included in'
                ' the list of states.')
        del(vm)
        #
        # Demote unnecessary states, remove bindings and validate model.
        #
        # Get values of all states
        # Note: Do this _before_ changing the model!
        self._default_state = np.array([v.state_value() for v in self._states])
        # Freeze remaining, non-markov-model states
        s = self._model.state() # Get state values before changing anything!
        for k, state in enumerate(self._model.states()):
            if state not in self._states:
                state.demote()
                state.set_rhs(s[k])
        del(s)
        # Unbind everything except time
        for label, var in self._model.bindings():
            if label != 'time':
                var.set_binding(None)
        # Check if current variable depends on selected states
        # (At this point, anything not dependent on the states is a constant)
        if self._current is not None and self._current.is_constant():
            raise ValueError('Current must be a function of the markov'
                ' model\'s state variables.')
        # Validate modified model
        self._model.validate()
        #
        # Create functions:
        #   matrix_function(vm, p1, p2, ...   ) --> A, B
        #       where dot(x) = Ax and I = Bx
        #   rate_list_function(vm, p1, p2, ...) --> R
        #       where R contains tuples (i, j, rij)
        #
        # Create a list of inputs to the functions
        self._inputs = self._parameters + [self._membrane_potential]
        # Get the default values for all inputs
        self._default_inputs = np.array([v.eval() for v in self._inputs])
        # Create functions
        self._matrix_function = None
        self._rate_list_function = None
        self._generate_functions()
        #
        # Partial validation
        #
        # Check if dependencies are bidirectional
        for s in self._states:
            for d in s.refs_to(state_refs=True):
                if s not in d.refs_to(state_refs=True):
                    raise Exception('State <' + s.qname() + '> depends on <'
                        + d.qname() + '> but not vice versa.')
        # Check the sum of all states is 1
        tolerance = 1e-8
        x = np.sum(self._default_state)
        if np.abs(x - 1) > tolerance:
            raise Exception('The sum of states is not equal to 1: ' + str(x))
        # Check the sum of all derivatives per column
        A, B = self.matrices()
        for k, x in enumerate(np.sum(A, axis=0)):
            if abs(x) > tolerance:
                raise Exception('Derivatives in column ' + str(1 + k)
                    + ' sum to non-zero value: ' + str(x) + '.')
    def _generate_functions(self):
        """
        Creates a function that takes parameters as input and returns matrices
        A and B.
        
        (This method is called only once, by the constructor, but it's
        complicated enough to warrant its own method...)
        """
        from myokit.formats import sympy
        # Create mapping from states to index
        state_indices = {}
        for k, state in enumerate(self._states):
            state_indices[state] = k
        # Extract expressions for state & current variables:
        #  1. Get expanded equation
        #  2. Convert to sympy
        #  3. Simplify / attempt to rewrite as linear combination
        #  4. Import from sympy
        expressions = []
        for state in self._states:
            e = state.rhs().clone(expand=True, retain=self._inputs)
            e = sympy.write(e)
            e = sympy.read(e.expand(), self._model)
            expressions.append(e)
        if self._current is not None:
            e = self._current.rhs().clone(expand=True, retain=self._inputs)
            e = sympy.write(e)
            current_expression = sympy.read(e.expand(), self._model)
        # Create parametrisable matrices to evaluate the state & current
        #  1. For each expression, get the terms. This works because Sympy
        #     returns the equations as an addition of terms (or a single term).
        #  2. Each term can be written as f*s, where f is a constant factor and
        #     s is a state. Gather these factors
        n = len(self._states)
        A = []      # Ode matrix
        T = set()   # List of transitions
        for i in xrange(n):
            A.append([myokit.Number(0) for j in xrange(n)])
        for row, e in enumerate(expressions):
            # Scan terms
            for term in _list_terms(e):
                # Look for the single state this expression depends on
                state = None
                for ref in term.references():
                    if ref.var().is_state():
                        if state is not None:
                            raise ValueError('Unable to write expression as'
                                ' linear combination of states: ' + str(e))
                        state = ref.var()
                # Get factor
                state, factor = _find_factor(term, e)
                # Add factor to transition matrix
                col = state_indices[state]
                cur = A[row][col]
                if cur != myokit.Number(0):
                    factor = myokit.Plus(cur, factor)
                A[row][col] = factor
                # Store transition in transition list
                if row != col:
                    T.add((col, row)) # A is mirrored
        # Create a parametrisable matrix for the current
        B = [myokit.Number(0) for i in xrange(n)]
        if self._current is not None:
            for term in _list_terms(current_expression):
                state = None
                for ref in term.references():
                    if ref.var().is_state():
                        if state is not None:
                            raise ValueError('Unable to write expression for'
                                ' current as linear combination of states: '
                                + str(current_expression))
                        state = ref.var()
                state, factor = _find_factor(term, current_expression)
                col = state_indices[state]
                cur = B[col]
                if cur != myokit.Number(0):
                    factor = myokit.Plus(cur, factor)
                B[col] = factor
        # Create list of transition rates and associated equations
        T = list(T)
        T.sort()
        R = []
        for i in xrange(len(A)):
            for j in xrange(len(A)):
                if (i, j) in T:
                    R.append((i, j, A[j][i])) # A is mirrored
        del(T)
        #
        # Create function to create parametrisable matrices
        #
        self._model.reserve_unique_names('A', 'B', 'n', 'numpy')
        writer = myokit.numpywriter()
        w = writer.ex
        head = 'def matrix_function('
        head += ','.join([w(p.lhs()) for p in self._inputs])
        head += '):'
        body = []
        body.append('A = numpy.zeros((n, n))')
        zero = myokit.Number(0)
        for i, row in enumerate(A):
            for j, e in enumerate(row):
                if e != zero:
                    body.append('A[' + str(i) + ',' +  str(j) + '] = ' + w(e))
        body.append('B = numpy.zeros(n)')
        for j, e in enumerate(B):
            if e != zero:
                body.append('B[' +  str(j) + '] = ' + w(e))
        body.append('return A, B')        
        code = head +'\n' + '\n'.join(['    ' + line for line in body])
        globl = {'numpy' : np, 'n' : n,}
        local = {}
        #TODO What's the difference between these two?
        #If the function version works, use that for python3 compatibility!
        #exec(code, globl, local)
        exec code in globl, local
        self._matrix_function = local['matrix_function']
        #
        # Create function to return list of transition rates
        #
        self._model.reserve_unique_names('R', 'n', 'numpy')
        head = 'def rate_list_function('
        head += ','.join([w(p.lhs()) for p in self._inputs])
        head += '):'
        body = []
        body.append('R = []')
        for i, j, e in R:
            body.append('R.append((' + str(i) +','+  str(j) +','+ w(e) +'))')
        body.append('return R')        
        code = head +'\n' + '\n'.join(['    ' + line for line in body])
        globl = {'numpy' : np}
        local = {}
        #exec(code, globl, local)
        exec code in globl, local
        self._rate_list_function = local['rate_list_function']
    def current(self):
        """
        Returns the name of the current variable used by this model, or None if
        no current variable was specified.
        """
        return self._current
    def default_membrane_potential(self):
        """
        Returns this markov model's default membrane potential value.
        """
        return self._default_inputs[-1]
    def default_parameters(self):
        """
        Returns this markov model's default parameter values
        """
        return list(self._default_inputs[:-1])
    def default_state(self):
        """
        Returns this markov model's default state values.
        """
        return list(self._default_state)
    @staticmethod
    def from_component(component, states=None, parameters=None, current=None,
            vm=None):
        """
        Creates a Markov model from a component, using the following rules:
        
          1. Every state in the component is a state in the Markov model
          2. Every unnested constant in the component is a parameter
          3. The component should contain exactly one unnested intermediary 
             variable whose value depends on the model states, this will be
             used as the current variable.
          4. The model contains a variable labeled "membrane_potential".
        
        Any of the automatically set variables can be overridden using the
        keyword arguments ``states``, ``parameters``, ``current`` and ``vm``.
        
        The parameters, if determined automatically, will be specified in
        alphabetical order (using a natural sort).
        """
        model = component.model()
        # Get or check states
        if states is None:
            # Get state variables
            states = [x for x in component.variables(state=True)]
            # Sort by state indice
            states.sort(key = lambda x: x.indice())
        else:
            # Make sure states are variables. This is required to automatically
            # find a current variable.
            states_in = states
            states = []
            for state in states_in:
                if isinstance(state, myokit.Variable):
                    state = state.qname()
                states.append(model.get(state))
        # Get parameters
        if parameters is None:
            # Get parameters
            parameters = [x for x in component.variables(const=True)
                if x.is_literal()]
            # Sort by qname, using natural sort
            parameters.sort(key=lambda x: myokit.natural_sort_key(x.qname()))
        # Get current
        if current is None:
            currents = []
            for x in component.variables(inter=True):
                for y in x.refs_to(state_refs=True):
                    if y in states:
                        # Found a candidate!
                        currents.append(x)
                        break
            if len(currents) > 1:
                raise ValueError('The given component has more than one'
                    ' variable that could be a current: '
                    + ', '.join(['<' + x.qname() + '>' for x in currents])
                    + '.')
            try:
                current = currents[0]
            except IndexError:
                raise ValueError('No current variable found.')
        # Get membrane potential
        if vm is None:
            vm = model.label('membrane_potential')
            if vm is None:
                raise ValueError('The model must define a variable labeled as'
                    ' "membrane_potential".')
        # Create and return LinearModel
        return LinearModel(model, states, parameters, current, vm)
    def matrices(self, membrane_potential=None, parameters=None):
        """
        For a given value of the ``membrane_potential`` and a list of values
        for the ``parameters``, this method calculates and returns the matrices
        ``A`` and ``B`` such that::
        
            dot(x) = A * x
            I = B * x
            
        where ``x`` is the state vector and ``I`` is the current.
        
        Arguments:
        
        ``membrane_potential``
            The value to use for the membrane potential, or ``None`` to use the
            value from the original :class:`myokit.Model`.
        ``parameters``
            The values to use for the parameters, given in the order they were
            originally specified in (if the model was created using
            :meth:`from_component()`, this will be alphabetical order).
        
        """
        inputs = list(self._default_inputs)
        if membrane_potential is not None:
            inputs[-1] = float(membrane_potential)
        if parameters is not None:
            if len(parameters) != len(self._parameters):
                raise ValueError('Illegal parameter vector size: '
                    + str(len(self._parameters)) + ' required, '
                    + str(len(paramaters)) + ' provided.')
            inputs[:-1] = [float(x) for x in parameters]
        return self._matrix_function(*inputs)
    def membrane_potential(self):
        """
        Returns the name of the membrane potential variable used by this model.
        """
        return self._membrane_potential.qname()
    def parameters(self):
        """
        Returns the names of the parameter variables used by this model.
        """
        return [v.qname() for v in self._parameters]
    def rates(self, membrane_potential=None, parameters=None):
        """
        For a given value of the ``membrane_potential`` and a list of values
        for the ``parameters``, this method calculates and returns an ordered
        list of tuples ``(i, j, rij)`` such that ``rij`` is a non-zero
        transition rate from the ``i``-th state to the ``j``-th state.
        
        Arguments:
        
        ``membrane_potential``
            The value to use for the membrane potential, or ``None`` to use the
            value from the original :class:`myokit.Model`.
        ``parameters``
            The values to use for the parameters, given in the order they were
            originally specified in (if the model was created using
            :meth:`from_component()`, this will be alphabetical order).
        
        """
        inputs = list(self._default_inputs)
        if membrane_potential is not None:
            inputs[-1] = float(membrane_potential)
        if parameters is not None:
            if len(parameters) != len(self._parameters):
                raise ValueError('Illegal parameter vector size: '
                    + str(len(self._parameters)) + ' required, '
                    + str(len(paramaters)) + ' provided.')
            inputs[:-1] = [float(x) for x in parameters]
        return self._rate_list_function(*inputs)
    def states(self):
        """
        Returns the names of the state variables used by this model.
        """
        return [v.qname() for v in self._states]
    def steady_state(self, membrane_potential=None, parameters=None):
        """
        Analytically determines a steady state solution for this Markov model.
        
        ``membrane_potential``
            The value to use for the membrane potential, or ``None`` to use the
            value from the original :class:`myokit.Model`.
        ``parameters``
            The values to use for the parameters, given in the order they were
            originally specified in (if the model was created using
            :meth:`from_component()`, this will be alphabetical order).
        
        """
        # Calculate Jacobian and derivatives
        A, B = self.matrices(membrane_potential, parameters)
        A = np.matrix(A)
        del(B)
        # Set up reduced system with full rank: dot(x) = Ay + B
        B = A[:-1, -1]
        A = A[:-1, :-1] - B
        # Check eigenvalues
        if np.max(np.linalg.eigvals(A) >= 0):
            raise Exception('System has positive eigenvalues: won\'t'
                ' converge to steady state!')
        # Solve system Ax + B = 0 --> Ax = -B
        x = np.linalg.solve(A, -B)
        # Recreate full state vector and return
        x = np.array(x).reshape((len(x),))
        x = np.concatenate((x, [1 - np.sum(x)]))
        return x
class AnalyticalSimulation(object):
    """
    Analytically evaluates a :class:`LinearModel`'s state over a given set of
    points in time.
    
    Solutions are calculated for the "law of large numbers" case, i.e. without
    stochastic behavior. The solution algorithm is based on eigenvalue
    decomposition.
    
    Each simulation object maintains an internal state consisting of
    
    * The current simulation time
    * The current state
    * The default state
    
    When a simulation is created, the simulation time is set to zero and both
    the current and default state are initialized using the ``LinearModel``.
    After each call to :meth:`run()` the time and current state are updated,
    so that each successive call to run continues where the previous simulation
    left off.
    
    A :class:`protocol <myokit.Protocol>` can be used to set the membrane
    potential during the simulation, or the membrane potential can be adjusted
    manually between runs.
    
    Example::
        
        import myokit
        import myokit.lib.markov as markov
                
        # Create a linear markov model
        m = myokit.load_model('clancy-1999.mmt')
        m = markov.LinearModel.from_component(m.get('ina'))

        # Create an analytical simulation object
        s = markov.AnalyticalSimulation(m)

        # Run a simulation
        s.set_membrane_potential(-30)
        d = s.run(10)

        # Show the results
        import matplotlib.pyplot as pl
        pl.figure()
        pl.subplot(211)
        for state in m.states():
            pl.plot(d.time(), d[state], label=state)
        pl.legend(loc='center right')
        pl.subplot(212)
        pl.plot(d.time(), d[m.current()])
        pl.show()

    """
    def __init__(self, model, protocol=None):
        super(AnalyticalSimulation, self).__init__()
        # Check model
        if not isinstance(model, LinearModel):
            raise ValueError('First parameter must be a `LinearModel`.')
        self._model = model
        # Check protocol
        if protocol is None:
            self._protocol = None
        elif not isinstance(protocol, myokit.Protocol):
            raise ValueError('Protocol must be a myokit.Protocol object')
        else:
            self._protocol = protocol.clone()
        # Check if we have a current variable
        self._has_current = self._model.current() is not None
        # Set state
        self._state = np.array(self._model.default_state(), copy=True,
            dtype=float)
        # Set default state
        self._default_state = np.array(self._state, copy=True)
        # Get membrane potential
        self._membrane_potential = self._model.default_membrane_potential()
        # Get parameters
        self._parameters = np.array(self._model.default_parameters(),
            copy=True, dtype=float)
        # Cached matrices
        self._cached_matrices = None
        # Cached partial solution (eigenvalue decomposition etc.)
        self._cached_solution = None
        # Time variable
        self._time = 0
        # If protocol was given, create pacing system, update vm
        self._pacing = None
        if self._protocol:
            self._pacing = myokit.PacingSystem(self._protocol)
            self._membrane_potential = self._pacing.advance(self._time)
    def current(self, state):
        """
        Calculates the current for a given state.
        """
        if not self._has_current:
            raise ValueError('The used model did not specify a current'
                ' variable.')
        A, B = self._matrices()
        return B.dot(state)
    def default_state(self):
        """
        Returns the default state used by this simulation.
        """
        return list(self._default_state)
    def _matrices(self):
        """
        Returns the (cached or re-generated) matrices ``A`` and ``B`` if this
        simulation's model has a current variable, or just ``A`` if it doesn't.
        """
        if self._cached_matrices is None:
            self._cached_matrices = self._model.matrices(
                self._membrane_potential, self._parameters)
        return self._cached_matrices
    def membrane_potential(self):
        """
        Returns the currently set membrane potential.
        """
        return self._membrane_potential
    def parameters(self):
        """
        Returns the currently set parameter values.
        """
        return list(self._parameters)
    def pre(self, duration):
        """
        Performs an unlogged simulation for ``duration`` time units and uses
        the final state as the new default state.
        
        After the simulation:
        
        - The simulation time is **not** affected
        - The current state and the default state are updated to the final
          state reached in the simulation.
        
        Calls to :meth:`reset` after using :meth:`pre` will set the current
        state to this new default state.
        """
        # Check arguments
        duration = float(duration)
        if duration < 0:
            raise ValueError('Duration must be non-negative.')
        # Run
        # This isn't much faster, but this way the simulation's interface is
        # similar to the standard simulation one.
        old_time = self._time
        self.run(duration, log_interval=2*duration)
        # Update default state
        self._default_state = np.array(self._state, copy=True)
        # Reset time, reset protocol
        self._time = old_time
        if self._protocol:
            self._pacing = myokit.PacingSystem(self._protocol)
            self._membrane_potential = self._pacing.advance(self._time)
    def reset(self):
        """
        Resets the simulation:
        
        - The time variable is set to zero.
        - The state is set to the default state.
        
        """
        self._time = 0
        self._state = np.array(self._default_state, copy=True)
        if self._protocol:
            self._pacing = myokit.PacingSystem(self._protocol)
            self._membrane_potential = self._pacing.advance(self._time)
    def run(self, duration, log=None, log_interval=0.01):
        """
        Runs a simulation for ``duration`` time units.
        
        After the simulation:
        
        - The simulation time will be increased by ``duration`` time units.
        - The simulation state will be updated to the last reached state.

        Arguments:
        
        ``duration``
            The number of time units to simulate.
        ``log``
            A log from a previous run can be passed in, in which case the
            results will be appended to this one.
        ``log_interval``
            The time between logged points.

        Returns a :class:`myokit.DataLog` with the simulation results.
        """
        # Check arguments
        duration = float(duration)
        if duration < 0:
            raise ValueError('Duration must be non-negative.')
        log_interval = float(log_interval)
        if log_interval <= 0:
            raise ValueError('Log interval must be greater than zero.')
        # Set up logging
        vm_key = self._model._membrane_potential
        time_key = self._model._model.time().qname()
        if log is None:
            # Create new log
            log = myokit.DataLog()
            log[time_key] = []
            log.set_time_key(time_key)
            log[vm_key] = []
            for key in self._model.states():
                log[key] = []
            if self._has_current:
                log[self._model.current()] = []
        else:
            # Check existing log
            n = 2 + len(self._state) + (1 if self._has_current else 0)
            if len(log.keys()) > n:
                raise ValueError('Invalid log: contains extra keys.')
            test = self._model.states()
            test.append(time_key)
            test.append(vm_key)
            if self._has_current:
                test.append(self._model.current())
            try:
                for key in test:
                    log[key]
            except KeyError:
                raise ValueError('Invalid log: missing entry for <'+key+'>.')
        # Run simulation
        if self._protocol is None:
            # User defined membrane potential
            self._run(duration, log, log_interval)
        else:
            # Voltage clamp
            tfinal = self._time + duration
            while self._time < tfinal:
                # Run simulation
                tnext = min(tfinal, self._pacing.next_time())
                self._run(tnext - self._time, log, log_interval)
                # Update pacing
                self._membrane_potential = self._pacing.advance(tnext, tfinal)
                self._cached_matrices = None
                self._cached_solution = None
        # Return
        return log
    def _run(self, duration, log, log_interval):
        """
        Runs a simulation with the current membrane potential.
        """
        # Simulate with fixed V
        times = np.arange(0, duration, log_interval)
        if self._has_current:
            states, currents = self.solve(times)
        else:
            states = self.solve(times)
        times += self._time
        # Log results
        key = log.time_key()
        log[key] = np.concatenate((log[key], times))
        for i, key in enumerate(self._model.states()):
            log[key] = np.concatenate((log[key], states[i]))
        if self._has_current:
            key = self._model.current()
            log[key] = np.concatenate((log[key], currents))
        vm_key = self._model._membrane_potential
        log[vm_key] = np.concatenate((log[vm_key],
            [self._membrane_potential] * len(times)))
        # Now run simulation for final time (which might not be included in the
        # list of logged times, and should not, if you want to be able to
        # append logs without creating duplicate points).
        times = np.array([duration])
        states, currents = self.solve(times)
        # Update simulation state
        self._state = np.array(states[:,-1], copy=True)
        self._time += duration
    def set_default_state(self, state):
        """
        Changes this simulation's default state.
        """
        state = np.array(state, copy=True, dtype=float)
        if len(state) != len(self._state):
            raise ValueError('Wrong size state vector, expecing ('
                 + str(len(self._state)) + ') values.')
        if np.abs(np.sum(state) - 1) > 1e-6:
            raise ValueError('The values in `state` must sum to 1.')
        self._default_state = state
    def set_membrane_potential(self, v):
        """
        Changes the membrane potential used in this simulation.
        """
        if self._protocol:
            raise Exception('Membrane potential can not be set if a protocol'
                ' is used.')
        self._membrane_potential = float(v)
        self._cached_matrices = None
        self._cached_solution = None
    def set_parameters(self, parameters):
        """
        Changes the parameter values used in this simulation.
        """
        if len(parameters) != len(self._parameters):
            raise ValueError('Wrong size parameter vector, expecting ('
                + str(len(self._parameters)) + ') values.')
        self._parameters = np.array(parameters, copy=True, dtype=float)
        self._cached_matrices = None
        self._cached_solution = None
    def set_state(self, state):
        """
        Changes the initial state used by in this simulation.
        """
        state = np.array(state, copy=True, dtype=float)
        if len(state) != len(self._state):
            raise ValueError('Wrong size state vector, expecing ('
                 + str(len(self._state)) + ') values.')
        if np.abs(np.sum(state) - 1) > 1e-6:
            raise ValueError('The values in `state` must sum to 1.')
        self._state = state
    def solve(self, times):
        """
        Evaluates and returns the states at the given times.
        
        In contrast to :meth:`run()`, this method simply evaluates the states
        (and current) at the given times, using the last known settings for
        the state and membrane potential. It does not use a protocol and does
        not take into account the simulation time. After running this method,
        the state and simulation time are *not* updated.
                
        Arguments:
        
        ``times``
            A series of times, where each time must be some ``t >= 0``.
        
        For models with a current variable, this method returns a tuple
        ``(state, current)`` where ``state`` is a matrix of shape
        ``(len(states), len(times))`` and ``current`` is a vector
        of length ``len(times)``.
        
        For models without a current variable, only ``state`` is returned.
        """
        n = len(self._state)
        # Check for cached partial solution
        if self._cached_solution is None:
            # Get matrices
            if self._has_current:
                A, B = self._matrices()
            else:
                A = self._matrices()
                B = None
            # Get eigenvalues, matrix of eigenvectors
            E, P = np.linalg.eig(A)
            E = E.reshape((n, 1))
            PI = np.linalg.inv(P)
            # Cache results
            self._cached_solution = (E, P, PI, B)
        else:
            E, P, PI, B = self._cached_solution
        # Calculate transform of initial state
        y0 = PI.dot(self._state.reshape((n, 1)))
        # Reshape times array
        times = np.array(times, copy=False).reshape((len(times),))
        # Calculate state
        x = P.dot(y0 * np.exp(times * E))
        # Calculate current and/or return
        if self._has_current:
            return x, B.dot(x)
        else:
            return x
    def state(self):
        """
        Returns the initial state used by this simulation.
        """
        return list(self._state)
class DiscreteSimulation(object):
    """
    Performs stochastic simulations of a :class:`LinearModel`'s behavior for a
    finite number of channels.
    
    Simulations are run using the "Direct method" proposed by Gillespie [1].
    
    Each simulation object maintains an internal state consisting of
    
    * The current simulation time
    * The current state
    * The default state
    
    When a simulation is created, the simulation time is set to zero and both
    the current and default state are initialized using the ``LinearModel``.
    After each call to :meth:`run()` the time and current state are updated,
    so that each successive call to run continues where the previous simulation
    left off.
    
    A :class:`protocol <myokit.Protocol>` can be used to set the membrane
    potential during the simulation, or the membrane potential can be adjusted
    manually between runs.
    
    Example::
    
        import myokit
        import myokit.lib.markov as markov
        
        # Create linear markov model
        m = myokit.load_model('clancy-1999.mmt')
        m = markov.LinearModel.from_component(m.get('ina'))

        # Run discrete simulation
        s = markov.DiscreteSimulation(m, nchannels=1000)
        s.set_membrane_potential(-30)
        d = s.run(10)
        
        import matplotlib.pyplot as pl
        for state in m.states():
            pl.step(d.time(), d[state], label=state)
        pl.legend()
        pl.show()

    References
    
    [1] Gillespie (1976) A General Method for Numerically Simulating the
        stochastic time evolution of coupled chemical reactions
        The Journal of Computational Physics, 22, 403-434.

    Arguments:
    
    ``model``
        A :class:`LinearModel`.
    ``nchannels``
        The number of channels to simulate.
        
    """
    def __init__(self, model, protocol=None, nchannels=100):
        # Check model
        if not isinstance(model, LinearModel):
            raise ValueError('First parameter must be a `LinearModel`.')
        self._model = model
        # Check protocol
        if protocol is None:
            self._protocol = None
        elif not isinstance(protocol, myokit.Protocol):
            raise ValueError('Protocol must be a myokit.Protocol object')
        else:
            self._protocol = protocol.clone()
        # Get state and discretize
        nchannels = int(nchannels)
        if nchannels < 1:
            raise ValueError('The number of channels must be at least 1.')
        self._nchannels = nchannels
        # Set state
        self._state = self.discretize_state(self._model.default_state())
        # Set default state
        self._default_state = list(self._state)
        # Set membrane potential
        self._membrane_potential = self._model.default_membrane_potential()
        # Set parameters
        self._parameters = np.array(self._model.default_parameters(),
            copy=True, dtype=float)
        # Cached transition rate list
        self._cached_rates = None
        # Set simulation time
        self._time = 0
        # If protocol was given, create pacing system, update vm
        self._pacing = None
        if self._protocol:
            self._pacing = myokit.PacingSystem(self._protocol)
            self._membrane_potential = self._pacing.advance(self._time)
    def default_state(self):
        """
        Returns the default simulation state.
        """
        return list(self._default_state)
    def discretize_state(self, x):
        """
        Converts a list of fractional state occupancies to a list of channel
        counts.
        
        Arguments:
        
        ``x``
            A fractional state where ``sum(x) == 1``.
            
        Returns a discretized state ``y`` where ``sum(y) = nchannels``.
        """
        x = np.array(x, copy=False, dtype=float)
        if (np.abs(1 - np.sum(x))) > 1e-6:
            raise ValueError('The sum of fractions in the state to be'
                ' discretized must equal 1.')
        y = np.round(x * self._nchannels)
        # To make sure it always sums to 1, correct the value found at the
        # indice with the biggest rounding error.
        i = np.argmax(np.abs(x - y))
        y[i] = 0
        y[i] = self._nchannels - np.sum(y)
        return list(y)
    def membrane_potential(self):
        """
        Returns the current membrane potential.
        """
        return self._membrane_potential
    def number_of_channels(self):
        """
        Returns the number of channels used in this simulation.
        """
        return self._nchannels
    def parameters(self):
        """
        Returns the current parameter values.
        """
        return list(self._parameters)
    def pre(self, duration):
        """
        Performs an unlogged simulation for ``duration`` time units and uses
        the final state as the new default state.
        
        After the simulation:
        
        - The simulation time is **not** affected
        - The current state and the default state are updated to the final
          state reached in the simulation.
        
        Calls to :meth:`reset` after using :meth:`pre` will set the current
        state to this new default state.
        """
        # Check arguments
        duration = float(duration)
        if duration < 0:
            raise ValueError('Duration must be non-negative.')
        # Run
        # This isn't much faster, but this way the simulation's interface is
        # similar to the standard simulation one.
        old_time = self._time
        self.run(duration)
        # Update default state
        self._default_state = list(self._state)
        # Reset time, reset protocol
        self._time = old_time
        if self._protocol:
            self._pacing = myokit.PacingSystem(self._protocol)
            self._membrane_potential = self._pacing.advance(self._time)
    def _rates(self):
        """
        Returns the (cached or regenerated) transition rate list.
        """
        if self._cached_rates is None:
            self._cached_rates = self._model.rates(self._membrane_potential,
                self._parameters)
        return self._cached_rates
    def reset(self):
        """
        Resets the simulation:
        
        - The time variable is set to zero.
        - The state is set to the default state.
        
        """
        self._time = 0
        self._state = list(self._default_state)
        if self._protocol:
            self._pacing = myokit.PacingSystem(self._protocol)
            self._membrane_potential = self._pacing.advance(self._time)
    def run(self, duration, log=None):
        """
        Runs a simulation for ``duration`` time units.
        
        After the simulation:
        
        - The simulation time will be increased by ``duration`` time units.
        - The simulation state will be updated to the last reached state.
                        
        Arguments:
        
        ``duration``
            The number of time units to simulate.
        ``log``
            A log from a previous run can be passed in, in which case the
            results will be appended to this one.

        Returns a :class:`myokit.DataLog` with the simulation results.        
        """
        # Check arguments
        duration = float(duration)
        if duration < 0:
            raise ValueError('Duration must be non-negative.')
        # Set up logging
        time_key = self._model._model.time().qname()
        vm_key = self._model._membrane_potential
        if log is None:
            # Create new log
            log = myokit.DataLog()
            log[time_key] = []
            log.set_time_key(time_key)
            log[vm_key] = []
            for key in self._model.states():
                log[key] = []
        else:
            # Check existing log
            if len(log.keys()) > 2 + len(self._state):
                raise ValueError('Invalid log: contains extra keys.')
            try:
                for key in [vm_key, time_key] + self._model.states():
                    log[key]
            except KeyError:
                raise ValueError('Invalid log: missing entry for <'+key+'>.')
        if self._protocol is None:
            # Simulate with fixed V
            self._run(duration, log)
        else:
            # Voltage clamp
            tfinal = self._time + duration
            while self._time < tfinal:
                # Run simulation
                tnext = min(tfinal, self._pacing.next_time())
                self._run(tnext - self._time, log)
                # Update pacing
                self._membrane_potential = self._pacing.advance(tnext, tfinal)
                self._cached_rates = None
        # Return
        return log
    def _run(self, duration, log):
        """
        Runs a simulation with the current membrane potential.
        """
        # Get logging lists
        log_time = log.time()
        log_states = []
        for key in self._model.states():
            log_states.append(log[key])
        # Get current time and state
        t = self._time
        state = np.array(self._state, copy=True, dtype=int)
        # Get list of transitions
        R  = []     # Transition rates
        SI = []     # From state
        SJ = []     # To state
        for i, j, rij in self._rates():
            SI.append(i)
            SJ.append(j)
            R.append(rij)
        R = np.array(R)
        SI = np.array(SI)
        SJ = np.array(SJ)
        debug = False
        # Run
        n_steps = 0
        t_stop = self._time + duration
        while t < t_stop:
            # Log
            log_time.append(t)
            for i, x in enumerate(state):
                log_states[i].append(x)
            n_steps += 1
            if debug:
                print(t, state)
            # Get lambdas
            lambdas = R * state[SI]
            # Get sum of lambdas
            lsum = np.sum(lambdas)
            # Sample time until next transition from an exponential
            # distribution with mean 1/lsum
            tau = np.random.exponential(1/lsum)
            # Don't step beyond the stopping time!
            if t + tau > t_stop:
                break
            # Get type of transition
            transition = np.random.uniform(0, lsum)
            rsum = 0
            for i, r in enumerate(lambdas):
                rsum += r
                if rsum > transition:
                    break
            if debug:
                print(str(t) + ': ' + str(SI[i]) + ' --> ' + str(SJ[i]))
            # Perform transition
            state[SI[i]] -= 1
            state[SJ[i]] += 1
            t += tau
        # Perform final step using the "brute-force" approach, ensuring we
        # reach self._time + duration exactly.
        # Note that for large tau, the estimates of the probability that
        # something changes may become inaccurate (and > 1)
        # I didn't see this in testing...
        tau = (self._time + duration) - t
        lambdas *= tau
        for i, r in enumerate(lambdas):
            if np.random.uniform(0, 1) < r:
                if debug:
                    print('Final: ' + str(SI[i]) + ' --> ' + str(SJ[i]))
                # Perform transition
                state[SI[i]] -= 1
                state[SJ[i]] += 1
        # Add vm to log
        vm_key = self._model._membrane_potential
        log[vm_key].extend([self._membrane_potential] * n_steps)
        # Update current state and time
        self._state = list(state)
        self._time += duration
    def set_default_state(self, state):
        """
        Changes the default state used in the simulation.
        """
        state = np.asarray(state, dtype=int)
        if np.min(state) < 0:
            raise ValueError('The number of channels in a markov model state'
                ' can not be negative.')
        if np.sum(state) != self._nchannels:
            raise ValueError('The number of channels in the default state'
                ' vector must equal ' + str(self._nchannels) + '.')
        self._default_state = list(state)
    def set_membrane_potential(self, v):
        """
        Changes the membrane potential used in this simulation.
        """
        if self._protocol:
            raise Exception('Membrane potential can not be set if a protocol'
                ' is used.')
        self._membrane_potential = float(v)
        self._cached_rates = None
    def set_parameters(self, parameters):
        """
        Changes the parameter values used in this simulation.
        """
        if len(parameters) != len(self._parameters):
            raise ValueError('Wrong size parameter vector, expecting ('
                + str(len(self._parameters)) + ') values.')
        self._parameters = np.array(parameters, copy=True, dtype=float)
        self._cached_rates = None
    def set_state(self, state):
        """
        Changes the current state used in the simulation (i.e. the number of
        channels in every markov model state).
        """
        state = np.asarray(state, dtype=int)
        if np.min(state) < 0:
            raise ValueError('The state must be given as a list of'
                ' non-negative integers.')
        if np.sum(state) != self._nchannels:
            raise ValueError('The number of channels in the state vector must'
                ' equal ' + str(self._nchannels) + '.')
        self._state = list(state)
    def state(self):
        """
        Returns the current simulation state.
        """
        return list(self._state)
def _list_terms(expression, terms=None):
    """
    Takes an expression tree of myokit.Plus elements and splits it into terms.
    
    This method is specifically for use in analyzing expressions returned by
    Sympy: A more general implementation would look at Minus objects at well,
    but since Sympy has no substraction operator this works fine.
    """
    if terms is None:
        terms = []
    if type(expression) == myokit.Plus:
        for e in expression:
            _list_terms(e, terms)
    else:
        terms.append(expression)
    return terms
def _find_factor(expression, original):
    """
    Takes an expression tree of myokit myokit.Multiply objects and splits the
    expression into a state variable and a constant factor. If the term
    contains multiple states an error is raised.
    """
    t = type(expression)
    if t == myokit.Name:
        var = expression.var()
        if not var.is_state():
            raise ValueError('Unable to write expression as linear'
                ' combination of states: ' + str(original))
        return var, myokit.Number(1)
    elif t == myokit.Multiply:
        a, b = expression
        # Check if a contains a state and b is constant
        ac, bc = a.is_constant(), b.is_constant()
        if (ac and bc) or not (ac or bc):
            raise ValueError('Unable to write expression as linear'
                ' combination of states: ' + str(original))
        if ac:
            a, b = b, a
        # Check if a is a state
        if type(a) == myokit.Name and a.var().is_state():
            return a.var(), b
        else:
            # Get the factor and state from a, multiply by b and return
            state, factor = _find_factor(a, original)
            factor = myokit.Multiply(factor, b)
            return state, factor
    else:
        raise ValueError('Unable to write expression as linear'
            ' combination of states: ' + str(original))
class MarkovModel(object):
    """
    **Deprecated**: Since version 1.22.0 this class has been replaced by the
    classes :class:`LinearModel` and :class:`AnalyticalSimulation`. Please
    update your code to use these classes instead. This class will be removed
    in future versions of Myokit.
    """
    def __init__(self):
        raise NotImplementedError('Please use the class LinearModel instead.')
    @staticmethod
    def from_component(component, states=None, parameters=None, current=None,
            vm=None):
        """
        Creates and returns an :class:`AnalyticalSimulation` using a
        :class:`LinearModel` based on a Myokit model component.
        """
        return AnalyticalSimulation(LinearModel.from_component(
            component, states, parameters, current, vm))
    def __new__(self, model, states, parameters=None, current=None, vm=None):
        return AnalyticalSimulation(LinearModel(
            model, states, parameters, current, vm))
