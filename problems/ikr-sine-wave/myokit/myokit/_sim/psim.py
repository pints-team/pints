#
# Runs a simulation with differential objects, to obtain the state and the
# partial derivatives of the state with respect to a list of parameters.
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
import myokit
import numpy as np
# Location of C template
SOURCE_FILE = 'psim.cpp'
class PSimulation(myokit.CppModule):
    """
    Runs a forward-Euler based simulation and calculates the partial
    derivatives of the model variables with respect to a given set of
    parameters.
    
    The simulation is based on automatic differentiation implemented using a
    C++ data type that replaces a single scalar float with a float and a list
    of partial derivatives. Any operations on this pair update both the float
    and the set of derivatives.
    
    The resulting output is a set of logged variables plus a matrix of
    derivatives ``dy/dp`` where ``y`` is a non-constant variable and ``p`` is
    a constant parameter. The variables and parameters to track can be
    specified using :class:`myokit.Variable` objects or by their names. The
    parameters should be given as a list ``parameters`` while the variables
    ``y`` should be given in the list ``variables``.
    
    N.B. Partial derivatives can not be calculated for the functions ``floor``,
    ``ceil`` and ``abs`` or for quotients and remainders. If these are
    encountered the resulting derivatives will be yielded as ``NaN``.

    A protocol can be passed in as ``protocol`` or set later using
    :meth:`set_protocol`.

    The model and protocol passed to the simulation are cloned and stored
    internally. Any changes to the original model or protocol will not affect
    the simulation.

    Simulations maintain an internal state consisting of

      - the current simulation time
      - the current state
      - the partial derivatives of the current state with respect to the
        parameters

    When a simulation is created, the simulation time is set to 0 and the
    state is obtained from the given model. The derivatives matrix is
    initialised as a matrix of size ``(n, m)`` with a row for each of the ``n``
    states and a column for each of ``m`` parameters.
        
    After each call to :meth:`run` the time, state and derivative variables are
    updated so that each successive call to run continues where the previous
    one left off. A :meth:`reset` method is provided that will set the time
    back to 0, revert the current state to the default state and reset the
    calculated derivatives.

    The simulation provides two inputs a variable can bind to:

    ``time``
        This variable contains the simulation time.
    ``pace``
        This variable contains the current value of the pacing variable
        as given by the protocol passed to the Simulation.
        
    No labeled variables are required.
    """
    _index = 0 # Simulation id
    def __init__(self, model, protocol=None, variables=None, parameters=None):
        super(PSimulation, self).__init__()
        # Check presence of variables and parameters arguments (are required
        # arguments but protocol is not...)
        if variables is None:
            raise ValueError('Please specify a set of variables whose'
                ' derivatives should be tracked.')
        if parameters is None:
            raise ValueError('Please specify a set of parameters.')
        # Require a valid model
        if not model.is_valid():
            model.validate()
        model = model.clone()
        self._model = model
        # Set protocol
        self.set_protocol(protocol)
        # Check tracked variables
        if len(variables) != len(set(variables)):
            raise ValueError('Variables to track can only be specified once.')
        self._variables = []
        for v in variables:
            if isinstance(v, myokit.Variable):
                v = v.qname()
            v = self._model.get(v, myokit.Variable)
            if not (v.is_state() or v.is_intermediary()):
                if v.is_bound():
                    raise ValueError('Variables to track cannot be bound to'
                        ' external inputs.')
                else:
                    raise ValueError('Variables to track cannot be constants.')
            self._variables.append(v)
        # Check parameters
        if len(parameters) != len(set(parameters)):
            raise ValueError('Parameters can only be specified once.')
        self._parameters = []
        for p in parameters:
            if isinstance(p, myokit.Variable):
                p = p.qname()
            p = self._model.get(p, myokit.Variable)
            if not p.is_literal():
                if p.is_bound():
                    raise ValueError('Parameters cannot be bound to external'
                        ' inputs.')
                else:
                    raise ValueError('Parameters must be literal constants.')
            self._parameters.append(p)
        del(parameters)
        # Create list of parameter values
        self._values = []
        for p in self._parameters:
            self._values.append(p.rhs().eval())
        # Get state and default state from model
        self._state = self._model.state()
        self._default_state = list(self._state)
        # Create list of state-parameter-derivatives
        ms = len(self._state)
        mp = len(self._parameters)
        self._state_ddp = [0.0] * (ms * mp)
        # Starting time
        self._time = 0
        # Default time step
        self._dt = 0
        self.set_step_size()
        # Unique simulation id
        PSimulation._index += 1
        module_name = 'myokit_PSimulation_' + str(PSimulation._index)
        # Arguments
        args = {
            'module_name' : module_name,
            'model' : self._model,
            'variables' : self._variables,
            'parameters' : self._parameters,
            }
        fname = os.path.join(myokit.DIR_CFUNC, SOURCE_FILE)
        # Debug
        if myokit.DEBUG:
            print(self._code(fname, args,
                line_numbers=myokit.DEBUG_LINE_NUMBERS))
            import sys
            sys.exit(1)
        # Create simulation
        libs = ['m']
        libd = []
        incd = [myokit.DIR_CFUNC]
        self._sim = self._compile(module_name, fname, args, libs, libd, incd)
    def block(self, log, derivatives):
        """
        Takes the output of a simulation (a simulation log and a list of
        derivatives) and combines it into a single :class:`DataBlock2d` object.
        
        Each entry in the log is converted to a 0d entry in the log. The
        calculated derivatives are stored as the 2d field ``derivatives``.
        """
        # Get time data
        tvar = self._model.time().qname()
        try:
            time = log[tvar]
        except KeyError:
            raise ValueError('The given log must contain an entry for <' + tvar
                + '>.')
        # Check shape of derivatives array
        n = len(self._variables)
        m = len(self._parameters)
        shape = (len(time), n, m)
        if derivatives.shape != shape:
            raise ValueError('Wrong input: Expecting a derivatives array of'
                ' shape ' + str(shape) + '.')
        # Create datablock
        block = myokit.DataBlock2d(m, n, time)
        for k, v in log.iteritems():
            if k != tvar:
                block.set0d(k, v)
        block.set2d('derivatives', derivatives)
        return block
    def default_state(self):
        """
        Returns the default state.
        """
        return list(self._default_state)
    def derivatives(self):
        """
        Return the partial derivatives of the current state with respect to the
        parameters
        """
        ms = len(self._state)
        mp = len(self._parameters)
        return np.array(self._state_ddp, copy=True).reshape((ms, mp))
    def reset(self):
        """
        Resets the simulation:

        - The time variable is set to 0
        - The state is set back to the default state
        - The derivatives are set to zero

        """
        # Reset time
        self._time = 0
        # Reset state
        self._state = list(self._default_state)
        # Reset state-parameter-derivatives
        ms = len(self._state)
        mp = len(self._parameters)
        self._state_ddp = [0.0] * (ms * mp)
    def run(self, duration, log=None, log_interval=1, progress=None,
            msg='Running PSimulation'):
        """
        Runs a simulation and returns the logged results. Running a simulation
        has the following effects:

        - The internal state is updated to the last state in the simulation.
        - The simulation's time variable is updated to reflect the time
          elapsed during the simulation.

        The number of time units to simulate can be set with ``duration``.

        The variables to log can be indicated using the ``log`` argument. There
        are several options for its value:

        - ``None`` (default), to log all states.
        - An integer flag or a combination of flags. Options: 
          ``myokit.LOG_NONE``, ``myokit.LOG_STATE``, ``myokit.LOG_INTER``,
          ``myokit.LOG_BOUND``.
        - A list of qnames or variable objects
        - A :class:`myokit.DataLog` obtained from a previous simulation.
          In this case, the newly logged data will be appended to the existing
          log.
           
        For more details on the ``log`` argument, see the function
        :meth:`myokit.prepare_log`.

        The method returns a :class:`myokit.DataLog` and a 3d numpy
        array. In the returned array, the first axis represents the time, 
        the second axis is a tracked variable y and the third is a parameter p
        such that the point ``(t, y, p)`` represents ``dy/dp`` at time ``t``.
        For example, if ``d`` is the array of derivatives, to get the
        derivative of variables ``0`` with respect to parameter 2, use
        ``d[:,0,2]``.
        
        A log entry is created every time *at least* ``log_interval`` time
        units have passed.

        To obtain feedback on the simulation progress, an object implementing
        the :class:`myokit.ProgressReporter` interface can be passed in.
        passed in as ``progress``. An optional description of the current
        simulation to use in the ProgressReporter can be passed in as `msg`.
        """
        # Simulation times
        if duration < 0:
            raise Exception('Simulation time can\'t be negative.')
        tmin = self._time
        tmax = tmin + duration
        # Number of states, state derivatives
        ms = len(self._state)
        mv = len(self._variables)
        mp = len(self._parameters)
        # Final state and final state-parameter-derivative output lists
        state = [0.] * ms
        state_ddp = [0.] * (ms * mp)
        # Parse log argument
        log = myokit.prepare_log(
            log,
            self._model,
            if_empty=myokit.LOG_STATE+myokit.LOG_BOUND,
            allowed_classes=myokit.LOG_STATE+myokit.LOG_BOUND+myokit.LOG_INTER,
            )
        # Logging period (0 = disabled)
        log_interval = float(log_interval)
        if log_interval < 0:
            log_interval = 0
        # Create empty list for variable-parameter-derivative lists
        varab_ddp = []
        # Get progress indication function (if any)
        if progress is None:
            progress = myokit._Simulation_progress
        if progress:
            if not isinstance(progress, myokit.ProgressReporter):
                raise ValueError('The argument `progress` must be either a'
                    ' subclass of myokit.ProgressReporter or None.')
        # Run simulation
        if duration > 0:
            # Initialize
            self._sim.sim_init(
                tmin,
                tmax,
                self._dt,
                list(self._values),
                list(self._state),
                list(self._state_ddp),
                state,
                state_ddp,
                self._protocol,
                log,
                varab_ddp,
                log_interval,
                )
            t = tmin
            try:
                if progress:
                    # Loop with feedback
                    with progress.job(msg):
                        r = 1.0 / duration
                        while t < tmax:
                            t = self._sim.sim_step()
                            if not progress.update(min((t - tmin) * r, 1)):
                                raise myokit.SimulationCancelledError()
                else:
                    # Loop without feedback
                    # (But with repeated returns to Python to allow Ctrl-C etc)
                    while t < tmax:
                        t = self._sim.sim_step()
            finally:
                # Clean even after KeyboardInterrupt or other Exception
                self._sim.sim_clean()
            # Update internal state
            self._state = list(state)
            self._state_ddp = list(state_ddp)
            self._time += duration
            # Convert derivatives to numpy arrays
            varab_ddp = np.array([
                np.array(np.array(x).reshape(mv, mp), copy=True)
                for x in varab_ddp])
        # Return
        return log, varab_ddp
    def set_constant(self, var, value):
        """
        Changes a model constant. Only literal constants (constants not
        dependent on any other variable) can be changed. Constants set as
        parameters cannot be changed with this method but may be set using
        :meth:`set_parameters`.

        The constant ``var`` can be given as a :class:`Variable` or a string
        containing a variable qname. The ``value`` should be given as a float.
        """
        value = float(value)
        if isinstance(var, myokit.Variable):
            var = var.qname()
        var = self._model.get(var)
        if not var.is_literal():
            raise ValueError('The given variable <' + var.qname() + '> is'
                ' not a literal (it depends on other variables)')
        if var in self._parameters:
            raise ValueError('The given variable <' + var.qname() + '> is'
                ' set as a parameter. Use set_parameters() instead.')
        # Update value in compiled simulation module
        self._sim.set_constant(var.qname(), value)
        # Update value in internal model
        self._model.set_value(var.qname(), value)
    def set_parameters(self, values):
        """
        Changes the values of the parameters under investigation.
        
        The argument ``values`` must either be an ordered sequence containing
        the values for every parameter, or a mapping from one or more parameter
        names to their new values.
        
        **N.B. Calling this method will reset the simulation.**
        """
        if isinstance(values, dict):
            # Create list to update so that property change only happens after
            # error checks.
            new_values = list(self._values)
            # Check all key-value pairs
            for k, v in values:
                if isinstance(k, myokit.Variable):
                    k = k.qname()
                try:
                    k = self._model.get(k, myokit.Variable)
                except KeyError:
                    raise ValueError('Unknown parameter: <' + str(k) + '>.')
                try:
                    i = self._parameters.index(k)
                except ValueError:
                    raise ValueError('Variable <' + str(k) + '> was not set as'
                        ' a parameter.')
                new_values[i] = float(v)
            self._values = new_values
        else:
            # Check size of list & set
            if len(values) != len(self._values):
                raise ValueError('Argument `values` should be either a dict or'
                    ' a list of ' + str(len(self._values)) + ' values.')
            self._values = [float(x) for x in values]
        # Reset the simulation: the stored partial derivatives are no longer
        # accurate.
        self.reset()
    def set_protocol(self, protocol=None):
        """
        Changes the pacing protocol used by this simulation.
        """
        if protocol is None:
            self._protocol = None
        else:
            self._protocol = protocol.clone()
    def set_step_size(self, dt=0.01):
        """
        Sets the step size used in the forward Euler solving routine.
        """
        dt = float(dt)
        if dt <= 0:
            raise ValueError('Step size must be greater than zero.')
        self._dt = dt
    def state(self):
        """
        Returns the current state.
        """
        return list(self._state)
    def time(self):
        """
        Returns the current simulation time.
        """
        return self._time
