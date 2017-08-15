#
# Simulation that integrates the Jacobian to obtain the partial derivatives of
# the state vector with respect to the initial conditions.
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
SOURCE_FILE = 'icsim.cpp'
class ICSimulation(myokit.CppModule):
    """
    Runs a forward-Euler based simulation and calculates the partial
    derivatives of the state vector with respect to the initial conditions.
    
    The simulation is based on automatic differentiation implemented using a
    C++ data type that replaces a single scalar float with a float and a list
    of partial derivatives. Any operations on this pair update both the float
    and the set of derivatives. A normal simulation starts with a state
    ``y(tmin)`` and a right-hand side function (RHS) ``f(y) = dy/dt``. It then
    integrates ``f(y)`` from ``tmin`` to ``tmax`` resulting in an output state
    ``y(tmax)``. In this simulation the data type of ``f`` is replaced by a
    ``(f, df/dy)``, where ``df/dy`` is the matrix of partial derivatives of
    ``f`` with respect to ``y``. By integrating ``f`` from ``tmin`` to ``tmax``
    we obtain the state at ``tmax``. This can be seen as a function
    ``F(y(tmin))``, that gives the state at ``tmax`` given ``y(tmin)``. By
    integrating ``df/dy`` the derivative of ``F`` to ``y(tmin)`` is obtained.
    This result allows the sensitivity of the system to its initial conditions
    to be evaluated.
    
    N.B. The partial derivatives can not be calculated for the following 
    functions: ``floor``, ``ceil``, ``abs``, quotients and remainders. If these
    are encountered the resulting derivatives will be yielded as ``NaN``.
    However, in many cases, these functions will only occur as part of a
    condition in an if statement, so the ``NaN``'s won't propagate to the final
    result.

    The model passed to the simulation is cloned and stored internally, so
    changes to the original model object will not affect the simulation.

    A protocol can be passed in as ``protocol`` or set later using
    :meth:`set_protocol`.

    Simulations maintain an internal state consisting of

    - the current simulation time
    - the current state
    - the derivatives of the current state with respect to the initial state

    When a simulation is created, the simulation time is set to 0 and the
    state is obtained from the given model. The initial derivatives matrix is
    an identity matrix of size ``(n, n)``, where ``n`` is the number of states
    in the model.
    After each call to :meth:`run` the time, state and derivative variables are
    updated so that each successive call to run continues where the previous
    one left off. A :meth:`reset` method is provided that will set the 
    time back to 0, revert the current state to the default state and set the
    derivatives back to ``I``.

    The simulation provides two inputs a variable can bind to:

    ``time``
        This variable contains the simulation time.
    ``pace``
        This variable contains the current value of the pacing variable
        as given by the protocol passed to the Simulation.
        
    No labeled variables are required.
    """
    _index = 0 # Simulation id
    def __init__(self, model, protocol=None):
        super(ICSimulation, self).__init__()
        # Require a valid model
        if not model.is_valid():
            model.validate()
        model = model.clone()
        self._model = model
        # Set protocol
        self.set_protocol(protocol)
        # Get state and default state from model
        self._state = self._model.state()
        self._default_state = list(self._state)
        # Create initial list of derivatives
        n = len(self._state)
        self._deriv = [0.0] * n**2
        for i in xrange(n):
            self._deriv[i * (n + 1)] = 1.0
        # Starting time
        self._time = 0
        # Default time step
        self._dt = 0
        self.set_step_size()
        # Unique simulation id
        ICSimulation._index += 1
        module_name = 'myokit_ICSimulation_' + str(ICSimulation._index)
        # Arguments
        args = {
            'module_name' : module_name,
            'model' : self._model,
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
        n = self._model.count_states()
        shape = (len(time), n, n)
        if derivatives.shape != shape:
            raise ValueError('Wrong input: Expecting a derivatives array of'
                ' shape ' + str(shape) + '.')
        # Create datablock
        block = myokit.DataBlock2d(n, n, time)
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
        initial state.
        """
        n = len(self._state)
        return np.array(self._deriv, copy=True).reshape((n, n))
    def reset(self):
        """
        Resets the simulation:

        - The time variable is set to 0
        - The state is set back to the default state

        """
        # Reset time
        self._time = 0
        # Reset state
        self._state = list(self._default_state)
        # Reset derivatives
        n = len(self._state)
        self._deriv = [0.0] * n**2
        for i in xrange(n):
            self._deriv[i * (n + 1)] = 1.0
    def run(self, duration, log=None, log_interval=5, progress=None,
            msg='Running ICSimulation'):
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
        the second axis is a state x and the third is a state y such that the
        point ``(t, x, y)`` represents ``dx/dy(0)`` at time t. For example, if
        ``p`` is the array of derivatives, to get the derivative of state 0
        with respect to the initial value of state 1, use ``p[:,0,1]``.
        
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
        # Create empty list for derivative lists
        derivs = []
        # Get progress indication function (if any)
        if progress is None:
            progress = myokit._Simulation_progress
        if progress:
            if not isinstance(progress, myokit.ProgressReporter):
                raise ValueError('The argument "progress" must be either a'
                    ' subclass of myokit.ProgressReporter or None.')
        # Run simulation
        if duration > 0:
            # Initialize
            n = len(self._state)
            state = [0] * n
            deriv = [0] * (n ** 2)
            self._sim.sim_init(
                tmin,
                tmax,
                self._dt,
                list(self._state),
                list(self._deriv),
                state,
                deriv,
                self._protocol,
                log,
                derivs,
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
                    while t < tmax:
                        t = self._sim.sim_step()
            finally:
                # Clean even after KeyboardInterrupt or other Exception
                self._sim.sim_clean()
            # Update internal state
            self._state = list(state)
            self._deriv = list(deriv)
            self._time += duration
            # Convert derivatives to numpy arrays
            # Using
            #   derivs = [np.array(x).reshape(n,n) for x in derivs]
            # will create a list of views of arrays
            # to avoid the overhead of views, perhaps it's better to copy this
            # view into a new array explicitly?
            derivs = np.array([
                np.array(np.array(x).reshape(n,n), copy=True) for x in derivs])
        # Return
        return log, derivs
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
