#
# Cable simulation using a simple forward Euler implementation
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
# Location of source file
SOURCE_FILE = 'cable.c'
class Simulation1d(myokit.CModule):
    """
    Can run 1d cable simulations based on a :class:`model <Model>`.

    ``model``
        The model to simulate with. This model will be cloned when the
        simulation is created so that no changes to the given model will be
        made.
    ``protocol``
        An optional pacing protocol, used to stimulate a number of cells at the
        start of the cable.
    ``ncells``
        The number of cells in the cable

    This simulation provides the following inputs variables can bind to:
    
    ``time``
        The simulation time
    ``pace``
        The pacing level, this is set if a protocol was passed in.
    ``diffusion_current``
        The current flowing from the cell to its neighbours. This will be
        positive when the cell is acting as a source, negative when it is
        acting as a sink.
        
    The variable ``time`` is set globally, meaning each cell uses the same
    value. The variables ``pace`` and ``diffusion_current`` have different
    values per cell. The number of paced cell can be set with
    :meth:`set_paced_cells`.
        
    A single labeled variable is required for this simulation to work:
    
    ``membrane_potential``
        The variable representing the membrane potential.

    Simulations maintain an internal state consisting of

    - the current simulation time
    - the current state
    - the default state
      
    When a simulation is created, the simulation time is set to 0 and both the
    current and the default state are equal to the state of the given model,
    copied once for each cell.
    After each call to :meth:`run` the time variable and current state are
    updated, so that each successive call to run continues where the previous
    simulation left off. A :meth:`reset` method is provided that will set the 
    time back to 0 and revert the current state to the default state. To change
    the time or state manually, use :meth:`set_time` and :meth:`set_state`.

    A pre-pacing method :meth:`pre` is provided that doesn't affect the
    simulation time but will update the current *and the default state*. This
    allows you to pre-pace, run a simulation, reset to the pre-paced state, run
    another simulation etc.
    
    The ``diffusion_current`` is calculated as::
    
        i = sum[g * (V - V_j)]
        
    Where the sum is taken over all neighbouring cells j.
    """
    _index = 0 # Cable simulation id
    def __init__(self, model, protocol=None, ncells=50):
        super(Simulation1d, self).__init__()
        # Require a valid model
        model.validate()
        # Clone model, store
        model = model.clone()
        self._model = model
        # Set protocol
        self.set_protocol(protocol)        
        # Set number of cells
        ncells = int(ncells)
        if ncells < 1:
            raise ValueError('The number of cells must be at least 1.')
        self._ncells = ncells
        # Set number of cells paced
        self.set_paced_cells()
        # Set conductance
        self.set_conductance()
        # Set step size
        self.set_step_size()
        # Set remaining properties
        self._time = 0
        self._nstate = self._model.count_states()
        # Get membrane potential variable
        self._vm = model.label('membrane_potential')
        if self._vm is None:
            raise Exception('This simulation requires the membrane potential'
                ' variable to be labelled as "membrane_potential".')
        # Check for binding to diffusion_current
        if model.binding('diffusion_current') is None:
            raise Exception('This simulation requires a variable to be bound'
                ' to "diffusion_current" to pass current from one cell to the'
                ' next')
        # Set state and default state
        self._state = self._model.state() * ncells
        self._default_state = list(self._state)
        # Unique simulation id
        Simulation1d._index += 1
        module_name = 'myokit_sim1d_'+str(Simulation1d._index)
        # Arguments
        args = {
            'module_name' : module_name,
            'model' : self._model,
            'vmvar' : self._vm,
            'ncells' : self._ncells,
            }
        fname = os.path.join(myokit.DIR_CFUNC, SOURCE_FILE)
        # Debug
        if myokit.DEBUG:
            print(self._code(fname, args,
                line_numbers=myokit.DEBUG_LINE_NUMBERS))
            import sys
            sys.exit(1)
        # Create simulation
        libd = None
        incd = [myokit.DIR_CFUNC]
        self._sim = self._compile(module_name, fname, args, ['m'], libd, incd)
    def pre(self, duration, progress=None, msg='Pre-pacing Simulation1d'):
        """
        This method can be used to perform an unlogged simulation, typically to
        pre-pace to a (semi-)stable orbit.

        After running this method

        - The simulation time is **not** affected
        - The current state and the default state are updated to the final
          state reached in the simulation.

        Calls to :meth:`reset` after using :meth:`pre` will revert the
        simulation to this new default state.

        To obtain feedback on the simulation progress, an object implementing
        the :class:`myokit.ProgressReporter` interface can be passed in.
        passed in as ``progress``. An optional description of the current
        simulation to use in the ProgressReporter can be passed in as `msg`.
        """
        self._run(duration, myokit.LOG_NONE, 1, progress, msg)
        self._default_state = list(self._state)
    def reset(self):
        """
        Resets the simulation:

        - The time variable is set to 0
        - The state is set to the default state

        """
        self._time = 0
        self._state = list(self._default_state)
    def run(self, duration, log=None, log_interval=1.0, progress=None,
            msg='Running Simulation1d'):
        """
        Runs a simulation and returns the logged results. Running a simulation
        has the following effects:

        - The internal state is updated to the last state in the simulation.
        - The simulation's time variable is updated to reflect the time
          elapsed during the simulation.

        The number of time units to simulate can be set with ``duration``.

        The variables to log can be indicated using the ``log`` argument. There
        are several options for its value:

        - ``None`` (default), to log all states
        - An integer flag or a combination of flags. Options: 
          ``myokit.LOG_NONE``, ``myokit.LOG_STATE``, ``myokit.LOG_INTER``,
          ``myokit.LOG_BOUND``, ``myokit.LOG_ALL``.
        - A list of qnames or variable objects
        - A :class:`myokit.DataLog` object.
           
        For more details on the ``log`` argument, see the function
        :meth:`myokit.prepare_log`.

        Any variables bound to "time" will be logged globally, all others will
        be logged per cell. These variables will be prefixed with a number
        indicating the cell index. For example, when using::
        
            s = Simulation1d(m, p, ncells=3)
            d = s.run(1000, log=['engine.time', 'membrane.V']
            
        where <engine.time> is bound to "time" and <membrane.V> is the membrane
        potential variable, the resulting log will contain the following
        variables::
        
            {
                'engine.time'  : [...],
                '0.membrane.V' : [...],
                '1.membrane.V' : [...],
                '2.membrane.V' : [...],
            }
            
        A log entry is created every time *at least* ``log_interval`` time
        units have passed.
            
        To obtain feedback on the simulation progress, an object implementing
        the :class:`myokit.ProgressReporter` interface can be passed in.
        passed in as ``progress``. An optional description of the current
        simulation to use in the ProgressReporter can be passed in as `msg`.
        """
        r = self._run(duration, log, log_interval, progress, msg)
        self._time += duration
        return r
    def _run(self, duration, log, log_interval, progress, msg):
        # Simulation times
        if duration < 0:
            raise Exception('Simulation time can\'t be negative.')
        tmin = self._time
        tmax = tmin + duration
        # Gather global variables in model
        g = [self._model.time().qname()]
        # Parse log argument
        log = myokit.prepare_log(
            log,
            self._model,
            dims=(self._ncells,),
            global_vars=g,
            if_empty=myokit.LOG_STATE+myokit.LOG_BOUND,
            allowed_classes=myokit.LOG_STATE+myokit.LOG_BOUND+myokit.LOG_INTER,
            )
        # Get event tuples
        # Logging period
        log_interval = 0 if log_interval is None else float(log_interval)
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
            state_in = self._state
            state_out = list(state_in)
            self._sim.sim_init(
                self._ncells,
                self._conductance,
                tmin,
                tmax,
                self._step_size,
                state_in,
                state_out,
                self._protocol,
                min(self._npaced, self._ncells),
                log,
                log_interval)
            t = tmin
            try:
                if progress:
                    # Loop with feedback
                    with progress.job(msg):
                        r = 1.0 / duration if duration != 0 else 1
                        while t < tmax:
                            t = self._sim.sim_step()
                            if not progress.update(min((t - tmin) * r, 1)):
                                raise myokit.SimulationCancelledError()
                else:
                    # Loop without feedback
                    while t < tmax:
                        t = self._sim.sim_step()
            finally:
                # Clean even after keyboardinterrupt or exception
                self._sim.sim_clean()
            # Update state
            self._state = state_out
        # Return log
        return log
    def _set_state(self, state, icell, update):
        """
        Handles set_state and set_default_state.
        """
        n = len(state)  
        if n == self._nstate:
            if icell is None:
                # State might not be a list, at this point...
                return list(state) * self._ncells
            else:
                icell = int(icell)
                if icell < 0 or icell >= self._ncells:
                    raise KeyError('Given cell index out of range.')
                offset = icell * self._nstate
                update[offset : offset + self._nstate] = state
                return update
        elif n == self._nstate * self._ncells:
            return list(state)
    def set_conductance(self, g=10):
        """
        Changes the cell-to-cell conductance.
        """
        g = float(g)
        if g < 0:
            raise ValueError('Conductance cannot be negative.')
        self._conductance = g
    def set_default_state(self, state, icell=None):
        """
        Changes this simulation's default state.
        
        This can be used in three different ways:
        
        1. When called with an argument ``state`` of size ``n_states`` and
           ``i_cell=None`` the given state will be set as the new state of all
           cells in the simulation.
        2. Called with an argument ``state`` of size n_states and
           ``i_cell`` equal to a valid cell index, this method will update only
           the selected cell's state.
        3. Finally, when called with a ``state`` of size ``n_states * ncells``
           the method will treat ``state`` as a concatenation of state vectors
           for each cell.
           
        """
        self._default_state = self._set_state(state, icell,self._default_state)
    def set_paced_cells(self, n=5):
        """
        Sets the number of cells that will receive a stimulus from the pacing
        protocol.
        """
        n = int(n)
        if n < 0:
            raise ValueError('The number of cells to stimulate cannot be'
                ' negative.')
        self._npaced = n
    def set_state(self, state, icell=None):
        """
        Changes the state of this simulation's model.
        
        This can be used in three different ways:
        
        1. When called with an argument ``state`` of size ``nstates`` and
           ``icell=None`` the given state will be set as the new state of all
           cells in the simulation.
        2. Called with an argument ``state`` of size ``nstates`` and
           ``icell`` equal to a valid cell index, this method will update only
           the selected cell's state.
        3. Finally, when called with a ``state`` of size ``nstates * ncells``
           the method will treat ``state`` as a concatenation of state vectors
           for each cell.
           
        """
        self._state = self._set_state(state, icell, self._state)
    def set_step_size(self, step_size=0.005):
        """
        Sets the solver step size. In some cases, the solver will take a
        slightly smaller step size, either to arrive exactly at the start/end
        of a pacing event or to arrive exactly at the end of a simulation.
        """
        step_size = float(step_size)
        if step_size <= 0:
            raise ValueError('Step size must be greater than zero.')
        self._step_size = step_size
    def set_protocol(self, protocol=None):
        """
        Changes the pacing protocol used by this simulation.
        """
        if protocol is None:
            self._protocol = None
        else:
            self._protocol = protocol.clone()
    def set_time(self, time=0):
        """
        Sets the current simulation time.
        """
        self._time = float(time)
    def state(self, icell=None):
        """
        Returns the current simulation state as a list of ``len(state) *
        ncells`` floating point values. If the optional argument ``icell`` is 
        set to a valid cell index only the state of that cell is returned.
        """
        if icell is None:
            return list(self._state)
        else:
            icell = int(icell)
            if icell < 0 or icell >= self._ncells:
                raise KeyError('Given cell index out of range.')
            offset = icell * self._nstate
            return self._state[offset : offset + self._nstate]
    def time(self):
        """
        Returns the current simulation time.
        """
        return self._time
