#
# OpenCL driven fiber-tissue simulation.
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
# Location of C and OpenCL sources
SOURCE_FILE = 'fiber_tissue.c'
KERNEL_FILE = 'openclsim.cl'
# OpenCLSim keywords
from openclsim import KEYWORDS
class FiberTissueSimulation(myokit.CModule):
    """
    Runs a simulation of a fiber leading up to a rectangular piece of tissue.

    Takes the following input arguments:
    
    ``fiber_model``
        The model to simulate the fiber with.
    ``tissue_model``
        The model to simulate the tissue with. Both models will be cloned when 
        the simulation is created so that no changes to the given models will
        be made.        
    ``protocol``
        An optional pacing protocol, used to stimulate a number of cells at the
        start of the fiber.
    ``ncells_fiber``
        The number of cells in the fiber (a tuple).
    ``ncells_tissue``
        The number of cells in the tissue (a tuple).
    ``nx_paced``
        The width (in cells) of the stimulus applied to the fiber. The fiber
        will be stimulated along its full height.
    ``join``
        A tuple ``(x,y)`` specifying the top-left coordinate on the tissue that
        the fiber connects to.
    ``g_fiber``
        The cell to cell conductance in the fiber (a tuple).
    ``g_tissue``
        The cell to cell conductance in the tissue (a tuple).
    ``g_fiber_tissue``
        The fiber-cell to tissue-cell conductance at the junction (a scalar).
    ``dt``
        The time step to use in the forward-Euler integration scheme.
    ``double_precision``
        Set this to True if your OpenCL device supports double precision. This
        will greatly reduce the chance of a divide-by-zero or other numerical
        error introducing NaNs into the simulation.
    
    The simulation provides the following inputs variables can bind to:
    
    ``time`` (global)
        The simulation time
    ``pace`` (per-cell)
        The pacing level, this is set if a protocol was passed in.
    ``diffusion_current`` (per-cell)
        The current flowing from the cell to its neighbours. This will be
        positive when the cell is acting as a source, negative when it is
        acting as a sink.
        
    The variable ``time`` is set globally, meaning each cell uses the same
    value. The variables ``pace`` and ``diffusion_current`` have different 
    values per cell.
        
    The following labeled variables are required for this simulation to work:
    
    ``membrane_potential``
        The variable representing the membrane potential.
        
    The ``diffusion_current`` is calculated as::
    
        i = gx * ((V - V_xnext) - (V_xlast - V))
          + gy * ((V - V_ynext) - (V_ylast - V))
        
    At the boundaries, where either ``V_ilast`` or ``V_inext`` is unavailable,
    the value of ``V`` is substituted, causing the term to go to zero. The
    values of ``gx`` and ``gy`` can be set in the simulation's constructor.
    
    For a typical model with currents in ``[uA/uF]`` and voltage in ``[mV]``, 
    `gx`` and ``gy`` have the unit ``[mS/uF]``.

    Simulations maintain an internal state consisting of

    - the current simulation time
    - the current states
    - the default states
      
    When a simulation is created, the simulation time is set to 0 and both the
    current and the default state are equal to the state of the given models,
    copied once for each cell.
    After each call to :meth:`run` the time variable and current state are
    updated, so that each successive call to run continues where the previous
    simulation left off. A :meth:`reset` method is provided that will set the 
    time back to 0 and revert the current state to the default state. To change
    the time or state manually, use :meth:`set_time` and
    :meth:`set_fiber_state` and :meth:`set_tissue_state`.

    A pre-pacing method :meth:`pre` is provided that doesn't affect the
    simulation time but will update the current *and the default state*. This
    allows you to pre-pace, run a simulation, reset to the pre-paced state, run
    another simulation etc.
    
    The model passed to the simulation is cloned and stored internally, so
    changes to the original model object will not affect the simulation.
    
    Models used with this simulation need to have independent components: it
    should be possible to evaluate the model's equations one component at a
    time. A model's suitability can be tested using :meth:`
    has_interdependent_components
    <myokit.Model.has_interdependent_components>`.
    """
    _index = 0 # Unique id for the created simulation module
    def __init__(self, fiber_model, tissue_model, protocol=None,
            ncells_fiber=(128,2), ncells_tissue=(128,128), nx_paced=5,
            g_fiber=(9, 6), g_tissue=(9, 6), g_fiber_tissue=9,
            dt=0.005, double_precision=False):
        super(FiberTissueSimulation, self).__init__()
        # List of globally logged inputs
        self._global = ['time', 'pace']
        # Require valid models
        fiber_model.validate()
        tissue_model.validate()
        # Require independent components
        if fiber_model.has_interdependent_components():
            cycles = fiber_model.component_cycles()
            cycles = '\n  '.join(
                [' > '.join([x.var().qname() for x in c])
                for c in cycles])
            raise Exception('This simulation requires models without'
                ' interdependent components. Please restructure the fiber'
                ' model and re-run. Cycles:\n' + cycles)
        if tissue_model.has_interdependent_components():
            cycles = tissue_model.component_cycles()
            cycles = '\n  '.join(
                [' > '.join([x.var().qname() for x in c])
                for c in cycles])
            raise Exception('This simulation requires models without'
                ' interdependent components. Please restructure the tissue'
                ' model and re-run. Cycles:\n' + cycles)
        # Clone models, store
        fiber_model = fiber_model.clone()
        tissue_model = tissue_model.clone()
        self._modelf = fiber_model
        self._modelt = tissue_model
        # Set protocol
        self.set_protocol(protocol)
        # Check dimensionality, number of cells
        msg = 'The fiber size must be a tuple (nx, ny).'
        try:
            if len(ncells_fiber) != 2:
                raise ValueError(msg)
        except TypeError:
            raise ValueError(msg)
        msg = 'The tissue size must be a tuple (nx, ny).'
        try:
            if len(ncells_tissue) != 2:
                raise ValueError(msg)
        except TypeError:
            raise ValueError(msg)
        self._ncellsf = [int(x) for x in ncells_fiber]
        self._ncellst = [int(x) for x in ncells_tissue]
        if self._ncellsf[0] < 1 or self._ncellsf[1] < 1:
            raise ValueError('The fiber must be at least (1,1).')
        if self._ncellst[0] < 1 or self._ncellst[1] < 1:
            raise ValueError('The tissue size must be at least (1,1).')
        if self._ncellsf[1] > self._ncellst[1]:
            raise ValueError('The fiber y-dimension cannot exceed that of the'
                ' tissue.')
        # Check width of pacing stimulus
        nx_paced = int(nx_paced)
        if nx_paced < 0:
            raise ValueError('The width of the stimulus pulse must be'
                ' non-negative.')
        nx_paced = min(nx_paced, self._ncellsf[0])
        self._paced_cells = []
        for y in xrange(self._ncellsf[1]):
            for x in xrange(min(self._ncellsf[0], nx_paced)):
                self._paced_cells.append(x + y * self._ncellsf[0])
        # Check conductivities
        if len(g_fiber) != 2:
            raise ValueError('The fiber conductivity must be a tuple (gx,gy).')
        if len(g_tissue) != 2:
            raise ValueError('The tissue conductivity must be a tuple (gx,gy).'
                )
        self._gf = [float(x) for x in g_fiber]
        self._gt = [float(x) for x in g_tissue]
        self._gft = float(g_fiber_tissue)
        # Point of connection to tissue
        self._cfx = self._ncellsf[0] - 1
        self._ctx = 0
        self._cty = int(0.5 * (self._ncellst[1] - self._ncellsf[1]))
        # Check step size
        dt = float(dt)
        if dt <= 0:
            raise ValueError('The step size must be greater than zero.')
        self._step_size = dt
        # Check precision, set native math flag
        self._precision = myokit.DOUBLE_PRECISION if double_precision else \
            myokit.SINGLE_PRECISION
        # Always use native maths
        self._native_math = True
        # Set remaining properties
        self._time = 0
        self._nstatef = self._modelf.count_states()
        self._nstatet = self._modelt.count_states()
        self._ntotalf = self._ncellsf[0] * self._ncellsf[1]
        self._ntotalt = self._ncellst[0] * self._ncellst[1]
        # Get membrane potential variables
        self._vmf = self._modelf.label('membrane_potential')
        if self._vmf is None:
            raise Exception('This simulation requires the membrane potential'
                ' variable to be labelled as "membrane_potential" in the fiber'
                ' model.')
        if not self._vmf.is_state():
            raise Exception('The variable labelled as membrane potential in'
                ' the fiber model must be a state variale.')
        self._vmt = self._modelt.label('membrane_potential')
        if self._vmt is None:
            raise Exception('This simulation requires the membrane potential'
                ' variable to be labelled as "membrane_potential" in the'
                ' tissue model.')
        if not self._vmt.is_state():
            raise Exception('The variable labelled as membrane potential in'
                ' the tissue model must be a state variale.')
        # Check for binding to diffusion_current
        if self._modelf.binding('diffusion_current') is None:
            raise Exception('This simulation requires a variable in the fiber'
                ' model to be bound to "diffusion_current" to pass current'
                ' from one cell to the next.')
        if self._modelt.binding('diffusion_current') is None:
            raise Exception('This simulation requires a variable in the tissue'
                ' model to be bound to "diffusion_current" to pass current'
                ' from one cell to the next.')
        # Check if membrane potentials have same unit
        uvf = self._vmf.unit()
        if uvf is None:
            raise Exception('The fiber model must specify a unit for the'
                ' membrane potential.')
        uvt = self._vmt.unit()
        if uvt is None:
            raise Exception('The tissue model must specify a unit for the'
                ' membrane potential.')
        if uvf != uvt:
            raise Exception('The membrane potential must have the same unit in'
                ' the fiber and the tissue model: ' + str(uvf) + ' vs '
                + str(uvt) + '.')
        # Check if diffusion current variables have same unit
        ucf = self._modelf.binding('diffusion_current').unit()
        uct = self._modelt.binding('diffusion_current').unit()
        if ucf is None:
            raise Exception('The fiber model must specify a unit for the'
                ' diffusion current.')
        if uct is None:
            raise Exception('The tissue model must specify a unit for the'
                ' diffusion current.')
        if ucf != uct:
            raise Exception('The diffusion current must have the same unit in'
                ' the fiber and the tissue model: ' + str(ucf) + ' vs '
                + str(uct) + '.')
        # Set state and default state
        self._statef = self._modelf.state() * self._ntotalf
        self._statet = self._modelt.state() * self._ntotalt
        self._default_statef = list(self._statef)
        self._default_statet = list(self._statet)
        # Process bindings, remove unsupported bindings, get map of bound
        # variables to internal names.
        self._bound_variablesf = self._modelf.prepare_bindings({
            'time' : 'time',
            'pace' : 'pace',
            'diffusion_current' : 'idiff',
            })
        self._bound_variablest = self._modelt.prepare_bindings({
            'time' : 'time',
            'pace' : 'pace',
            'diffusion_current' : 'idiff',
            })
        # Reserve keywords
        from myokit.formats import opencl
        self._modelf.reserve_unique_names(*opencl.keywords)
        self._modelt.reserve_unique_names(*opencl.keywords)
        self._modelf.reserve_unique_names(
            *['calc_' + c.name() for c in self._modelf.components()])
        self._modelt.reserve_unique_names(
            *['calc_' + c.name() for c in self._modelt.components()])
        self._modelf.reserve_unique_names(*KEYWORDS)
        self._modelt.reserve_unique_names(*KEYWORDS)
        self._modelf.create_unique_names()
        self._modelt.create_unique_names()
        # Create back-end
        self._create_backend()
    def _create_backend(self):
        """
        Creates this simulation's backend.
        """
        # Unique simulation id
        FiberTissueSimulation._index += 1
        mname = 'myokit_sim_fiber_tissue_' + str(FiberTissueSimulation._index)
        # Arguments
        args = {
            'module_name' : mname,
            'modelf' : self._modelf,
            'modelt' : self._modelt,
            'vmf' : self._vmf,
            'vmt' : self._vmt,
            'boundf' : self._bound_variablesf,
            'boundt' : self._bound_variablest,
            'precision' : self._precision,
            'native_math' : self._native_math,
            }
        fname = os.path.join(myokit.DIR_CFUNC, SOURCE_FILE)
        # Create simulation module
        if myokit.DEBUG:
            print(self._code(fname, args,
                line_numbers=myokit.DEBUG_LINE_NUMBERS))
            return
        libs = ['OpenCL']
        libd = list(myokit.OPENCL_LIB)
        incd = list(myokit.OPENCL_INC)
        incd.append(myokit.DIR_CFUNC)
        self._sim = self._compile(mname, fname, args, libs, libd, incd)
    def fiber_state(self, x=None):
        """
        Returns the current simulation state in the fiber as a list of
        ``len(state_fiber) * ncells_fiber`` floating point values.
        
        If the optional arguments ``x`` and ``y`` specify a valid cell index a
        single cell's state is returned.
        """
        if x is None:
            return list(self._statef)
        else:
            x = int(x)
            if x < 0 or x > self._ncellsf[0]:
                raise KeyError('Given x-index out of range.')
            y = int(y)
            if y < 0 or y > self._ncellsf[1]:
                raise KeyError('Given y-index out of range.')
            x += y * self._ncellsf[0] 
            return self._statef[x * self._nstatef : (x + 1) * self._nstatef]
    def find_nan(self, logf, logt):
        """
        Searches for the origin of a ``NaN`` (or ``inf``) in a set of
        simulation logs generated by this Simulation.
        
        The logs must contain the state of each cell and all bound variables.
        The NaN can occur at any point in time except the first.
        
        Returns a tuple ``(part, time, icell, variable, value, states, bound)``
        where ``time`` is the time the first ``NaN`` was found and ``icell`` is
        the index of the cell in which it happened. The entry ``part`` is a
        string containing either "fiber" or "tissue", indicating which part of
        the simulation triggered the error. The offending variable's name is 
        given as ``variable`` and its (illegal) value as ``value``. The current
        state and, if available, any previous states are given in the list
        ``states``. Here, ``states[0]`` points to the current state in the
        simulation part causing the error, ``state[1]`` is the previous state
        and so on. Similarly the values of the error causing model's bound
        variables is given in ``bound``.
        """
        import numpy as np
        # Check if logs contain all states and bound variables
        lt = []
        lf = []
        for label in self._global:
            v = self._modelf.binding(label)
            if v is not None:
                lf.append(v.qname())
            v = self._modelt.binding(label)
            if v is not None:
                lt.append(v.qname())
        lf = myokit.prepare_log(
                myokit.LOG_STATE+myokit.LOG_BOUND,
                self._modelf,
                dims=self._ncellsf,
                global_vars=lf)
        lt = myokit.prepare_log(
                myokit.LOG_STATE+myokit.LOG_BOUND,
                self._modelt,
                dims=self._ncellst,
                global_vars=lt)
        for key in lf:
            if key not in logf:
                raise myokit.FindNanError('Method requires a simulation log'
                    ' from the fiber model containing all states and bound'
                    ' variables. Missing variable <' + key + '>.')
        for key in lt:
            if key not in logt:
                raise myokit.FindNanError('Method requires a simulation log'
                    ' from the tissue model containing all states and bound'
                    ' variables. Missing at least variable <' + key + '>.')
        del(lf)
        del(lt)
        # Error criterium: NaN/inf detection
        def bisect(ar, lo, hi):
            if not np.isfinite(ar[lo]):
                return lo
            md = lo + int(np.ceil(0.5 * (hi - lo)))
            if md == hi:
                return hi
            if not np.isfinite(ar[md]):
                return bisect(ar, lo, md)
            else:
                return bisect(ar, md, hi)
        # Search for first occurrence of propagating NaN in the log
        def find_error_position(log):
            ifirst = None   # Log list index
            kfirst = None   # Log key
            for key, ar in log.iteritems():
                if ifirst is None:
                    if not np.isfinite(ar[-1]):
                        # First NaN found
                        kfirst = key
                        ifirst = bisect(ar, 0, len(ar)-1)
                        if ifirst == 0: break                            
                elif not np.isfinite(ar[ifirst - 1]):
                        # Earlier NaN found
                        kfirst = key
                        ifirst = bisect(ar, 0, ifirst)
                        if ifirst == 0: break
            return ifirst, kfirst
        # Get the name of a time variable in the fiber model
        time_varf = self._modelf.time().qname()
        # Deep searching function
        def relog(_logf, _logt, _dt):
            # Get first occurence of error
            ifirstf, kfirstf = find_error_position(_logf)
            ifirstt, kfirstt = find_error_position(_logt)
            if kfirstf is None and kfirstt is None:
                raise myokit.FindNanError('Error condition not found in logs.')
            elif kfirstf is None:
                ifirst = ifirstt
            elif kfirstt is None:
                ifirst = ifirstf
            elif ifirstf == 0 or ifirstt == 0:
                raise myokit.FindNanError('Unable to work with simulation logs'
                    ' where the error condition is met in the very first data'
                    ' point.')
            else:
                ifirst = min(ifirstf, ifirstt)
            # Position to start deep search at
            istart = ifirst - 1
            # Get last logged states before error
            statef = []
            statet = []
            for dims in myokit.dimco(*self._ncellsf):
                pre = '.'.join([str(x) for x in dims]) + '.'
                for s in self._modelf.states():
                    statef.append(_logf[pre + s.qname()][istart])
            for dims in myokit.dimco(*self._ncellst):
                pre = '.'.join([str(x) for x in dims]) + '.'
                for s in self._modelt.states():
                    statet.append(_logt[pre + s.qname()][istart])
            # Get last time before error            
            time = _logf[time_varf][istart]
            # Save current state & time
            old_statef = self._statef
            old_statet = self._statet
            old_time = self._time
            self._statef = statef
            self._statet = statet
            self._time = time
            # Run until next time point, log every step
            duration = _logf[time_varf][ifirst] - time
            log = myokit.LOG_BOUND + myokit.LOG_STATE
            _logf, _logt = self.run(duration, logf=log, logt=log,
                log_interval=_dt, report_nan=False)
            # Reset simulation to original state
            self._statef = old_statef
            self._statet = old_statet
            self._time = old_time
            # Return new logs        
            return _logf, _logt
        # Get time step
        dt = logf[time_varf][1] - logf[time_varf][0]
        # Search with successively fine log interval
        while dt > 0:
            dt *= 0.1
            if dt < 0.5: dt = 0
            logf, logt = relog(logf, logt, dt)
        # Search for first occurrence of error in the detailed log
        ifirstf, kfirstf = find_error_position(logf)
        ifirstt, kfirstt = find_error_position(logt)
        if kfirstt is None or (kfirstf is not None and kfirstf < kfirstt):
            part = 'fiber'
            ifirst = ifirstf
            kfirst = kfirstf
            model = self._modelf
            log = logf
        else:
            part = 'tissue'
            ifirst = ifirstt
            kfirst = kfirstt
            model = self._modelt
            log = logt
        # Get indices of cell in state vector
        icell = [int(x) for x in kfirst.split('.')[0:2]]
        nstate = model.count_states()
        istate = icell*nstate
        # Get state & bound before, during and after error
        def state(index, icell):
            s = []
            b = {}
            for var in model.states():
                s.append(log[var.qname(), icell][index])
            for var in model.variables(bound=True):
                if var.binding() in self._global:
                    b[var.qname()] = log[var.qname()][index]
                else:
                    b[var.qname()] = log[var.qname(), icell][index]
            return s, b
        # Get error cell's states before, during and after
        #states = state(ifirst-1, ifirst, ifirst+1)
        states = []
        bound = []
        max_states = 3
        for k in xrange(ifirst, ifirst - max_states - 1, -1):
            if k < 0: break
            s, b = state(k, icell)
            states.append(s)
            bound.append(b)
        # Get variable causing error
        var = model.get('.'.join(kfirst.split('.')[2:]))
        # Get value causing error
        value = states[1][var.indice()]
        var = var.qname()
        # Get time error occurred
        time = logf[time_varf][ifirst]
        # Return part, time, icell, variable, value, states, bound
        return part, time, icell, var, value, states, bound
    def pre(self, duration, report_nan=True, progress=None,
            msg='Pre-pacing FiberTissueSimulation'):
        """
        This method can be used to perform an unlogged simulation, typically to
        pre-pace to a (semi-)stable orbit.

        After running this method

        - The simulation time is **not** affected
        - The current state and the default state are updated to the final
          state reached in the simulation.

        simulation to this new default state.

        Calls to :meth:`reset` after using :meth:`pre` will revert the
        To obtain feedback on the simulation progress, an object implementing
        the :class:`myokit.ProgressReporter` interface can be passed in.
        passed in as ``progress``. An optional description of the current
        simulation to use in the ProgressReporter can be passed in as `msg`.
        """
        self._run(duration, myokit.LOG_NONE, myokit.LOG_NONE, 1, report_nan,
            progress, msg)
        self._default_statef = list(self._statef)
        self._default_statet = list(self._statet)
    def reset(self):
        """
        Resets the simulation:

        - The time variable is set to 0
        - The current state is set to the default state (either the model's
          initial state or the last state reached using :meth:`pre`)

        """
        self._time = 0
        self._statef = list(self._default_statef)
        self._statet = list(self._default_statet)
    def run(self, duration, logf=None, logt=None, log_interval=1.0,
            report_nan=True, progress=None,
            msg='Running FiberTissueSimulation'):
        """
        Runs a simulation and returns the logged results as a tuple containing
        two :class:`myokit.DataLog` objects.. Running a simulation has
        the following effects:

        - The internal state is updated to the last state in the simulation.
        - The simulation's time variable is updated to reflect the time
          elapsed during the simulation.

        The number of time units to simulate can be set with ``duration``.

        The variables to log can be indicated using the arguments ``logf`` and
        ``logt``. There are several options for their values:

        - ``None`` (default), to log all states
        - An integer flag or a combination of flags. Options: 
          ``myokit.LOG_NONE``, ``myokit.LOG_STATE``, ``myokit.LOG_BOUND``,
          ``myokit.LOG_INTER``, ``myokit.LOG_ALL``.
        - A list of qnames or variable objects
        - A :class:`myokit.DataLog`. In this case, new data will be appended
          to the existing log.
           
        For more details on the log arguments, see the function
        :meth:`myokit.prepare_log`.

        Any variables bound to "time" or "pace" will be logged globally, all
        others will be logged per cell. These variables will be prefixed with a
        single number indicating the cell index in the fiber, and with two
        numbers indicating the index in the tissue.
        
        A log entry will be made every time *at least* ``log_interval`` time
        units have passed. No guarantee is given about the exact time log
        entries will be made, but the value of any logged time variable is
        guaranteed to be accurate.
        
        Intermediary variables can be logged, but with one small drawback: for
        performance reasons the logged values of states and bound variables
        will always be one time step ``dt`` ahead of the intermediary
        variables. For example if running the simulation with a step size
        ``dt=0.001`` the entry for a current ``IKr`` stored at ``t=1`` will be
        ``IKr(0.999)``, while the entry for state ``V`` will be ``V(1)``. If
        exact intermediary variables are needed it's best to log only states
        and bound variables and re-calculate the intermediary variables from
        these manually.
        
        To obtain feedback on the simulation progress, an object implementing
        the :class:`myokit.ProgressReporter` interface can be passed in.
        passed in as ``progress``. An optional description of the current
        simulation to use in the ProgressReporter can be passed in as `msg`.
        """
        r = self._run(duration, logf, logt, log_interval, report_nan, progress,
            msg)
        self._time += duration
        return r
    def _run(self, duration, logf, logt, log_interval, report_nan, progress,
            msg):
        # Simulation times
        if duration < 0:
            raise Exception('Simulation duration can\'t be negative.')
        tmin = self._time
        tmax = tmin + duration
        # Gather global variables in fiber model
        gf = []
        gt = []
        for label in self._global:
            v = self._modelf.binding(label)
            if v is not None:
                gf.append(v.qname())
            v = self._modelt.binding(label)
            if v is not None:
                gt.append(v.qname())
        # Parse logf argument
        logf = myokit.prepare_log(
            logf,
            self._modelf,
            dims=self._ncellsf,
            global_vars=gf,
            if_empty=myokit.LOG_STATE+myokit.LOG_BOUND,
            allowed_classes=myokit.LOG_STATE+myokit.LOG_BOUND+myokit.LOG_INTER,
            precision=self._precision)
        # Parse logt argument
        logt = myokit.prepare_log(
            logt,
            self._modelt,
            dims=self._ncellst,
            global_vars=gt,
            if_empty=myokit.LOG_STATE+myokit.LOG_BOUND,
            allowed_classes=myokit.LOG_STATE+myokit.LOG_BOUND+myokit.LOG_INTER,
            precision=self._precision)
        # Create list of intermediary fiber variables that need to be logged
        inter_logf = []
        vars_checked = set()
        for var in logf.iterkeys():
            var = myokit.split_key(var)[1]
            if var in vars_checked:
                continue
            vars_checked.add(var)
            var = self._modelf.get(var)
            if var.is_intermediary() and not var.is_bound():
                inter_logf.append(var)
        # Create list of intermediary tissue variables that need to be logged
        inter_logt = []
        vars_checked = set()
        for var in logt.iterkeys():
            var = myokit.split_key(var)[1]
            if var in vars_checked:
                continue
            vars_checked.add(var)
            var = self._modelt.get(var)
            if var.is_intermediary() and not var.is_bound():
                inter_logt.append(var)
        # Get preferred platform/device combo from configuration file
        platform, device = myokit.OpenCL.load_selection()
        # Generate kernels
        kernel_file = os.path.join(myokit.DIR_CFUNC, KERNEL_FILE)
        args = {
            'precision' : self._precision,
            'native_math' : self._native_math,
            'diffusion' : True,
            'fields' : [],
            }
        args['model'] = self._modelf
        args['vmvar'] = self._vmf
        args['bound_variables'] = self._bound_variablesf
        args['inter_log'] = inter_logf
        args['paced_cells'] = self._paced_cells
        if myokit.DEBUG:
            print('-'*79)
            print(self._code(kernel_file, args,
                line_numbers=myokit.DEBUG_LINE_NUMBERS))
        else:
            kernelf = self._export(kernel_file, args)
        args['model'] = self._modelt
        args['vmvar'] = self._vmt
        args['bound_variables'] = self._bound_variablest
        args['inter_log'] = inter_logt
        args['paced_cells'] = []
        if myokit.DEBUG:
            print('-'*79)
            print(self._code(kernel_file, args,
                line_numbers=myokit.DEBUG_LINE_NUMBERS))
            import sys
            sys.exit(1)
        else:
            kernelt = self._export(kernel_file, args)
        # Logging period (0 = disabled)
        log_interval = 1e-9 if log_interval is None else float(log_interval)
        if log_interval <= 0:
            log_interval = 1e-9
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
            state_inf = self._statef
            state_int = self._statet
            state_outf = list(state_inf)
            state_outt = list(state_int)
            self._sim.sim_init(
                platform,
                device,
                kernelf,
                kernelt,
                self._ncellsf[0],
                self._ncellsf[1],
                self._ncellst[0],
                self._ncellst[1],
                self._vmf.indice(),
                self._vmt.indice(),
                self._gf[0],
                self._gf[1],
                self._gt[0],
                self._gt[1],
                self._gft,
                self._cfx,
                self._ctx,
                self._cty,
                tmin,
                tmax,
                self._step_size,
                state_inf,
                state_int,
                state_outf,
                state_outt,
                self._protocol,
                logf,
                logt,
                log_interval,
                [x.qname() for x in inter_logf],
                [x.qname() for x in inter_logt],
                )
            try:
                t = tmin
                if progress:
                    # Loop with feedback
                    with progress.job(msg):
                        r = 1.0 / duration if duration != 0 else 1
                        while t < tmax:
                            t = self._sim.sim_step()
                            if t < tmin:
                                # A numerical error has occurred.
                                break
                            if not progress.update(min((t - tmin) * r, 1)):
                                raise myokit.SimulationCancelledError()
                else:
                    # Loop without feedback
                    while t < tmax:
                        t = self._sim.sim_step()
                        if t < tmin:
                            # A numerical error has occurred.
                            break
            finally:
                # Clean even after KeyboardInterrupt or other Exception
                self._sim.sim_clean()
            # Update states
            self._statef = state_outf
            self._statet = state_outt
        # Check for NaN's, print error output
        if report_nan and (logf.has_nan() or logt.has_nan()):
            txt =[ 'Numerical error found in simulation logs.']
            try:
                # NaN encountered, show how it happened
                part, time, icell, var, value, states, bound = self.find_nan(
                    logf, logt)
                model = self._modelt if part == 'tissue' else self._modelf
                txt.append('Encountered numerical error in ' + part
                    + ' simulation at t=' + str(time) + ' in cell (' 
                    + ','.join([str(x) for x in icell]) + ') when ' + var
                    + '=' + str(value) + '.')
                n_states = len(states)
                txt.append('Obtained ' + str(n_states) + ' previous state(s).')
                if n_states > 1:
                    txt.append('State before:')
                    txt.append(model.format_state(states[1]))
                txt.append('State during:')
                txt.append(model.format_state(states[0]))
                if n_states > 1:
                    txt.append('Evaluating derivatives at state before...')
                    try:
                        derivs = model.eval_state_derivatives(states[1],
                            precision=self._precision)
                        txt.append(model.format_state_derivs(states[1],
                            derivs))
                    except myokit.NumericalError as ee:
                        txt.append(ee.message)
            except myokit.FindNanError as e:
                txt.append('Unable to pinpoint source of NaN, an error'
                    ' occurred:')
                txt.append(e.message)
            raise myokit.SimulationError('\n'.join(txt))
        # Return logs
        return logf, logt
    def _set_statef(self, state, x, y, update):
        """
        Handles set_state and set_default_state for the tissue model.
        """
        n = len(state)
        ntotalf = self._ncellsf[0] * self._ncellsf[1]
        if n == self._nstatef * ntotalf:
            return list(state)
        elif n != self._nstatef:
            raise ValueError('Given state must have the same size as a'
                ' single cell state or a full simulation state')
        if x is None:
            return list(state) * ntotalf
        # Set specific cell state
        x = int(x)
        if x < 0 or x > self._ncellsf[0]:
            raise KeyError('Given x-index out of range.')
        y = int(y)
        if y < 0 or y > self._ncellsf[1]:
            raise KeyError('Given y-index out of range.')
        offset = (x + y * self._ncellsf[0]) * self._nstatef
        update[offset : offset + self._nstatef] = state
        return update
    def _set_statet(self, state, x, y, update):
        """
        Handles set_state and set_default_state for the tissue model.
        """
        n = len(state)
        ntotalt = self._ncellst[0] * self._ncellst[1]
        if n == self._nstatet * ntotalt:
            return list(state)
        elif n != self._nstatet:
            raise ValueError('Given state must have the same size as a single'
                ' cell state or a full simulation state')
        if x is None:
            return list(state) * ntotalt
        # Set specific cell state
        x = int(x)
        if x < 0 or x > self._ncellst[0]:
            raise KeyError('Given x-index out of range.')
        y = int(y)
        if y < 0 or y > self._ncellst[1]:
            raise KeyError('Given y-index out of range.')
        offset = (x + y * self._ncellst[0]) * self._nstatet
        update[offset : offset + self._nstatet] = state
        return update        
    def set_default_fiber_state(self, state, x=None, y=None):
        """
        Changes this simulation's default state for the fiber model.
        
        This can be used in three different ways:
        
        1. When called with an argument ``state`` of size ``n_states`` and
           ``x=y=None`` the given state will be set as the new default state of
           all fiber cells in the simulation.
        2. Called with an argument ``state`` of size n_states and
           ``x, y`` equal to a valid cell index, this method will update only
           the selected fiber cell's default state.
        3. Finally, when called with a ``state`` of size ``n_states * n_cells``
           the method will treat ``state`` as a concatenation of state vectors
           for each fiber cell.
           
        """
        self._default_statef = self._set_statef(state,x,y,self._default_statef)
    def set_default_tissue_state(self, state, x=None, y=None):
        """
        Changes this simulation's default state for the tissue model.
        
        This can be used in three different ways:
        
        1. When called with an argument ``state`` of size ``n_states`` and
           ``x=None`` the given state will be set as the new default state of
           tissue cells in the simulation.
        2. Called with an argument ``state`` of size n_states and
           ``x, y`` equal to a valid cell index, this method will update only
           the selected tissue cell's default state.
        3. Finally, when called with a ``state`` of size ``n_states * n_cells``
           the method will treat ``state`` as a concatenation of state vectors
           for each tissue cell.
           
        """
        self._default_statet = self._set_statet(state,x,y,self._default_statet)
    def set_fiber_state(self, state, x=None, y=None):
        """
        Changes the state of this simulation's fiber model.
        
        This can be used in three different ways:
        
        1. When called with an argument ``state`` of size ``n_states`` and
           ``x=y=None`` the given state will be set as the new state of all
           fiber cells in the simulation.
        2. Called with an argument ``state`` of size n_states and
           ``x, y`` equal to a valid cell index, this method will update only
           the selected fiber cell's state.
        3. Finally, when called with a ``state`` of size ``n_states * n_cells``
           the method will treat ``state`` as a concatenation of state vectors
           for each fiber cell.
           
        """
        self._statef = self._set_statef(state, x, y, self._statef)
    def set_tissue_state(self, state, x=None, y=None):
        """
        Changes the state of this simulation's tissue model.
        
        This can be used in three different ways:
        
        1. When called with an argument ``state`` of size ``n_states`` and
           ``x=y=None`` the given state will be set as the new state of all
           tissue cells in the simulation.
        2. Called with an argument ``state`` of size n_states and
           ``x, y`` equal to a valid cell index, this method will update only
           the selected tissue cell's state.
        3. Finally, when called with a ``state`` of size ``n_states * n_cells``
           the method will treat ``state`` as a concatenation of state vectors
           for each tissue cell.
           
        """
        self._statet = self._set_statet(state, x, y, self._statet)
    def set_step_size(self, step_size=0.005):
        """
        Sets the solver step size.
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
    def tissue_state(self, x=None):
        """
        Returns the current simulation state in the tissue as a list of
        ``len(state_tissue) * ncells_tissue`` floating point values.
        
        If the optional arguments ``x`` and ``y`` specify a valid cell index a
        single cell's state is returned.
        """
        if x is None:
            return list(self._statet)
        else:
            x = int(x)
            if x < 0 or x > self._ncellst[0]:
                raise KeyError('Given x-index out of range.')
            y = int(y)
            if y < 0 or y > self._ncellst[1]:
                raise KeyError('Given y-index out of range.')
            x += y * self._ncellst[0] 
            return self._statet[x * self._nstatet : (x + 1) * self._nstatet]
    def time(self):
        """
        Returns the current simulation time.
        """
        return self._time
