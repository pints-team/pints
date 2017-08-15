#
# Can run common experiments
#
# Some functions in this module require a recent version of scipy (i.e. the
# method scipy.optimize.curve_fit must exist).
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import numpy as np
import myokit
class StepProtocol(object):
    """
    An abstract base class for step protocol experiments.
    """
    def __init__(self, model, var, vvar=None):
        # Clone model
        self._model = model.clone()
        # Get variable names
        self._vars = []
        if type(var) in [str, unicode] or isinstance(var, myokit.Variable):
            var = [var]
        for v in var:
            if isinstance(v, myokit.Variable):
                v = v.qname()
            self._vars.append(self._model.get(v).qname())
        # Turn the membrane potential into a constant
        if vvar is None:
            vvar = self._model.label('membrane_potential')
            if vvar is None:
                raise ValueError('Membrane potential variable must be given as'
                    ' the argument "vvar" or labeled in the model as'
                    ' "membrane_potential".')
        else:
            if isinstance(vvar, myokit.Variable):
                vvar = vvar.qname()
            vvar = self._model.get(vvar)
        if vvar.is_state():
            vvar.demote()
        vvar.set_binding(None)
        vvar.set_rhs(0)
        # Get membrane potential name
        self._vvar = vvar.qname()
        # Get time variable name
        self._tvar = self._model.time().qname()
        # Start without any conversion factor
        self._factor = None
        # Maximum step size during logged simulation or None
        self._max_step_size = None
        # No simulation data yet
        self._logs = None
        # Holding potential & step potentials
        self.set_holding_potential()
        self.set_step_potential()
    def convert_g2i(self, vrev=60, gmax=1):
        """
        Converts any data logged during this experiment from conductances to
        currents by multiplying each step's data by::
        
            gmax * (v - vrev)
            
        This can be useful to obtain the original current traces when a
        conductance value was logged or when creating an IV protocol.
        
        Calling this method will override any previously set conversion factor.
        """
        self._factor = float(gmax) * (self._steps - float(vrev))
    def convert_i2g(self, vrev=60, gmax=1):
        """
        Converts any data logged during this experiment from currents to
        conductance by multiplying each step's data by::
        
            1 / (gmax * (v - vrev))
            
        This can be useful to obtain an activation or inactivation curve from
        logged current data. However, since this leads to numerical issues
        around ``v == vrev`` its better to run these experiments directly on a
        conductance variable.
        
        Calling this method will override any previously set conversion factor.
        """
        self._factor = 1.0 / (float(gmax) * (self._steps - float(vrev)))
    def disable_conversion(self):
        """
        Disables any previously set conversion factor.
        """
        self._factor = None
    def fit_boltzmann(self, var=None):
        """
        Attempts to fit a Boltzmann curve to a voltage-peaks dataset.
        
        The variable whose peaks to use can be specified as ``var``. If no
        variable is given the first value specified in the constructor is used.
        The fitted curve is given by::
        
            g = gmin + (gmax - gmin) / (1 + np.exp((v - v_half) / k))
        
        Note: This method requires a scipy installation providing the method
        scipy.optimize.curve_fit
        """
        from scipy.optimize import curve_fit
        d = self.peaks(normalize=True).npview()
        v = d[self._vvar]
        g = d[var] if var else d[self._vars[0]]
        o = np.ones(v.shape)
        gmin = np.min(g)
        gmax = np.max(g)
        def f(v, v_half, k):
            return np.select(
                [v == v_half],
                [o * k],
                gmin + (gmax - gmin) / (1.0 + np.exp((v - v_half) / k)))
        vmid = v[-1] - v[0]
        slope = -5 if g[0] < g[-1] else 5
        p = curve_fit(f, v, g, [vmid, slope])
        return p[0]
    def peaks(self, normalize=False):
        """
        Returns a :class:`myokit.DataLog` containing the tested step
        voltages and the peak values of the logged variable(s).
        
        The names used in the simulation log correspond to those used in the
        model. For example, when doing an experiment on a Sodium channel the
        simulation log might have entries ``membrane.v`` and ``ina.g`` where
        ``membrane.v`` contains the used voltage steps while ``ina.g`` contains
        the peak values measured at those voltages.
        
        If ``normalize`` is set to ``True``, the peak data returned will be
        normalized by dividing all values by the largest (most positive) peak
        in the list. If no positive, non-zero values are found no normalization
        will be applied.
        
        If any conversion factor was specified the data will be converted
        before normalization.
        """
        # Run simulation if needed
        if self._logs is None:
            self._run()
        # Create a copy of the voltage steps
        v = np.array(self._steps, copy=True)
        # Create a simulation log
        d = myokit.DataLog()
        d[self._vvar] = v
        # Factor
        factor = self._factor if self._factor is not None else 1
        # Find the peaks
        for var in self._vars:
            peaks = np.zeros(len(v))
            for k, log in enumerate(self._logs):
                peaks[k] = log[var][np.argmax(np.abs(log[var]))]
            d[var] = peaks * factor
        # Normalize (only if log contains positive values)
        if normalize:
            for var in self._vars:
                x = d[var]
                m = np.max(x)
                if m > 0:
                    d[var] = x / m
        return d
    def _run(self):
        """
        Should run the simulation and save the current traces in self._logs
        """
        raise NotImplementedError   
    def set_constant(self, var, value):
        """
        Changes the value of a constant in the used model.
        """
        if isinstance(var, myokit.Variable):
            var = var.qname()
        var = self._model.get(var)
        self._model.set_value(var, value)
        self._logs = None
    def set_holding_potential(self, vhold=-140, thold=1800):
        """
        Sets the holding potential and the time to hold. During the experiment,
        the cell will be held at ``vhold`` for ``thold`` time units before
        every voltage step.
        """
        vhold = float(vhold)
        thold = float(thold)
        if thold < 0:
            raise ValueError('Time to hold cannot be negative.')
        self._vhold = vhold
        self._thold = thold
        self._logs = None
    def set_max_step_size(self, dtmax=None):
        """
        Can be used to set a maximum step size to use in the logged parts of
        the simulation. Use ``dtmax==None`` to let the solver chose any size it
        likes.
        """
        self._max_step_size = None if dtmax is None else float(dtmax)
    def set_step_potential(self, vmin=-100, vmax=50, dv=1, tstep=200):
        """
        Sets the step potentials and the step duration. Each experiment will
        step linearly from ``vmin`` to ``vmax`` with steps of size ``dv``. The
        cell is held at each step for ``tstep`` time units.
        """
        vmin = float(vmin)
        vmax = float(vmax)
        dv = float(dv)
        tstep = float(tstep)
        if vmax <= vmin:
            raise ValueError('Maximum voltage must be greater than minimum'
                ' voltage.')
        if dv <= 0:
            raise ValueError('The voltage increment dv must be greater'
                ' than zero.')
        if tstep < 0:
            raise ValueError('Step duration cannot be negative.')
        self._vmin = vmin
        self._vmax = vmax
        self._dv = dv
        self._tstep = tstep
        self._steps = np.arange(self._vmin, self._vmax + self._dv, self._dv)
        self._logs = None
    def steps(self):
        """
        Returns the list of steps this protocol will use.
        """
        return list(self._steps)
    def times(self):
        """
        Returns a :class:`myokit.DataLog` containing the time-to-peak for
        each logged variable at each voltage step.
        """
        # Run simulation if needed
        if self._logs is None:
            self._run()
        # Create a copy of the voltage steps
        v = np.array(self._steps, copy=True)
        # Create a simulation log
        d = myokit.DataLog()
        d[self._vvar] = v
        # Find the peaks
        for var in self._vars:
            times = np.zeros(len(v))
            for k, log in enumerate(self._logs):
                times[k] = log[self._tvar][np.argmax(np.abs(log[var]))]
            d[var] = times
        return d
    def traces(self):
        """
        Returns the logged traces for each variable as an ordered list of
        tuples ``(v, DataLog)``.
        
        If any conversion factor was specified the data will be converted
        before returning.
        """
        if self._logs is None:
            self._run()
        data = []
        steps = iter(self._steps)
        factor = self._factor
        if factor is None:
            factor = np.ones(len(self._steps))
        for k, log in enumerate(self._logs):
            v = steps.next()
            d = myokit.DataLog()
            for var in self._vars:
                d[var] = np.array(log[var]) * factor[k]
                d[self._tvar] = log[self._tvar]
            data.append((v, d))
        return data
class Activation(StepProtocol):
    """
    Runs a step protocol and measures during the step. Can be used to create
    activation curves and I-V curves.
    
    The protocol is defined as follows: Initially, the membrane potential is
    held at a holding potential ``vhold`` for the duration of ``thold`` time
    units. Then, the voltage is changed to ``vstep`` and held there for
    ``vstep`` time units. During this step, the cell's response is logged. The
    experiment is repeated for ``vstep`` values ranging linearly from ``vmin``
    to ``vmax`` with an increment of ``dv``.
    
    ::
    
      .               +--- vstep ---+
      .               +-------------+
      .               +-------------+
      .               |             |
      .  vhold -------+             +-
      .  t=0          t=thold       t=thold+tstep
      .  no current   current!
    
    Accepts the following input arguments:
    
    ``model``
        The model for which to run the simulations
    ``var``
        A variable or a list of variables to log and use in calculations
    ``vvar``
        The membrane potential variable or its qname. If not given, the method
        will search for a variable labelled ``membrane_potential``.
        
    Depending on the experiment being run, ``var`` could be a conductance or a
    current variable (or a list of conductances or currents).
    
    The experiment is not performed until a call to one of the post-processing
    methods is made. After this, the raw data will be cached. Any change to the
    protocol variables after this point will delete the cached data.
    """
    def __init__(self, model, var, vvar=None):
        super(Activation, self).__init__(model, var, vvar)
    def _run(self):
        """
        Runs the experiment, logs during the voltage steps.
        """
        log = self._vars + [self._tvar]
        s = myokit.Simulation(self._model)
        d = []
        for v in self._steps:
            s.reset()
            s.set_constant(self._vvar, self._vhold)
            s.set_max_step_size(None)
            s.pre(self._thold)
            s.set_constant(self._vvar, v)
            s.set_max_step_size(self._max_step_size)
            d.append(s.run(self._tstep, log=log))
        self._logs = d
class Inactivation(StepProtocol):
    """
    Can run an inactivation step protocol on a model and calculate various
    entities using the results.
    
    The protocol starts by holding the membrane potential at a high value,
    causing the channels to fully activate and then inactivate. Next, the
    potential is stepped to a lower value, causing the inactivation to go away
    while the activation stays close to full. Current is measured after the
    step, when the cell is at the holding potential.

    ::
    
      .  --- vstep ---+
      .  -------------+              
      .  -------------+
      .               |
      .               +--- vhold ---+-
      .  t=0          t=tstep       t=tstep+thold
    
    Accepts the following input arguments:
    
    ``model``
        The model for which to run the simulations
    ``var``
        A variable or a list of variables to log and use in calculations
    ``vvar``
        The membrane potential variable or its qname. If not given, the method
        will search for a variable labelled ``membrane_potential``.
        
    Depending on the experiment being run, ``var`` could be a conductance or a
    current variable (or a list of conductances or currents).
    
    The experiment is not performed until a call to one of the post-processing
    methods is made. After this, the raw data will be cached. Any change to the
    protocol variables after this point will delete the cached data.
    """
    def __init__(self, model, var, vvar=None):
        super(Inactivation, self).__init__(model, var, vvar)
    def _run(self):
        """
        Runs the simulation, saves the current traces.
        """
        log = self._vars + [self._tvar]
        d = []
        s = myokit.Simulation(self._model)
        for v in self._steps:
            s.reset()
            s.set_constant(self._vvar, v)
            s.pre(self._tstep)
            s.set_constant(self._vvar, self._vhold)
            d.append(s.run(self._thold, log=log))
        self._logs = d
    def set_holding_potential(self, vhold=-20, thold=50):
        """
        Sets the holding potential and the time to hold. During the experiment,
        the cell will be held at ``vhold`` for ``thold`` time units before
        every voltage step.
        """
        super(Inactivation, self).set_holding_potential(vhold, thold)
    def set_step_potential(self, vmin=-100, vmax=-40, dv=1, tstep=1000):
        """
        Sets the step potentials and the step duration. Each experiment will
        step linearly from ``vmin`` to ``vmax`` with steps of size ``dv``. The
        cell is held at each step for ``tstep`` time units.
        """
        super(Inactivation, self).set_step_potential(vmin, vmax, dv, tstep)
class Recovery(object):
    """
    Can run a two-pulse recovery from inactivation experiment and process the
    results.
    
    The experiment proceeds as follows: first, the cell is held at a low
    holding potential ``vhold`` for ``thold`` time units. This causes the 
    channels to fully deactivate (activation=0) and fully recover from
    inactivation (inactivation=1). Next, the membrane is stepped up to a 
    voltage ``vstep`` and kept there for ``tstep1`` time units. This causes the
    channels to activate (activation=1) and then inactivate (inactivation=0).
    The model then steps back down to ``vhold`` for ``twait`` time units,
    during which partial recovery from inactivation is expected. After this
    short recovery period, the cell is stepped back to ``vstep`` for another
    ``tstep2`` time units. The recovery from inactivation can then be judged
    by comparing the peak current during step2 with the peak current during
    step1. By varying the time between steps ``twait`` a plot of recovery
    characteristics can be made.
    
    ::
    
      .                +--- vstep ---+         +- vstep -+
      .                |             |  twait  |         |
      .                |             | <-----> |         |
      .                |             |         |         |
      .  +--- vhold ---+             +- vhold -+         +---
      .  t=0           t=thold       t+=tstep1 t+=twait  t+=tstep2
    
    Accepts the following input arguments:
    
    ``model``
        The model for which to run the simulations
    ``var``
        A conductance variable (or a list of variables) to record during the
        experiment. Variables can be specified using Variable objects or
        through their fully qualified names.
    ``vvar``
        The membrane potential variable or its qname. If not given, the method
        will search for a variable labelled ``membrane_potential``.

    The argument ``var`` expects conduction variables, not currents. In other
    words, it expects the name of a variable ``g`` such that
    ``I = Gmax * g * (V - E)``. Using conductance in this test rather than
    current avoids the numerical problems incurred by dividing ``I`` through
    ``(V-E)``.
    """
    def __init__(self, model, var, vvar=None):
        if not model.is_valid():
            raise ValueError('This method requires a valid model.')
        # Clone model & variables
        self._model = model.clone()
        # Get time variable
        self._tvar = self._model.time()
        # Check conductance variables        
        self._vars = []
        if type(var) in [str, unicode] or isinstance(var, myokit.Variable):
            var = [var]
        for v in var:
            if isinstance(v, myokit.Variable):
                v = v.qname()
            elif '.' not in v:
                raise ValueError('The variable name(s) given as var must be'
                    ' given as fully qualified names <component.variable>.')
            self._vars.append(self._model.get(v))
        # Check membrane potential
        if vvar is None:
            self._vvar = self._model.label('membrane_potential')
            if self._vvar is None:
                raise ValueError('Membrane potential variable must be given by'
                    ' vvar or specified using the label "membrane_potential".')
        else:
            if isinstance(vvar, myokit.Variable):
                vvar = vvar.qname()
            elif '.' not in vvar:
                raise ValueError('The variable name vvar must be given as a'
                    ' fully qualified variable name <component.var>.')
            self._vvar = self._model.get(vvar)
        # Update membrane potential variable
        if self._vvar.is_state():
            self._vvar.demote()
        self._vvar.set_binding(None)
        # Set voltages
        self.set_holding_potential()
        self.set_step_potential()
        self.set_pause_duration()        
    def ratio(self):
        """
        Returns the ratios of the peak conductances (p1 / p2) for step 1 and
        step 2.
        
        The returned value is a :class:`myokit.DataLog` with entries
        corresponding to the given variable names.
        """
        # Times to wait
        twaits = np.exp(
            np.linspace(np.log(self._tmin), np.log(self._tmax), self._nt))
        # Variables to log
        log_vars = [x.qname() for x in self._vars]        
        # Run simulations
        self._vvar.set_rhs(self._vhold) # Make V a constant
        s = myokit.Simulation(self._model)
        log = myokit.DataLog()
        gvars = [x.qname() for x in self._vars]
        for g in gvars:
            log[g] = []
        log[self._tvar.qname()] = list(twaits)
        for twait in twaits:
            s.set_constant(self._vvar, self._vhold)
            s.run(self._thold, log=myokit.LOG_NONE)
            s.set_constant(self._vvar, self._vstep)
            d1 = s.run(self._tstep1, log=log_vars)
            s.set_constant(self._vvar, self._vhold)
            s.run(twait, log=myokit.LOG_NONE)
            s.set_constant(self._vvar, self._vstep)
            d2 = s.run(self._tstep2, log=log_vars)
            for g in gvars:
                ratio = np.max(d1[g])
                ratio = np.nan if ratio == 0 else np.max(d2[g]) / ratio
                log[g].append(ratio)
        return log
    def set_holding_potential(self, vhold=-120, thold=2000):
        """
        Sets the holding potential and the time to hold. During the experiment,
        the cell will be held at ``vhold`` for ``thold`` time units before
        every voltage step.
        """
        vhold = float(vhold)
        thold = float(thold)
        if thold < 0:
            raise ValueError('Time to hold cannot be negative.')
        self._vhold = vhold
        self._thold = thold
    def set_pause_duration(self, tmin=0.5, tmax=1000, nt=50):
        """
        Sets the duration of the pauses between the steps.
    
        ``tmin``
            The shortest time between steps
        ``tmax``
            The longest time between steps
        ``nt``
            The number of times to test 

        """
        tmin = float(tmin)
        tmax = float(tmax)
        nt = int(nt)
        if tmin < 0:
            raise ValueError('Minimum time cannot be negative.')
        if tmax < 0:
            raise ValueError('Maximum time cannot be positive.')
        if tmin >= tmax:
            raise ValueError('Maximum time must be grater than minimum.')
        if nt < 2:
            raise ValueError('The number of times to test must be greater than'
                ' one.')
        self._tmin = tmin
        self._tmax = tmax
        self._nt = nt
    def set_step_potential(self, vstep=-20, tstep1=500, tstep2=25):
        """
        Sets the step potential and the step durations.
        
        ``vstep``
            The potential used during the steps
        ``tstep1``
            The duration of the first pulse
        ``tstep2``
            The duration of the second pulse

        """
        vstep = float(vstep)
        tstep1 = float(tstep1)
        tstep2 = float(tstep2)
        if tstep1 < 0:
            raise ValueError('Time of first step cannot be negative.')
        if tstep2 < 0:
            raise ValueError('Time of second step cannot be negative.')
        self._vstep = vstep
        self._tstep1 = tstep1
        self._tstep2 = tstep2
class Restitution(object):
    """
    Can run a restitution experiment and return the values needed to make a
    plot.
    
    Accepts the following input arguments:
    
    ``model``
        The model for which to run the simulations
    ``vvar``
        The variable or variable name representing membrane potential. If not
        given, the method will look for the label ``membrane_potential``, if
        that's not found an exception is raised.

    """
    def __init__(self, model, vvar=None):
        # Check model
        if not model.is_valid():
            raise ValueError('This method requires a validated model.')
        self._model = model.clone()
        # Check membrane potential
        if vvar is None:
            self._vvar = self._model.label('membrane_potential')
            if self._vvar is None:
                raise ValueError('Membrane potential variable must be given by'
                    ' vvar or specified using the label "membrane_potential".')
        else:
            if isinstance(vvar, myokit.Variable):
                vvar = vvar.qname()
            elif '.' not in vvar:
                raise ValueError('The variable name vvar must be given as a'
                    ' fully qualified variable name <component.var>.')
            self._vvar = self._model.get(vvar)
        # Set default arguments
        self.set_max_step_size()
        self.set_times()
        self.set_beats()
        self.set_stimulus()
        self.set_threshold()
        # No data yet!
        self._data = None
    def _run(self):
        """
        Runs the simulations, saves the data.
        """
        # Create protocol
        e = {
            'level' : self._stim_level,
            'start' : 0,
            'duration' : self._stim_duration,
            'period' : self._clmin,
            'multiplier' : 0,
            }
        p = myokit.Protocol()
        p.schedule(**e)
        # Create simulation
        s = myokit.Simulation(self._model, p, apd_var=self._vvar)
        s.set_max_step_size(self._max_step_size)
        # Start testing
        c = self._clmin
        pcls = []
        apds = []
        while c < self._clmax:
            # Run simulation
            s.reset()
            s.pre(c * self._pre_beats)
            d,a = s.run(
                c * self._beats,
                log=myokit.LOG_NONE,
                apd_threshold=self._apd_threshold)
            # Save apds
            for apd in a['duration']:
                pcls.append(c)
                apds.append(apd)
            # Increase cycle length
            c += self._dcl
            # Create and set new protocol
            e['period'] = c
            p = myokit.Protocol()
            p.schedule(**e)
            s.set_protocol(p)
        # Store data
        self._data = pcls, apds
    def run(self):
        """
        Returns a :class:`DataLog` containing the tested cycle lengths as
        ``cl`` and the measured action potential durations as ``apd``. The
        diastolic intervals are given as ``di``.
        
        Each cycle length is repeated ``beats`` number of times, where
        ``beats`` is the number of beats specified in the constructor.
        """
        # Run
        if self._data is None:
            self._run()
        # Get data
        cl, apd = self._data
        d = myokit.DataLog()
        d['cl'] = list(cl)
        d['apd'] = list(apd)
        d['di'] = list(np.array(cl, copy=False) - np.array(apd, copy=False))
        return d
    def set_beats(self, beats=2, pre=50):
        """
        Sets the number of beats each cycle length is tested for.

        ``beats``
            The number of beats during which apd is measured
        ``pre``
            The number of pre-pacing beats done at each cycle length before the
            measurement.
        """
        beats = int(beats)
        pre = int(pre)
        if beats < 1:
            raise ValueError('The number of beats must be an integer greater'
                ' than zero.')
        if pre < 0:
            raise ValueError('The number of pre-pacing beats must be a'
                ' positive integer.')
        self._beats = beats
        self._pre_beats = pre
        self._data = None
    def set_max_step_size(self, dtmax=None):
        """
        Sets an (optional) maximum step size for the solver. To let the solver
        pick any step size it likes, use ``dtmax=None``.
        
        This method can be useful to avoid "CVODE flag 22" errors.
        """
        if dtmax is None:
            self._max_step_size=None
        else:
            dtmax = float(dtmax)
            if dtmax <= 0:
                raise ValueError('Maximum step size must be greater than'
                    ' zero.')
            self._max_step_size = dtmax
        self._data = None
    def set_stimulus(self, duration=2.0, level=1):
        """
        Sets the stimulus used to pace the model.
        
        ``stim_duration``
            The duration of the pacing stimulus.
        ``stim_level``
            The level of the dimensionless pacing stimulus.
        """
        duration = float(duration)
        level = float(level)
        if duration < 0:
            raise ValueError('The duratio cannot be negative.')
        self._stim_duration = duration
        self._stim_level = level
        self._data = None
    def set_threshold(self, threshold=-70):
        """
        Sets the APD threshold, specified as a fixed membrane potential.
        """
        self._apd_threshold = float(threshold)
        self._data = None
    def set_times(self, clmin=300, clmax=1200, dcl=20):
        """
        Sets the pacing cycle lengths tested in this experiment.

        ``clmin``
            The lowest cycle-time tested (where
            ``cl = systole + diastole length``)
        ``clmax``
            The highest cycle-time tested (where
            ``cl = systole + diastole length``)
        ``dcl``
            The size of the steps from ``clmin`` to ``clmax``
        
        """
        clmin = float(clmin)
        clmax = float(clmax)
        dcl = float(dcl)
        if clmin < 0:
            raise ValueError('Minimum time cannot be negative.')
        if clmax < 0:
            raise ValueError('Maximum time cannot be negative.')
        if clmin >= clmax:
            raise ValueError('Minimum time must be smaller than maximum time.')
        if dcl <= 0:
            raise ValueError('Step size must be greater than zero')
        self._clmin = clmin
        self._clmax = clmax
        self._dcl = dcl
        self._data = None
class StrengthDuration(object):
    """
    For a range of durations, this experiment checks the stimulus size needed
    to excite a cell.
    
    Accepts the following input arguments:
    
    ``model``
        The model to run the experiment with.
    ``ivar``
        A variable that can be used as the stimulus current.  It's value will
        be set as ``pace * amplitude`` where pace is a variable bound to the
        pacing signal and amplitude is varied by the method.
    ``vvar``
        A variable that indicates the membrane potential. If not specified (or
        given as ``None``), the method will search for a variable labeled as
        ``membrane_potential``.
        
    """
    def __init__(self, model, ivar, vvar=None):
        # Clone model
        self._model = model.clone()
        # Get stimulus current variable
        self._ivar = self._model.get(ivar)
        # Get membrane potential variable
        if vvar is None:
            self._vvar = model.label('membrane_potential')
            if self._vvar is None:
                raise ValueError('This method requires the membrane potential'
                    ' variable to be passed in as `vvar` or indicated in the'
                    ' model using the label `membrane_potential`.')
        else:
            if isinstance(vvar, myokit.Variable):
                vvar = vvar.qname()
            self._vvar = model.get(vvar)
        # Get time variable
        self._tvar = self._model.time()
        del(model, vvar, ivar)
        # Unbind any existing pace variable
        var = self._model.binding('pace')
        if var is not None:
            var.set_binding(None)
        # Create new pacing variable
        c = self._ivar.parent(myokit.Component)
        def add_variable(c, name):
            try:
                return c.add_variable(name)
            except myokit.DuplicateName:
                i = 2
                n = name + str(i)
                while True:
                    try:
                        return c.add_variable(n)
                    except myokit.DuplicateName:
                        i += 1
                        n = name + str(i)
        self._pvar = add_variable(c, 'pace')
        self._pvar.set_binding('pace')
        self._pvar.set_rhs(0)
        # Create new amplitude variable
        self._avar = add_variable(c, 'amplitude')
        self._avar.set_rhs(0)
        # Set rhs of current variable
        if self._ivar.is_state():
            self._ivar.demote()
        self._ivar.set_rhs(myokit.Multiply(self._pvar.lhs(), self._avar.lhs()))
        # Set default parameters
        self.set_currents()
        self.set_precision()
        self.set_threshold()
        self.set_times()
        # No data yet!
        self._data = None
    def run(self, debug=False):
        """
        Runs the experiment, returns a :class:`myokit.DataLog` with the
        entries `durations` and `amplitudes`, where the value in `amplitudes`
        is the minimum strength required to create a depolarization at the
        corresponding duration.
        """
        if self._data is None:
            self._run(debug)
        return self._data
    def _run(self, debug=False):
        """
        Inner version of run()
        """
        if debug:
            import traceback
        # Create simulation
        s = myokit.Simulation(self._model)
        # Variables to log
        tvar = self._tvar.qname()
        vvar = self._vvar.qname()
        # Output data
        durations = np.array(self._durations, copy=True)
        amplitudes = np.zeros(durations.shape)
        # Test every duration        
        for k, duration in enumerate(durations):
            if debug:
                print('Testing duration: ' + str(duration))
            s.set_protocol(myokit.pacing.blocktrain(self._time+1, duration))
            a1 = self._amin
            a2 = self._amax
            # Test minimum amplitude
            s.reset()
            s.set_constant(self._avar, a1)
            try:
                d = s.run(self._time, log=[vvar]).npview()
                t1 = (np.max(d[vvar]) > self._threshold)
            except Exception as e:
                if debug:
                    traceback.print_exc()
                t1 = False
            if debug:
                print(t1)
            # Test maximum amplitude
            s.reset()
            s.set_constant(self._avar, a2)
            try:
                d = s.run(self._time, log=[vvar]).npview()
                t2 = (np.max(d[vvar]) > self._threshold)
            except Exception as e:
                if debug:
                    traceback.print_exc()
                t2 = False
            if debug:
                print t2
            if t1 == t2:
                # No zero crossing found
                amplitudes[k] = np.nan
                if debug:
                    print('> no zero crossing')
                continue
            # Zero must lie in between. Start bisection search
            a = 0.5 * a1 + 0.5 * a2
            for j in xrange(0, self._precision):
                s.reset()
                s.set_constant(self._avar, a)
                try:
                    d = s.run(self._time, log=[vvar]).npview()
                except Exception as e:
                    if debug:
                        traceback.print_exc()
                    break
                t = (np.max(d[vvar]) > self._threshold)
                if t1 == t:
                    a1 = a
                else:
                    a2 = a
                a = 0.5 * a1 + 0.5 * a2
            amplitudes[k] = a
            if debug:
                print('> ' + str(a))
        # Set output data
        self._data = myokit.DataLog()
        self._data['duration'] = durations
        self._data['strength'] = amplitudes
    def set_currents(self, imin=-250, imax=0):
        """
        Sets the range of current levels tested.
        
        ``imin``
            The lowest (or most negative) current amplitude to test.
        ``imax``
            The greatest (or least negative) current amplitude to test.
        """
        amin = float(imin)
        amax = float(imax)
        if amin >= amax:
            raise ValueError('Minimum current amplitude must be smaller than'
                ' maximum value.')
        self._amin = amin
        self._amax = amax
        self._data = None
    def set_precision(self, precision=10):
        """
        Sets the number of different amplitudes tried. This is done using a
        bisection algorithm, so low values of ``precision`` can still produce a
        good result.
        """
        precision = int(precision)
        if precision < 1:
            raise ValueError('The number of tries must be greater than zero.')
        self._precision = precision
        self._data = None
    def set_threshold(self, threshold=10):
        """
        Sets the level above which the membrane potential must rise to count as
        a depolarization.
        """
        self._threshold = float(threshold)
        self._data = None
    def set_times(self, tmin=0.2, tmax=2.0, dt=0.1, twait=50):
        """
        Sets the tested stimulus durations.

        ``tmin``
            The smallest stimulus time tested.
        ``tmax``
            The largest stimulus time tested.
        ``dt``
            The step size when going from tmin to tmax.
        ``twait``
            The duration of the experiment that tries to measure a
            depolarization.
            
        """
        tmin = float(tmin)
        tmax = float(tmax)
        if tmin < 0:
            raise ValueError('Minimum time cannot be negative.')
        if tmax < 0:
            raise ValueError('Maximum time cannot be negative.')
        if tmin >= tmax:
            raise ValueError('The maximum time must be greater than the'
                ' minimum time.')
        dt = float(dt)
        if dt <= 0:
            raise ValueError('The step size must be greater than zero.')
        twait = float(twait)
        if twait < 0:
            raise ValueError('The time "twait" must be greater than zero.')
        self._durations = np.arange(tmin, tmax, dt)
        self._time = twait
        self._data = None
