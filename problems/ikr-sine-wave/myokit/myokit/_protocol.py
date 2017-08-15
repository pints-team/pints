# 
# Defines the python classes that represent a pacing protocol.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import myokit
import numpy as np
class Protocol(object):
    """
    Represents a pacing protocol as a sequence of :class:`events
    <ProtocolEvent>`.

    Every event represents a time span during which the stimulus is non-zero.
    All events specify a stimulus level, a starting time and a duration. The
    stimulus level is given as a dimensionless number where 0 is no stimulus
    and 1 is a "full" stimulus. No range is specified: 2 would indicate "twice
    the normal stimulus".

    Periodic events can be created by specifying their period. A period of 0 is
    used for non-periodic events. The number of times a periodic event occurs
    can be set using the `multiplier` value. For any non-zero value, the event
    will occur exactly this many times.

    If an event starts while another event is already active, the new event
    will determine the pacing value and the old event will be deactivated.
    
    Scheduling two events to occur at the same time will result in an error, as
    the resulting behavior would be undefined. When events are scheduled with
    the same starting time an error will be raised immediately. If the clash
    only happens when an event re-occurs, an error will be raised when using
    the protocol (for example during a simulation).
    
    An event with start time ``a`` and duration ``b`` will be active during
    the interval ``[a, b)``. In other words, time ``a`` will be the first time
    it is active and time ``b`` will be the first time after ``a`` at which it
    is not.
    """
    def __init__(self):
        super(Protocol, self).__init__()
        self._head = None
    def add(self, e):
        """
        (Re-)schedules an :class:`event <ProtocolEvent>`.
        """
        # No head? Then set as head
        if self._head is None:
            self._head = e
            return
        # Starts before head? Then replace as head
        if e._start < self._head._start:
            e._next = self._head
            self._head = e
            return
        # Find last event before this one
        f = self._head
        while f._next is not None and e._start >= f._next._start:
            f = f._next
        if e._start == f._start:
            raise myokit.SimultaneousProtocolEventError('Two events cannot'
                ' (re-)start at the same time: Error at time t='
                + str(e._start) + '.')
        e._next = f._next
        f._next = e
    def add_step(self, level, duration):
        """
        Appends an event to the end of this protocol. This method can be used
        to easily create voltage-step protocols.
        
        ``level``
            The stimulus level. 1 Represents a full-sized stimulus. Only
            non-zero levels should be set.
        ``duration``
            The length of the stimulus.

        A call to `p.append(level, duration)` is equivalent to
        `p.schedule(level, p.characteristic_time(), duration, 0, 0)`.        
        """
        self.schedule(level, self.characteristic_time(), duration)
    def characteristic_time(self):
        """
        Returns the characteristic time associated with this protocol.
        
        The characteristic time is defined as the maximum characteristic time
        of all events in the protocol. For a sequence of events, this is simply
        the protocol duration.

        Examples:
        
        >>> import myokit
        >>> # A sequence of events
        >>> p = myokit.Protocol()
        >>> p.schedule(1, 0, 100)
        >>> p.schedule(1, 100, 100)
        >>> p.characteristic_time()
        200.0
        
        >>> # A finitely reoccurring event
        >>> p = myokit.Protocol()
        >>> p.schedule(1, 100, 0.5, 1000, 3)
        >>> p.characteristic_time()
        3100.0
        
        >>> # An indefinitely reoccurring event, method returns period
        >>> p = myokit.Protocol()
        >>> p.schedule(1, 100, 0.5, 1000, 0)
        >>> p.characteristic_time()
        1000.0
        
        """
        # Find maximum
        tmax = 0
        e = self._head
        while e is not None:
            tmax = max(tmax, e.characteristic_time())
            e = e._next
        return tmax
    def clone(self):
        """
        Returns a deep clone of this protocol.
        """
        p = Protocol()
        e = self._head
        while e is not None:
            p.add(e.clone())
            e = e._next
        return p
    def code(self):
        """
        Returns the ``mmt`` code representing this protocol and its events.
        """
        out = [
            '[[protocol]]',
            '# Level  Start    Length   Period   Multiplier',
            ]
        e = self._head
        while(e is not None):
            out.append(e.code())
            e = e._next
        return '\n'.join(out)
    def create_log_for_interval(self, a, b, for_drawing=False):
        """
        Creates a :class:`myokit.DataLog` containing the entries `time`
        and `pace` representing the value of the pacing stimulus at each point.
        
        The time points in the log will be on the interval ``[a, b]``, such
        that every time at which the pacing value changes is present in the
        log.
        
        If ``for_drawing`` is set to ``True`` each time value between ``a`` and
        ``b`` will be listed twice, so that a vertical line can be drawn from
        the old to the new pacing value.
        """
        # Test the input
        a, b = float(a), float(b)
        if b < a:
            raise ValueError('The argument `b` cannot be smaller than `a`')
        # Create a simulation log
        log = myokit.DataLog()
        log.set_time_key('time')
        log['time'] = time = []
        log['pace'] = pace = []
        # Create a pacing system
        p = PacingSystem(self)
        # Fill in points
        t = a
        v = p.advance(t, max_time=b)
        time.append(t)
        pace.append(v)
        while t < b:
            t = p.next_time()
            if for_drawing:
                if t != b:
                    time.append(t)
                    pace.append(v)
            v = p.advance(t, max_time=b)
            time.append(t)
            pace.append(v)
        return log
    def create_log_for_times(self, times):
        """
        Creates a :class:`myokit.DataLog` containing the entries `time`
        and `pace` representing the value of the pacing stimulus at each point.
        
        The time entries ``times`` must be an non-descreasing series of
        non-negative points.
        """
        # Create empty log
        log = myokit.DataLog()
        log.set_time_key('time')
        # Times empty? Then return empty log
        if len(times) == 0:
            log['time'] = []
            log['pace'] = []
            return log
        # Test times
        times = np.asarray(times)
        dt = times[1:] - times[:-1]
        if np.any(dt < 0):
            raise ValueError('The argument `times` must contain a'
                ' non-decreasing sequence of time points.')
        if times[0] < 0:
            raise ValueError('Times cannot be negative.')
        # Create a pacing system and calculate the values
        p = PacingSystem(self)
        # Return log
        log['time'] = list(times)
        log['pace'] = [p.advance(t) for t in times]
        return log
    def events(self):
        """
        Returns a list of all events in this protocol.
        """
        # Note: This should remain a list, not an iterator. The pacing module
        # in pacing.h depends on it!
        out = []
        e = self._head
        while e is not None:
            out.append(e)
            e = e._next
        return out
    def guess_duration(self):
        """
        *Deprecated*
        
        This method now returns the value given by :meth:`characteristic_time`.
        """
        return self.characteristic_time()
    def head(self):
        """
        Returns the first event in this protocol.
        
        If the protocol is empty, ``None`` will be returned.
        """
        return self._head
    def in_words(self):
        """
        Returns a description of this protocol in words.
        """
        if self._head is None:
            return "Empty protocol"
        out = []
        e = self._head
        while e is not None:
            out.append(e.in_words())
            e = e._next
        return '\n'.join(out)
    def is_infinite(self):
        """
        Returns True if (and only if) this protocol contains indefinitely
        recurring events.
        """
        e = self._head
        while e is not None:
            if e._period != 0 and e._multiplier == 0:
                return True
            e = e._next
        return False
    def is_sequence(self, exception=False):
        """
        Checks if this protocol is a sequence of non-periodic steps.
        
        The following checks are performed:
        
        1. The protocol does not contain any periodic events
        2. The protocol does not contain any overlapping events
        
        If ``exception`` is set to ``True``, the method will raise an
        ``Exception`` instead of returning ``False``. This allows you to obtain
        specific information about the check that failed
        """
        def check():
            t = 0
            e = self._head
            while e is not None:
                if e._period != 0:
                    raise Exception('Protocol contains periodic event(s).')
                if e._start < t:
                    raise Exception('Event starting at t=' + str(e._start)
                        + ' overlaps with previous event which finishes at t='
                        + str(t) + '.')
                t = e._start + e._duration
                e = e._next
            return True
        if exception:
            return check()
        else:
            try:
                return check()
            except Exception:
                return False
    def is_unbroken_sequence(self, exception=False):
        """
        Checks if this protocol is an unbroken sequence of steps. Returns
        ``True`` only for an unbroken sequence.
        
        The following checks are performed:
        
        1. The protocol does not contain any periodic events
        2. The protocol does not contain any overlapping events
        3. Each new event starts where the last ended
        
        If ``exception`` is set to ``True``, the method will raise an
        ``Exception`` instead of returning ``False``. This allows you to obtain
        specific information about the check that failed
        """
        def check():
            e = self._head
            if e is None:
                return True
            if e._period != 0:
                raise Exception('Protocol contains periodic event(s).')
            while e._next is not None:
                t = e._start + e._duration
                e = e._next
                # Check for periodic events
                if e._period != 0:
                    raise Exception('Protocol contains periodic event(s).')
                # Check starting time
                if e._start < t:
                    raise Exception('Event starting at t=' + str(e._start)
                        + ' overlaps with previous event which finishes at t='
                        + str(t) + '.')
                elif e._start > t:
                    raise Exception('Event starting at t=' + str(e._start)
                        + ' does not start directly after previous event,'
                        + ' which finishes at t=' + str(t) + '.')
            return True
        if exception:
            check()
        else:
            try:
                return check()
            except Exception:
                return False
    def __iter__(self):
        """
        Returns an iterator over the events in this protocol.
        """
        return iter(self.events())
    def levels(self):
        """
        Returns the levels of the events scheduled in this protocol.
        
        For unbroken sequences of events this will produce a list of the levels
        visited by the protocol. For sequences with gaps or protocols with
        periodic events the relationship between actual levels and this
        method's output is more complicated.
        """
        e = self._head
        levels = []
        while e is not None:
            levels.append(e._level)
            e = e._next
        return levels
    def pop(self):
        """
        Removes and returns the event at the head of the queue.
        """
        e = self._head
        if self._head is not None:
            self._head = self._head._next
        return e
    def range(self):
        """
        Returns the minimum and maximum levels set in this protocol.
        """
        lo, hi = 0, 0
        e = self._head
        while e is not None:
            lo = min(e._level, lo)
            hi = max(e._level, hi)
            e = e._next
        return lo, hi
    def schedule(self, level, start, duration, period=0, multiplier=0):
        """
        Schedules a new event.

        ``level``
            The stimulus level. 1 Represents a full-sized stimulus. Only
            non-zero levels should be set.
        ``start``
            The time this event first occurs.
        ``duration``
            The length of the stimulus.
        ``period (optional)``
            This event's period, or ``0`` if it is a one-off event.
        ``multiplier (optional)``
            For periodic events, this indicates the number of times this event
            occurs. Non-periodic events or periodic events that continue
            indefinitely can use ``0`` here.
        """
        e = ProtocolEvent(level, start, duration, period, multiplier)
        self.add(e)
    def __str__(self):
        return self.code()
    def tail(self):
        """
        Returns the last event in this protocol. Note that recurring events
        can be rescheduled, so that the event returned by this method is not
        necessarily the last event that would occur when running the protocol.
        
        If the protocol is empty, ``None`` will be returned.
        """
        e = self._head
        while e._next is not None:
            e = e._next
        return e
class ProtocolEvent(object):
    """
    Describes an event occurring as part of a protocol.
    """
    def __init__(self, level, start, duration, period=0, multiplier=0):
        """
        Creates a new event

        ``level``
            The level that this event will cause the engine to hold the pacing
            variable at.
        ``start``
            The time that this event is launched.
        ``duration``
            The time this event lasts. During this time, the pacing level will
            be at the level specified by this event, unless a later event kicks
            in and overrules it.
        ``period (optional)``
            For singular events, this should be zero. To create recurring
            events, this value is used to set the interval between firings.
        ``multiplier (optional)``
            For recurring events, this value is used to indicate the number of
            times an event occurs. For singular events (with ``period=0``) this
            value should always be zero.
        
        """
        self._next = None
        self._level = float(level)
        self._start = float(start)
        self._duration = float(duration)
        self._period = float(period)
        self._multiplier = int(multiplier)
        if self._start < 0:
            raise myokit.ProtocolEventError('An event can not start at a'
                ' negative time.')
        if self._duration < 0:
            raise myokit.ProtocolEventError('An event can not have a negative'
                ' duration.')
        if self._period < 0:
            raise myokit.ProtocolEventError('An event\'s period must be zero'
                ' or greater.')
        if self._multiplier < 0:
            raise myokit.ProtocolEventError('An event\'s multiplier must be'
                ' zero or greater')
        if self._period == 0 and self._multiplier > 0:
            raise myokit.ProtocolEventError('Non-recurring events can not'
                ' specify a multiplier.')
        if float(self._multiplier) != float(multiplier):
            raise myokit.ProtocolEventError('The event multiplier must be an'
                ' integer.')
        if self._period > 0 and self._duration > self._period:
            raise myokit.ProtocolEventError('A recurring event can not have a'
                ' duration that\'s longer than its period.')
    def characteristic_time(self):
        """
        Returns a characteristic time associated with this event.
        
        The time is calculated as follows:
        
        Singular events
            ``characteristic_time = start + duration``
        Finitely recurring events
            ``characteristic_time = start + multiplier * period``
        Indefinitely recurring events, where ``start + duration < period``
            ``characteristic_time = period``
        Indefinitely recurring events, where ``start + duration >= period``
            ``characteristic_time = start + period``
            
        Roughly, this means that for finite events the full duration is
        returned, while indefinitely recurring events return the time until
        the first period is completed.
        """
        if self._period == 0:
            # Singular event
            return self._start + self._duration
        elif self._multiplier > 0:
            # Finitely recurring event
            return self._start + self._period * self._multiplier
        elif self._start + self._duration < self._period:
            # Indefinitely recurring event that starts directly
            return self._period
        else:
            # Indefinitely recurring event that stars after a delay
            return self._start + self._period
    def clone(self):
        """
        Returns a clone of this event.
        
        Note that links to other events are not included in the copy!
        """
        return ProtocolEvent(self._level, self._start, self._duration,
            self._period, self._multiplier)
    def code(self):
        """
        Returns a consistently formatted string representing an event.
        """
        # Level  Start    Length   Period   Multiplier
        z = 0
        x = [
            self._level,
            self._start,
            self._duration,
            self._period,
            self._multiplier,
            ]
        x = [str(x) for x in x]
        s = [8, 8, 8, 8, 0]
        for i in xrange(0,5):
            n = s[i] - len(x[i]) + z
            z = 0
            if n > 0:
                x[i] += ' ' * n
            elif n < 0:
                z = n
        return ' '.join(x)
    def duration(self):
        """
        Returns this even't duration.
        """
        return self._duration
    def in_words(self):
        """
        Returns a description of this event.
        """
        out = 'Stimulus of ' + str(self._level) + ' times the normal level ' \
            'applied at t=' + str(self._start) + ', lasting ' \
            + str(self._duration)
        if self._period != 0:
            if self._multiplier > 0:
                out += ' and occurring ' + str(self._multiplier) + ' times'
            else:
                out += ' and recurring indefinitely'
            out += ' with a period of ' + str(self._period)
        return out + '.'
    def level(self):
        """
        Returns this event's pacing level.
        """
        return self._level
    def multiplier(self):
        """
        Returns the number of times this event recurs. Zero is returned for
        singular events and indefinitely recurring events.
        """
        return self._multiplier
    def next(self):
        """
        If this event is part of a :class:`myokit.Protocol`, this returns the
        next scheduled event.
        """
        return self._next
    def period(self):
        """
        Returns this event's period (or zero if the event is singular).
        """
        return self._period
    def start(self):
        """
        Returns the time this event starts.
        """
        return self._start
    def stop(self):
        """
        Returns the time this event ends, i.e. `start() + duration()`.
        """
        return self._start + self._duration
    def __str__(self):
        return self.in_words()
class PacingSystem(object):
    """
    This class uses a :class:`myokit.Protocol` to update the value of a
    pacing variable over time.
    
    A pacing system is created by passing in a protocol:
    
    >>> import myokit
    ... p = myokit.load_protocol('example')
    ... s = myokit.PacingSystem(p)
    
    The given protocol will be cloned internally before use.
    
    Initially, all pacing systems are at time 0. Time can be updated (but never
    moved back!) by calling :meth:`advance(new_time)`. The current time can be
    obtained with :meth:`time()`. The value of the pacing variable is obtained
    from :meth:`pace()`. The next time the pacing variable will change can be
    obtained from :meth:`next_time()`.
    
    A pacing system can be used to calculate the values of the pacing variable
    at different times:
    
    >>> import myokit
    ... p = myokit.load_protocol('example')
    ... s = myokit.PacingSystem(p)
    ... import numpy as np
    ... time = np.linspace(0, 1000, 10001)
    ... pace = np.array([s.advance(t) for t in time])

    """
    def __init__(self, protocol):
        # The current time and pacing level
        self._time = 0
        self._pace = 0
        # Currently active event
        self._fire = None
        # Time the currently active event is over
        self._tdown = None
        # The next time the pacing variable changes
        self._tnext = 0
        # Create a copy of the protocol
        self._protocol = protocol.clone()
        # Advance to time zero
        self.advance(0)
    def advance(self, new_time, max_time=None):
        """
        Advances the time in the pacing system to ``new_time``. If ``max_time``
        is set, the system will never go beyond ``max_time``.
        
        Returns the current value of the pacing variable.
        """
        # Update time
        new_time = float(new_time)
        if self._time > new_time:
            raise ValueError('New time cannot be before the current time.')
        if max_time is not None:
            max_time = float(max_time)
            if new_time > max_time:
                new_time = max_time
        self._time = new_time
        # Advance pacing system
        while self._tnext <= self._time:
            # Stop at max_time, if given
            if max_time is not None:
                if self._tnext == max_time:
                    break
            # Active event finished
            if self._fire and self._tnext >= self._tdown:
                self._fire = None
                self._pace = 0
            # New event starting
            e = self._protocol._head
            if e and self._time >= e._start:
                self._protocol.pop()
                self._fire = e
                self._tdown = e._start + e._duration
                self._pace = e._level
                # Reschedule recurring events
                if e._period > 0 and e._multiplier != 1:
                    if e._multiplier > 1:
                        e._multiplier  -= 1
                    e._start += e._period
                    self._protocol.add(e)
            # Next stopping time
            if max_time is not None:
                self._tnext = max_time
            else:
                self._tnext = float('inf')
            if self._fire and self._tnext > self._tdown:
                self._tnext = self._tdown
            if e and self._tnext > e._start:
                self._tnext = e._start
        return self._pace
    def next_time(self):
        """
        Returns the next time the value of the pacing variable will be
        updated.
        """
        return self._tnext
    def pace(self):
        """
        Returns the current value of the pacing variable.
        """
        return self._pace
    def time(self):
        """
        Returns the current time in the pacing system.
        """
        return self._time
