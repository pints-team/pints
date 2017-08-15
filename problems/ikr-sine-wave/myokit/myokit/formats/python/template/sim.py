<?
# sim.py
# A pype template for a single file python simulation
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
import myokit.formats.python as python

tab  = 4*' '
tab2 = 2*tab

def v(lhs, unless=None):
    """
    Translates a left-hand-side expression.

    The name is given fully qualified, unless the variable's parent comp
     is equal to the second parameter.
    """
    if isinstance(lhs, myokit.Derivative):
        var = lhs.var()
        pre = 'self' if var._parent == unless else c(var._parent)[0]
        return pre + '.d_' + var.name().lower()
    if isinstance(lhs, myokit.Name):
        var = lhs.var()
    else:
        var = lhs
    # Local variable
    out = var.uname()
    par = var.parent(myokit.Component)
    pre = 'self.' if par == unless else c(par)[0] + '.'
    return pre + out
def c(comp):
    """
    Returns the names for this comp as (Class, object)
    """
    lname = comp.name().lower()
    uname = lname[0].upper() + lname[1:]
    return ('c_' + lname, 'C' + uname)

# Create python expression writer
w = python.PythonExpressionWriter()
w.set_lhs_function(v)
e = w.eq

# Process bindings
bound_variables = model.prepare_bindings({
    'time' : 'engine.time',
    'pace' : 'engine.pace',
    })

# Get equations
equations = model.solvable_order()
components = []
for comp in equations:
    if comp != '*remaining*':
        components.append(model.get(comp))

?>#!/usr/bin/env python2
#
# Generated on <?= myokit.date() ?>
#
import math

#
# Components and variables
#
<?
# Generate class per component
for comp in components:
    names = c(comp)
    w.set_lhs_function(lambda x: v(x, comp))

    ?>
class <?= names[1] ?>(object):
    def __init__(self):
<?
    for var in comp.variables(deep=True):
        print(tab2 + v(var, comp) + ' = None')
        if var.is_state():
            print(tab2 + v(var.lhs(), comp) + ' = None')
    ?>
        self._constants()
        self.init()
    def _constants(self):
        """
        Sets the constant values
        """
<?
    eqs = equations[comp.qname()]
    for eq in eqs.equations(const=True):
        print(tab2 + e(eq))
    ?>    def init(self):
        """
        Resets the state variables to their initial values
        """
<?
    for eq in model.inits():
        if eq.lhs.var().parent() == comp:
            print(tab2 + e(eq))
    ?>    def update(self):
        """
        Re-calculates all values for the current time and state
        """
<?
    printed = False
    for eq in eqs.equations(const=False):
        printed = True
        var = eq.lhs.var()
        if var.is_bound():
            print(tab2 + v(var) + ' = ' + bound_variables[var])
        else:
            print(tab2 + e(eq))
    if not printed:
        print(tab2 + 'pass')
?>
#
# Engine component
#
class Engine(object):
    """
    Calculates the derivatives in the current state
    """
    def __init__(self):
        self.pace = 0.0
        self.time = 0.0
<?

# Update function, calls all others
print(tab + 'def update(self):')
for comp in components:
    print(tab2 + c(comp)[0] + '.update()')

# Now handle remaining equations
print(tab2 + '# Remaining equations')
for eq in equations['*remaining*'].equations(const=False):
    print(tab2 + e(eq))

?>
#
# Create objects, set initial values
#
def init():
    """ (Re-)Initializes the model """
<?
# Set all component instances all global variables
n = 0
for comp in components:
    names = c(comp)
    if len(names[0]) > n:
        n = len(names[0])
    print(tab + 'global ' + names[0])
print(tab + 'global ' + 'engine')
# Create component instances
for comp in components:
    names = c(comp)
    print(tab + names[0] + ' '*(n - len(names[0])) + ' = ' + names[1] + '()')
print(tab + 'engine = Engine()')
# Any remaining constants
for eq in equations['*remaining*'].equations(const=True):
    print(tab + e(eq))
?>
#
# Update function (rhs function, takes a single step)
#
def update(stepSize):
    """ Calculates all derivatives, update state, advances time """
    engine.update()
<?
m = 0
for var in model.states():
    m = max(m, len(v(var)))
for var in model.states():
    name = v(var)
    print(tab + name + ' '*(m - len(name)) + ' += stepSize * ' + v(var.eq().lhs))
?>
#
# State vector returning function
#
def state():
    """ Returns the state vector """
<?
names = [v(var) for var in model.states()]
print(tab + 'return [' + (',\n'+tab2).join(names) + ']')
?>
#
# State vector printing function
#
def print_state():
    """ Prints the current state to the screen """
<?
n = max([len(x.qname()) for x in model.states()])
n = max(n, 4)
?>
    f = "{:<<?=str(n)?>}  {:<20}  {:<20}"
    print("-"*80)
    print(f.format("Name", "State value", "Derivative"))
    f = "{: <<?=str(n)?>}  {:< 20.13e}  {:< 20.13e}"
<?
for var in model.states():
    print(tab+'print(f.format("' +var.qname()+'", '+v(var)+', '+v(var.lhs()) + '))')
?>
#
# Test step function
#
def test_step():
    """ Calculates and prints the initial derivatives """
    init()
    engine.update()
    print_state()

#
# Pacing
#
class Protocol(object):
    """ Holds an ordered set of ProtocolEvent objects """
    def __init__(self):
        super(Protocol, self).__init__()
        self.head = None
    def add(self, e):
        """ Schedules an event """
        if self.head is None:
            self.head = e
            return
        if e.start < self.head.start:
            e.next = self.head
            self.head = e
            return
        f = self.head
        while (f.next is not None) and (e.start > f.next.start):
            f = f.next
        e.next = f.next
        f.next = e
    def pop(self):
        """ Returns the next event """
        e = self.head
        if self.head is not None:
            self.head = self.head.next
        return e
class ProtocolEvent(object):
    def __init__(self, level, start, duration, period=0, multiplier=0):
        super(ProtocolEvent, self).__init__()
        self.level = float(level)
        self.start = float(start)
        self.duration = float(duration)
        if self.duration <= 0:
            raise Exception('Duration must be greater than zero')
        self.period = float(period)
        if self.period < 0:
            raise Exception('multiplier must be zero or greater')
        self.multiplier = int(multiplier)
        if self.multiplier < 0:
            raise Exception('Multiplier must be zero or greater')
        if self.period == 0 and self.multiplier > 0:
            raise Exception('Non-periodic event cannot occur more than once')
        self.next = None

def pacing_protocol():
    pacing = Protocol()
<?
next = protocol.head()
while next:
    x = [
        next.level(),
        next.start(),
        next.duration(),
        next.period(),
        next.multiplier(),
        ]
    x = ', '.join([str(x) for x in x])
    print(tab + 'pacing.add(ProtocolEvent(' + x + '))')
    next = next.next()
print(tab + 'return pacing')

vm = v(model.states().next())
?>
#
# Solver
#
def beat(stepSmall = 0.005, stepLarge = 0.01):
    """
    Simulates a single beat
    """
    tmin = 0
    tmax = 1000
    # Feedback
    outInt = int(math.ceil(tmax / 10.0))
    outPos = engine.time
    outVal = 0
    # Logging
    logInt = 1
    logPos = engine.time
    log = []
    # Stepsize
    stepSize = stepSmall
    hadPulse = False
    vInit = <?= vm ?>
    # Pacing
    pacing = pacing_protocol()
    next = pacing.head
    while next and next.start < tmin:
        next = next.next
    if next.start < tmin:
        next = None
    fire = None
    fireDown = 0
    stopTime = min(next.start, tmin + stepSize)
    print('Starting integration with step sizes ' + str(stepSmall) + ' and ' + str(stepLarge) + '.')
    while engine.time < tmax:
        update(stopTime - engine.time)
        engine.time = stopTime
        # Event over
        if (fire and engine.time >= fireDown):
            engine.pace = 0
            fire = None
        # New event
        if (next and engine.time >= next.start):
            fire = next
            next = next.next
            engine.pace = fire.level
            fireDown = fire.start + fire.duration
            if fire.period > 0:
                if fire.multiplier == 1:
                    fire.period = 0
                else:
                    if fire.multiplier > 1:
                        fire.multiplier -=1
                    fire.start += fire.period
                    pacing.add(fire)
                    next = pacing.head
        # User feedback
        if engine.time >= outPos and outVal < 100:
            print(str(outVal) + "%")
            outVal += 10
            outPos += outInt
        # Logging
        if engine.time >= logPos:
            log.append((engine.time, state()))
            logPos += logInt
        # Step size update
        if fire: # or <?= vm ?> > -70:
            if stepSize != stepSmall:
                print("Small steps")
                stepSize = stepSmall
        else:
            if stepSize != stepLarge:
                print("Big steps")
                stepSize = stepLarge
        # Set next time
        stopTime = engine.time + stepSize
        if fire and fireDown < stopTime:
            stopTime = fireDown
        if next and next.start < stopTime:
            stopTime = next.start
        if logPos < stopTime:
            stopTime = logPos
    print("100% done")
    print("t = " + str(engine.time))
    print_state()
    return log

#
# Run if loaded as main script
#
if __name__ == '__main__':
    small = 0.005
    large = 0.01
    go = True
    done = False
    while go:
        go = False
        try:
            init()
            data = beat(small, large)
            done = True
        except ArithmeticError as e:
            print('Arithmetic error occurred')
            y = raw_input('Continue with smaller stepsize? (y/n): ')
            if y.lower()[0:1] == 'y':
                small /= 2
                large /= 2
                go = True
    if done:
        print('Showing result...')
        x = []
        y = []
        for time, state in data:
            x.append(time)
            y.append(state[0])
        import matplotlib.pyplot as py
        plot = py.plot(x, y)
        py.show()
