<?
#
# euler.c
# A pype template for a simple ansi C simulation using an explicit forward
# euler method
#
# Required variables
# ---------------------------
# model    A model
# protocol A pacing protocol
# ---------------------------
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import time
import myokit
import myokit.formats.ansic as ansic

# Clone model
model = model.clone()

# Merge interdepdent components
model.merge_interdependent_components()

# Reserve keywords
model.reserve_unique_names(*ansic.keywords)
model.create_unique_names()

# Process bindings, remove unsupported bindings, get map of bound variables to
# internal names
bound_variables = model.prepare_bindings({
    'time' : 'time',
    'pace' : 'pace',
    })

# Get equations
equations = model.solvable_order()

# Get component order
comp_order = equations.keys()[:-1] # Strip *remaining*, guaranteed empty
comp_order = [model.get(c) for c in comp_order]

# Get component inputs/output arguments
comp_in, comp_out = model.map_component_io(
    omit_states = True,
    omit_derivatives = False,
    omit_constants = True,
    )

# Time and pace will be passed in to every function, so any variable bound to
# time or pace can be removed.
def clear_io_list(comp_list):
    for comp, clist in comp_list.iteritems():
        for var in bound_variables:
            lhs = var.lhs()
            while lhs in clist:
                clist.remove(lhs)
clear_io_list(comp_in)
clear_io_list(comp_out)

# Get first protocol event, use to create very basic pacing scheme
pacing = protocol.head()
if pacing:
    stim_level    = pacing.level()
    stim_duration = pacing.duration()
    stim_offset   = pacing.start()
    stim_period   = pacing.period()
    if stim_period == 0:
        stim_period = 1000
else:
    stim_level    = 1
    stim_duration = 0.5
    stim_offset   = 100
    stim_period   = 1000

# Get expression writer
w = ansic.AnsiCExpressionWriter()

# Define var/lhs function
pointerize = [] # Set of lhs arguments to write as pointers
def v(var):
    """
    Accepts a variable or a left-hand-side expression and returns its C
    representation.
    """
    pre = '*' if var in pointerize else ''
    if isinstance(var, myokit.Derivative):
        # Explicitly asked for derivative
        return pre + 'D_' + var.var().uname()
    if isinstance(var, myokit.Name):
        var = var.var()
    if var in bound_variables:
        return bound_variables[var]
    if var.is_state():
        return pre + 'state[' + str(var.indice()) + ']'
    elif var.is_constant():
        return pre + 'C_' + var.uname()
    else:
        return pre + 'I_' + var.uname()
w.set_lhs_function(v)

# Tab
tab = '    '

?>/*
Explicit forward-Euler simulation for <?=model.name()?>

*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Define standard floating point type. */
typedef double Real;

/* Pacing */
#define STIM_LEVEL    <?= stim_level ?>
#define STIM_DURATION <?= stim_duration ?>
#define STIM_OFFSET   <?= stim_offset ?>
#define STIM_PERIOD   <?= stim_period ?>

/* Define aliases of state variables */
#define N_STATE <?= model.count_states() ?>
<?
for var in model.states():
    print('#define S_' + var.uname() + ' state[' + str(var.indice()) + ']')
?>

/* Define constants, calculated constants */
<?
for group in equations.itervalues():
    for eq in group.equations(const=True):
        if isinstance(eq.rhs, myokit.Number):
            print('#define ' + v(eq.lhs) + ' ' + w.ex(eq.rhs))
        else:
            print('#define ' + v(eq.lhs) + ' (' + w.ex(eq.rhs) + ')')
?>

/* Components */
<?
for comp, ilist in comp_in.iteritems():
    olist = comp_out[comp]
    if len(olist) == 0:
        continue

    # Tell v() to pointerize the pointer outputs
    global pointerize
    pointerize_backup = pointerize
    pointerize = list(olist)

    # Function header
    args = ['const Real time', 'const Real pace', 'const Real* state']
    args.extend(['const Real ' + v(lhs) for lhs in ilist])
    args.extend(['Real ' + v(lhs) for lhs in olist])
    name = 'calc_' + comp.qname()
    print('void ' + name + '(' + ', '.join(args) + ')')
    print('{')

    # Equations
    for eq in equations[comp.name()].equations(const=False):
        pre = tab
        if not (eq.lhs in ilist or eq.lhs in olist):
            pre += 'Real '
        var = eq.lhs.var()
        if var not in bound_variables:
            print(pre + w.eq(eq) + ';')

    print('}')
    print('')

    # Tell v() not to pointerize anything
    pointerize = pointerize_backup

?>
/* Perform Euler step */
int iterate(Real time, const Real dt, Real* state)
{
    /* Set pacing variable */
    Real pace = fmod(time, STIM_PERIOD) - STIM_OFFSET;
    if ((pace >= 0 ) && (pace < STIM_DURATION)) {
        pace = STIM_LEVEL;
    } else {
        pace = 0.0;
    }

    /* Evaluate derivatives */
<?
for comp in comp_order:
    ilist = comp_in[comp]
    olist = comp_out[comp]

    # Skip components without output
    if len(olist) == 0:
        continue

    # Declare any output variables
    for var in comp_out[comp]:
        print(tab + 'Real ' + v(var) + ' = 0;')

    # Function header
    args = ['time', 'pace', 'state']
    args.extend([v(lhs) for lhs in ilist])
    args.extend(['&' + v(lhs) for lhs in olist])
    print(tab + 'calc_' + comp.qname() + '(' + ', '.join(args) + ');')
?>
    /* Perform update */
<?
for var in model.states():
    print(tab + v(var) + ' += dt * ' + v(var.lhs()) + ';')
?>

    return 0;
}

/* Run a simulation */
int main()
{
    Real state[N_STATE];
<?
for var in model.states():
    print(tab + v(var) + ' = ' + myokit.strfloat(var.state_value()) + ';')

?>

    long steps = 0;
    Real dt = 0.01;
    Real time = 0;
    while (time < 1000.0) {
        iterate(time, dt, state);
        time = dt * steps;
        if (steps % 100 == 0) {
            printf("%4.3f     %14.6le\n", time, <?= v(model.states().next()) ?>);
        }
        steps += 1;
    }

    return 0;
}
