<?
# cable.c
#
# A pype template for a cable simulation
#
# Required variables
# ---------------------------
# module_name A module name
# model       A myokit model
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
import myokit
import myokit.formats.ansic as ansic

# Get model
model.reserve_unique_names(*ansic.keywords)
model.create_unique_names()

# Get expression writer
w = ansic.AnsiCExpressionWriter()

# Process bindings, remove unsupported bindings
bound_variables = model.prepare_bindings({
    'time'         : 't',
    'pace'         : 'pace',
    'diffusion_current' : 'current'
    }).keys()
    
if model.binding('diffusion_current') is None:
    raise Exception('This exporter requires a variable to be bound to'
        ' "diffusion_current".')
    
# Define var/lhs function
def v(var, pre='cell->'):
    """
    Accepts a variable or a left-hand-side expression and returns its C
    representation.
    """
    if isinstance(var, myokit.Derivative):
        # Explicitly asked for derivative
        return pre + 'D_' + var.var().uname()
    if isinstance(var, myokit.Name):
        var = var.var()
    if var.is_state():
        return pre + 'S_' + var.uname()
    elif var.is_constant():
        return pre + 'C_' + var.uname()
    else:
        return pre + 'I_' + var.uname()
w.set_lhs_function(v)

# Get membrane potential
vmvar = model.label('membrane_potential')
if vmvar is None:
    raise Exception('This exporter requires a variable to be labelled as'
        ' "membrane_potential".')
vm = v(vmvar, pre='')

# Tab
tab = '    '

# Get equations
equations = model.solvable_order()
?>/*
Cable simulation
<?= model.name() ?>
Generated on <?= myokit.date() ?>

Compiling on GCC:
 $ gcc -Wall -lm cable.c

Gnuplot example:
set terminal pngcairo enhanced linewidth 2 size 1200, 800;
set output 'V.png'
set size 1.0, 1.0
set xlabel 'time [ms]';
set grid
plot 'V.txt' using 1:2 with lines ls 1 title 'Vm'

*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define R_CELL_TO_CELL 0.2

/*
 * Cell component
 */
typedef struct Cell {
<?
print(tab + '/* Bound variables */')
for var in model.variables(bound=True, deep=True):
    print(tab + 'double ' + v(var, pre='') + ';')
print(tab + '/* State variables */')
for var in model.states():
    print(tab + 'double ' + v(var, pre='') + ';')
print(tab + '/* State derivatives */')
for var in model.states():
    print(tab + 'double ' + v(var.lhs(), pre='') + ';')
print(tab + '/* Intermediary variables */')
for var in model.variables(inter=True, bound=False, deep=True):
    print(tab + 'double ' + v(var, pre='') + ';')
print(tab + '/* Constants */')
for var in model.variables(const=True, bound=False, deep=True):
    print(tab + 'double ' + v(var, pre='') + ';')
?>} Cell;

/*
 * Initializes all bound variables in a cell (sets them to 0)
 */
static void
Cell_initialize_bound_variables(Cell *cell)
{
<?
for var in bound_variables:
    print(tab + v(var) + ' = 0;')
?>
}

/*
 * Sets all its literal constants in a cell.
 */
static void
Cell_set_literals(Cell *cell)
{
<?
for label, eqs in equations.iteritems():
    for eq in eqs.equations(const=True):
        if eq.rhs.is_literal():
            print(tab + w.eq(eq) + ';')
?>
}

/*
 * Calculates calculated constants from the literal constants.
 */
static void
Cell_calculate_constants(Cell *cell)
{
<?
for label, eqs in equations.iteritems():
    for eq in eqs.equations(const=True):
        if not eq.rhs.is_literal():
            print(tab + w.eq(eq) + ';')
?>}

/*
 * Sets initial state values
 */
static void
Cell_set_initial_state(Cell *cell)
{
<?
for eq in model.inits():
    print(tab + w.eq(eq) + ';')
?>}

/*
 * Calculates derivatives in a cell
 */
static void
Cell_calculate_derivatives(Cell *cell)
{
<?
for label, eqs in equations.iteritems():
    for eq in eqs.equations(const=False, bound=False):
        print(tab + w.eq(eq) + ';')
?>}

/*
 * Updates a cell to the next time step
 */
static void
Cell_step(Cell *cell, double dt)
{
<?
for var in model.states():
    print(tab + v(var) + ' += dt * ' + v(var.lhs()) + ';')
?>}

/*
 * Pacing event structure
 */
typedef struct PacingEvent {
    double level;       /* The stimulus level (dimensionless, normal range [0,1]) */
    double start;       /* The time this stimulus starts */
    double duration;    /* The stimulus duration */
    double period;      /* The period with which it repeats (or 0 if it doesn't) */
    double multiplier;  /* The number of times this period occurs (or 0 if it doesn't) */
    struct PacingEvent *next;   /* The next event */
} PacingEvent;


/*
 * Inserts a pacing event into a linked list of events at the correct position
 * according to its starting tkime.
 *
 * Arguments:
 *   top    The first event in a stack (the stack's head)
 *   add    The event to schedule
 * Returns:
 *   The new pointer to the head of the stack
 */
PacingEvent*
PacingEvent_Schedule(PacingEvent* top, PacingEvent* add)
{
    add->next = 0;
    if (add == 0) return top;
    if (top == 0) return add;
    if (add->start <= top->start) {
        add->next = top;
        return add;
    }
    PacingEvent* evt = top;
    while(evt->next != 0 && evt->next->start <= add->start) {
        evt = evt->next;
    }
    add->next = evt->next;
    evt->next = add;
    return top;
}

/*
 * Runs a simulation
 */
int run(int n_cells, double time_min, double time_max, double time_step)
{
    int success = -1;

    /* Declare structures that will need deallocating */
    Cell *cells = NULL;
    PacingEvent *events = NULL;

    /* Global time & pacing */
    double time = time_min;
    double pace = 0;

    /* Allocate cells */
    int i_cell;
    cells = (Cell*)malloc(n_cells*sizeof(Cell));
    if (cells == 0) goto error;

    /* Initialize cells: set constants, initial values */
    Cell *cell, *clast, *cnext;
    cell = cells;
    for(i_cell=0; i_cell<n_cells; i_cell++) {
        Cell_initialize_bound_variables(cell);
        Cell_set_literals(cell);
        Cell_calculate_constants(cell);
        Cell_set_initial_state(cell);
        cell++;
    }

    /* Write log header */
    printf("time, pace");
    for(i_cell=0; i_cell<n_cells; i_cell++) {
<?
#for var in model.variables(const=False):
for var in model.states():
    print(tab + tab + 'printf(",%d_' + var.uname() + '", i_cell);')
?>    }
    printf("\n");

<?
nEvents = 0
next = protocol.head()
while next:
    nEvents += 1
    next = next.next()
?>    /* Create pacing events */
    int n_events = <?= nEvents ?>;
    events = (PacingEvent*)malloc(n_events*sizeof(PacingEvent));
    if (events == 0) goto error;
    int i_event = 0;
<?
next = protocol.head()
while next:
    print(tab + 'events[i_event].level = ' + str(next.level()) + ';')
    print(tab + 'events[i_event].start = ' + str(next.start()) + ';')
    print(tab + 'events[i_event].duration = ' + str(next.duration()) + ';')
    print(tab + 'events[i_event].period = ' + str(next.period()) + ';')
    print(tab + 'events[i_event].multiplier = ' + str(next.multiplier()) + ';')
    print(tab + 'i_event++;')
    next = next.next()
?>
    /* Schedule events, make "next" point to the first event */
    PacingEvent* next = events;
    PacingEvent* fire = events + 1;
    for(i_event=1; i_event<n_events; i_event++) {
        next = PacingEvent_Schedule(next, fire++);
    }

    /* Fast forward events to starting time */
    double time_next = next->start;
    double time_down = 0.0;
    fire = 0;
    while (time_next <= time_min) {
        /* Event over? */
        if (fire != 0 && time_next >= time_down) {
            fire = 0;
        }
        /* New event? */
        if (next != 0 && time_next >= next->start) {
            fire = next;
            next = next->next;
            time_down = fire->start + fire->duration;
            if (fire->period > 0) {
                if (fire->multiplier != 1) {
                    if (fire->multiplier > 1) fire->multiplier--;
                    fire->start += fire->period;
                    next = PacingEvent_Schedule(next, fire);
                } else {
                    fire->period = 0;
                }
            }
        }
        /* Set next time */
        time_next = time_max;
        if (fire != 0 && time_down < time_next) time_next = time_down;
        if (next != 0 && next->start < time_next) time_next = next->start;
    }
    if (fire != 0) {
        pace = fire->level;
    } else {
        pace = 0.0;
    }

    /* Logging interval */
    double log_interval = 1;
    double time_log = time_min;
    
    /* Current in */
    double diffusion_current = 0;

    /* Start simulation */
    time = time_min;
    while(time <= time_max) {

        /* Set membrane currents */
        cell = clast = cnext = cells;
        cnext++;
        
        /* Current, first cell */
        diffusion_current = R_CELL_TO_CELL * (cell-><?=vm?> - cnext-><?=vm?>);
<?
for var in bound_variables:
    if var.binding() == 'diffusion_current':
        print(tab*2 + v(var) + ' = diffusion_current;')
?>        cnext++;
        cell++;

        /* Current, doubly-connected cells */
        for(i_cell=2; i_cell<n_cells; i_cell++) {
            diffusion_current = R_CELL_TO_CELL * (2.0*cell-><?=vm?> - clast-><?=vm?> - cnext-><?=vm?>);
<?
for var in bound_variables:
    if var.binding() == 'diffusion_current':
        print(tab*3 + v(var) + ' = diffusion_current;')
?>            clast++;
            cell++;
            cnext++;        
        }
        
        /* Current, final cell */
        diffusion_current = R_CELL_TO_CELL * (cell-><?=vm?> - clast-><?=vm?>);
<?
for var in bound_variables:
    if var.binding() == 'diffusion_current':
        print(tab*2 + v(var) + ' = diffusion_current;')
?>
        /* Set pacing for first cell */
        cell = cells;
<?
for var in bound_variables:
    if var.binding() == 'pace':
        print(tab*2 + v(var) + ' = pace;')
?>
        /* Update time variables, calculate derivatives, take time step */
        for(i_cell=0; i_cell<n_cells; i_cell++) {
<?
for var in bound_variables:
    if var.binding() == 'time':
        print(tab*3 + v(var) + ' = time;')
?>
            Cell_calculate_derivatives(cell);
            Cell_step(cell, time_step);
            cell++;
        }

        /* Log */
        if (time >= time_log) {
            printf("%f, %f", time, pace);
            cell = cells;
            for(i_cell=0; i_cell<n_cells; i_cell++) {
<?
#for var in model.variables(const=False):
for var in model.states():
    print(4*tab + 'printf(",%f", ' + v(var, 'cell->') + ');')
?>                cell++;
            }
            printf("\n");
            time_log += log_interval;
        }

        /* Event over? */
        if (fire != 0 && time >= time_down) {
            pace = 0;
            fire = 0;
        }
        /* New event? */
        if (next != 0 && time >= next->start) {
            fire = next;
            next = next->next;
            pace = fire->level;
            time_down = fire->start + fire->duration;
            if (fire->period > 0) {
                if (fire->multiplier == 1) {
                    fire->period = 0;
                } else {
                    if (fire->multiplier > 1) fire->multiplier--;
                    fire->start += fire->period;
                    next = PacingEvent_Schedule(next, fire);
                }
            }
        }
        /* Set next time */
        time_next = time_max;
        if (fire != 0 && time_down < time_next) time_next = time_down;
        if (next != 0 && next->start < time_next) time_next = next->start;

        /* Next step */
        time += time_step;
    }

    /* Success! */
    success = 0;

error:
    /* Free allocated space */
    free(cells);
    free(events);

    /* Return */
    return success;
}

/*
 * Main function
 */
int main()
{
    return run(50, 0, 800, 0.005);
}
