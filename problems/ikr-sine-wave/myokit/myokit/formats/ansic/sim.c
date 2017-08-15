<?
#
# sim.c
# A pype template for an ansi C simulation with CVODE
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
import myokit
import myokit.formats.ansic as ansic

# Get model
model.reserve_unique_names(*ansic.keywords)
model.create_unique_names()

# Define lhs function
def v(var):
    if isinstance(var, myokit.Derivative):
        return 'NV_Ith_S(ydot, ' + str(var.var().indice()) + ')'
    elif isinstance(var, myokit.Name):
        var = var.var()
    if var.is_state():
        return 'NV_Ith_S(y, ' + str(var.indice()) + ')'
    elif var.is_constant():
        return 'AC_' + var.uname()
    else:
        return 'AV_' + var.uname()

# Create expression writer
w = ansic.AnsiCExpressionWriter()
w.set_lhs_function(v)

# Process bindings, remove unsupported bindings, get map of bound variables to
# internal names
bound_variables = model.prepare_bindings({
    'time' : 't',
    'pace' : 'pace',
    })

# Tab
tab = '    '

# Get equations
equations = model.solvable_order()

# Times
tmin = 0
tlog = 0
tmax = 1000

?>/*
<?= model.name() ?>
Generated on <?= myokit.date() ?>

Compiling on GCC:
 $ gcc -Wall -lm -lsundials_nvecserial -lsundials_cvode sim.c

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
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <cvode/cvode_dense.h>
#include <sundials/sundials_types.h>

#define N_STATE <?= model.count_states() ?>

/* Declare intermediary, temporary and system variables */
static realtype t;
static realtype pace;
<?
for var in model.variables(state=False, deep=True):
    print('static realtype ' + v(var) + ';')
?>
/* Set values of constants */
static void
updateConstants(void)
{
<?
for label, eqs in equations.iteritems():
    if eqs.has_equations(const=True):
        print(tab + '/* ' + label + ' */')
        for eq in eqs.equations(const=True):
            print(tab + w.eq(eq) + ';')
        print(tab)
?>}

/* Right-hand-side function of the model ODE */
static int rhs(realtype t, N_Vector y, N_Vector ydot, void *f_data)
{
<?
for label, eqs in equations.iteritems():
    if eqs.has_equations(const=False):
        print(tab + '/* ' + label + ' */')
        for eq in eqs.equations(const=False):
            var = eq.lhs.var()
            if var in bound_variables:
                print(tab + v(var) + ' = ' + bound_variables[var] + ';')
            else:
                print(tab + w.eq(eq) + ';')
        print(tab)
?>
    return 0;
}

/* Set initial values */
static void
default_initial_values(N_Vector y)
{
<?
for eq in model.inits():
    print(tab + w.eq(eq) + ';')
?>
}

/* Pacing event (non-zero stimulus) */
struct PacingEventS {
    double level;       /* The stimulus level (dimensionless, normal range [0,1]) */
    double start;       /* The time this stimulus starts */
    double duration;    /* The stimulus duration */
    double period;      /* The period with which it repeats (or 0 if it doesn't) */
    double multiplier;  /* The number of times this period occurs (or 0 if it doesn't) */
    struct PacingEventS* next;
};
typedef struct PacingEventS PacingEvent;

/*
 * Schedules a pacing event.
 * @param top The first event in a stack (the stack's head)
 * @param add The event to schedule
 * @return The new pointer to the head of the stack
 */
static PacingEvent*
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

/* CVODE Flags */
static int check_flag(void *flagvalue, char *funcname, int opt)
{
    int *errflag;
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
        return(1);
    } /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *) flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n", funcname, *errflag);
            return(1);
        }
    } /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
        return(1);
    }
    return 0;
}

/* Show output */
static void PrintOutput(realtype t, realtype y)
{
    #if defined(SUNDIALS_EXTENDED_PRECISION)
        printf("%4.1f     %14.6Le\n", t, y);
    #elif defined(SUNDIALS_DOUBLE_PRECISION)
        printf("%4.1f     %14.6le\n", t, y);
    #else
        printf("%4.1f     %14.6e\n", t, y);
    #endif
    return;
}

/* Run a simulation */
int main()
{
    int success = -1;

    /* Sundials error flag */
    int flag;

    /* Declare variables that will need freeing */
    PacingEvent* events = NULL;
    N_Vector y = NULL;
    N_Vector dy = NULL;
    void *cvode_mem = NULL;

    /* Create state vector */
    y = N_VNew_Serial(N_STATE);
    if (check_flag((void*)y, "N_VNew_Serial", 0)) goto error;
    dy = N_VNew_Serial(N_STATE);
    if (check_flag((void*)dy, "N_VNew_Serial", 0)) goto error;

    /* Set calculated constants */
    updateConstants();

    /* Set initial values */
    default_initial_values(y);

    /* Set integration times */
    double tMin = <?= tmin ?>;
    double tMax = <?= tmax ?>;
    double tLog = <?= tlog ?>;

    /* Create pacing events */
<?
nEvents = 0
next = protocol.head()
while next:
    nEvents += 1
    next = next.next()
?>
    int nPacing = <?= nEvents ?>;
    events = (PacingEvent*)malloc(sizeof(PacingEvent)*nPacing);
    if (events == 0) goto error;
    int iPacing = 0;
<?
next = protocol.head()
while next:
    print(tab + 'events[iPacing].level = ' + str(next.level()) + ';')
    print(tab + 'events[iPacing].start = ' + str(next.start()) + ';')
    print(tab + 'events[iPacing].duration = ' + str(next.duration()) + ';')
    print(tab + 'events[iPacing].period = ' + str(next.period()) + ';')
    print(tab + 'events[iPacing].multiplier = ' + str(next.multiplier()) + ';')
    print(tab + 'iPacing++;')
    next = next.next()
?>
    /* Schedule events, make "next" point to the first event */
    PacingEvent* next = events;
    PacingEvent* fire = events + 1;
    for(iPacing=1; iPacing<nPacing; iPacing++) {
        next = PacingEvent_Schedule(next, fire++);
    }

    /* Fast forward events to starting time */
    double tNext = next->start;
    double tDown = 0.0;
    fire = 0;
    while (tNext <= tMin) {
        /* Event over? */
        if (fire != 0 && tNext >= tDown) {
            fire = 0;
        }
        /* New event? */
        if (next != 0 && tNext >= next->start) {
            fire = next;
            next = next->next;
            tDown = fire->start + fire->duration;
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
        tNext = tMax;
        if (fire != 0 && tDown < tNext) tNext = tDown;
        if (next != 0 && next->start < tNext) tNext = next->start;
    }
    if (fire != 0) {
        pace = fire->level;
    } else {
        pace = 0.0;
    }

    /* Set simulation starting time */
    t = tMin;

    /* Create solver */
    cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
    if (check_flag((void*)cvode_mem, "CVodeCreate", 0)) goto error;
    flag = CVodeInit(cvode_mem, rhs, t, y);
    if (check_flag(&flag, "CVodeInit", 1)) goto error;
    flag = CVDense(cvode_mem, N_STATE);
    if (check_flag(&flag, "CVDense", 1)) goto error;

    /* Set tolerances */
    double reltol = RCONST(1.0e-4);
    double abstol = RCONST(1.0e-6);
    flag = CVodeSStolerances(cvode_mem, reltol, abstol);
    if (check_flag(&flag, "CVodeSStolerances", 1)) goto error;

    /* Go! */
    while(1) {
        if (tMax < tNext) tNext = tMax;
        flag = CVode(cvode_mem, tNext, y, &t, CV_ONE_STEP);
        if (check_flag(&flag, "CVode", 1)) break;
        if (flag == CV_SUCCESS) {
            /* Shot past next discontinuity? */
            if (t > tNext) {
                flag = CVodeGetDky(cvode_mem, tNext, 0, y);
                if (check_flag(&flag, "CVodeGetDky", 1)) goto error;
                t = tNext;
                flag = CVodeReInit(cvode_mem, t, y);
                if (check_flag(&flag, "CVodeReInit", 1)) goto error;
                /* Recalculate logging values at this point in time */
                rhs(t, y, dy, 0);
            }
            /* Event over? */
            if (fire != 0 && t >= tDown) {
                pace = 0;
                fire = 0;
                flag = CVodeReInit(cvode_mem, t, y);
                if (check_flag(&flag, "CVodeReInit", 1)) goto error;
            }
            /* New event? */
            if (next != 0 && t >= next->start) {
                fire = next;
                next = next->next;
                pace = fire->level;
                tDown = fire->start + fire->duration;
                if (fire->period > 0) {
                    if (fire->multiplier == 1) {
                        fire->period = 0;
                    } else {
                        if (fire->multiplier > 1) fire->multiplier--;
                        fire->start += fire->period;
                        next = PacingEvent_Schedule(next, fire);
                    }
                }
                flag = CVodeReInit(cvode_mem, t, y);
                if (check_flag(&flag, "CVodeReInit", 1)) goto error;
            }
            /* Set next time */
            tNext = tMax;
            if (fire != 0 && tDown < tNext) tNext = tDown;
            if (next != 0 && next->start < tNext) tNext = next->start;
            /* Log */
            if (t >= tLog) {
                /* Log current position */
                PrintOutput(t, NV_Ith_S(y, 0));
            }
        }
        if (t >= tMax) break;
    }

    /* Success! */
    success = 0;

error:
    /* Free allocated space */
    free(events);

    /* Free CVODE space */
    N_VDestroy_Serial(y);
    N_VDestroy_Serial(dy);
    CVodeFree(&cvode_mem);

    /* Return */
    return success;
}
