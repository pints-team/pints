/*
 * pacing.h
 *
 * Ansi-C implementation for event-based pacing (using a Myokit Protocol
 * object) and fixed-form pacing (using a time-series).
 *
 * How to use event-based pacing:
 *
 *  1. Create a pacing system using ESys_Create
 *  2. Populate it with events using ESys_Populate
 *  3. Set the time in the pacing system with ESys_AdvanceTime.
 *  4. Get the time of the first event with ESys_GetNextTime
 *  5. Get the initial pacing level with ESys_GetLevel
 *  6. Now at each step of a simulation
 *    - Advance the system to the simulation time with ESys_AdvanceTime
 *    - Get the time of the next event start or finish with ESys_GetNextTime
 *    - Get the pacing level using ESys_GetLevel
 *  7. Tidy up using ESys_Destroy
 *
 * Events must always start at t>=0, negative times are not supported.
 *
 * Flags are used to indicate errors. If a flag other than ESys_OK is set, a
 * call to ESys_SetPyErr(flag) can be made to set a Python exception.
 *
 *
 * How to use fixed-form pacing:
 *
 *  1. Create a pacing system using FSys_Create
 *  2. Populate it using two Python lists via FSys_Populate
 *  3. Obtain the pacing value for any time using FSys_GetLevel
 *  4. Tidy up using FSys_Destroy
 * 
 * This file is part of Myokit
 *  Copyright 2011-2017 Maastricht University
 *  Licensed under the GNU General Public License v3.0
 *  See: http://myokit.org
 *
 * Authors:
 *  Michael Clerx
 *
 */
#ifndef MyokitPacing
#define MyokitPacing

#include <Python.h>
#include <stdio.h>

/*
 * Event-based pacing error flags
 */
typedef int ESys_Flag;
#define ESys_OK                              0
#define ESys_OUT_OF_MEMORY                  -1
// General
#define ESys_INVALID_SYSTEM                 -10
#define ESys_POPULATED_SYSTEM               -11
#define ESys_UNPOPULATED_SYSTEM             -12
// ESys_Populate
#define ESys_POPULATE_INVALID_PROTOCOL      -20
#define ESys_POPULATE_MISSING_ATTR          -21
#define ESys_POPULATE_INVALID_ATTR          -22
#define ESys_POPULATE_NON_ZERO_MULTIPLIER   -23
#define ESys_POPULATE_NEGATIVE_PERIOD       -24
#define ESys_POPULATE_NEGATIVE_MULTIPLIER   -25
// ESys_AdvanceTime
#define ESys_NEGATIVE_TIME_INCREMENT        -40
// ESys_ScheduleEvent
#define ESys_SIMULTANEOUS_EVENT             -50

/*
 * Sets a python exception based on an event-based pacing error flag.
 *
 * Arguments
 *  flag : The python error flag to base the message on.
 */
void
ESys_SetPyErr(ESys_Flag flag)
{
    PyObject *module, *dict, *exception;
    switch(flag) {
    case ESys_OK:
        break;
    case ESys_OUT_OF_MEMORY:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Memory allocation failed.");
        break;
    // General
    case ESys_INVALID_SYSTEM:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Invalid pacing system provided.");
        break;
    case ESys_POPULATED_SYSTEM:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Pacing system already populated.");
        break;
    case ESys_UNPOPULATED_SYSTEM:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Pacing system not populated.");
        break;
    // ESys_ScheduleEvent
    case ESys_SIMULTANEOUS_EVENT:
        module = PyImport_ImportModule("myokit");   // New ref
        dict = PyModule_GetDict(module);            // Borrowed ref
        exception = PyDict_GetItemString(dict, "SimultaneousProtocolEventError");   // Borrowed ref
        PyErr_SetString(exception, "E-Pacing error: Event scheduled or re-occuring at the same time as another event.");
        Py_DECREF(module);
        break;
    // ESys_Populate
    case ESys_POPULATE_INVALID_PROTOCOL:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Protocol.events() failed to return a list.");
        break;
    case ESys_POPULATE_MISSING_ATTR:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Missing event attribute.");
        break;        
    case ESys_POPULATE_INVALID_ATTR:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Failed to convert event attribute to Float.");
        break;
    case ESys_POPULATE_NON_ZERO_MULTIPLIER:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Non-zero multiplier found for non-periodic stimulus.");
        break;
    case ESys_POPULATE_NEGATIVE_PERIOD:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Pacing event period cannot be negative.");
        break;
    case ESys_POPULATE_NEGATIVE_MULTIPLIER:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: Pacing event multiplier cannot be negative.");
        break;
    // ESys_AdvanceTime
    case ESys_NEGATIVE_TIME_INCREMENT:
        PyErr_SetString(PyExc_Exception, "E-Pacing error: New time is before current time.");
        break;
    // Unknown
    default:
    {
        int i = (int)flag;
        char buffer[1024];
        sprintf(buffer, "E-Pacing error: Unlisted error %d", i);
        PyErr_SetString(PyExc_Exception, buffer);
        break;
    }};
}

/*
 * Pacing event
 * 
 * Pacing event structs hold the information about a single pacing event. Using
 * the Event_Schedule function, pacing events can be ordered into an
 * event queue. Each event may appear only once in such a queue.
 *
 * Events have a starting time `start` at which they are "fired" and considered
 * "active" until a period of time `duration` has passed.
 *
 * Recurring events can be created by specifying a non-zero value of `period`.
 * The value `multiplier` is used to indicate how often an event should recur,
 * where 0 indicates the event repeats indefinitely.
 *
 * Recurring events are implemented as follows: once a recurring event has been
 * deactivated (at time `start` + `duration`), the event is removed from the
 * event queue. The `start` time and possible the `multiplier` are then updated
 * to the new values and the event is rescheduled back into the queue.
 */
struct EventMem {
    double level;       // The stimulus level (non-zero, dimensionless, normal range [0,1])
    double duration;    // The stimulus duration
    double start;       // The time this stimulus starts    
    double period;      // The period with which it repeats (or 0 if it doesn't)
    double multiplier;  // The number of times this period occurs (or 0 if it doesn't)
    double ostart;      // The event start set when the event was created
    double operiod;     // The period set when the event was created
    double omultiplier; // The multiplier set when the event was created
    struct EventMem* next;
};
#define Event struct EventMem*

/*
 * Adds an event to an event queue.
 *
 * Arguments
 *  head  : The head of the event queue
 *  event : The event to schedule
 *  flag : The address of a pacing error flag or NULL
 *
 * Returns the new head of the event queue
 */
static Event
ESys_ScheduleEvent(Event head, Event add, ESys_Flag* flag)
{
    Event e;    // Needs to be declared here for visual C
    *flag = ESys_OK;
    add->next = 0;
    if (add == 0) return head;
    if (head == 0) return add;
    if (add->start < head->start) {
        add->next = head;
        return add;
    }
    e = head;
    while(e->next != 0 && add->start >= e->next->start) {
        e = e->next;
    }
    if (add->start == e->start) {
        *flag = ESys_SIMULTANEOUS_EVENT;
    }
    add->next = e->next;
    e->next = add;
    return head;
}

/*
 * Pacing system 
 */
struct ESys_Mem {
    double time;    // The current time
    int n_events;   // The number of events in this system
    Event events;   // The events, stored as an array
    Event head;     // The head of the event queue
    Event fire;     // The currently active event
    double tnext;   // The time of the next event start or finish
    double tdown;   // The time the active event is over
    double level;   // The current output value
};
typedef struct ESys_Mem* ESys;

/*
 * Creates a pacing system
 *
 * Arguments
 *  flag : The address of an event-based pacing error flag or NULL
 *
 * Returns the newly created pacing system
 */
ESys
ESys_Create(ESys_Flag* flag)
{
    ESys sys = (ESys)malloc(sizeof(struct ESys_Mem));
    if (sys == 0) {
        if(flag != 0) *flag = ESys_OUT_OF_MEMORY;
        return 0;
    }
    
    sys->time = 0;
    sys->n_events = -1; // Used to indicate unpopulated system
    sys->events = NULL;
    sys->head = NULL;
    sys->fire = NULL;
    sys->tnext = 0;
    sys->tdown = 0;
    sys->level = 0;
    
    if(flag != 0) *flag = ESys_OK;
    return sys;
}

/*
 * Destroys a pacing system and frees the memory it occupies.
 *
 * Arguments
 *  sys : The event-based pacing system to destroy
 *
 * Returns a pacing error flag.
 */
ESys_Flag
ESys_Destroy(ESys sys)
{
    if(sys == NULL) return ESys_INVALID_SYSTEM;
    if(sys->events != NULL) {
        free(sys->events);
        sys->events = NULL;
    }
    free(sys);
    return ESys_OK;
}

/*
 * Resets this pacing system to time=0.
 *
 * Arguments
 *  sys : The event-based pacing system to reset
 *
 * Returns a pacing error flag.
 */
ESys_Flag
ESys_Reset(ESys sys)
{
    Event next;     // Need to be declared here for C89 Visual C
    Event head;
    int i;
    
    if(sys == 0) return ESys_INVALID_SYSTEM;
    if(sys->n_events < 0) return ESys_UNPOPULATED_SYSTEM;

    // Reset all events
    next = sys->events;
    for(i=0; i<sys->n_events; i++) {
        next->start = next->ostart;
        next->period = next->operiod;
        next->multiplier = next->omultiplier;
        next->next = 0;
    }

    // Set up the event queue
    ESys_Flag flag;
    head = sys->events;
    next = head + 1;
    for(i=1; i<sys->n_events; i++) {
        head = ESys_ScheduleEvent(head, next++, &flag);
        if (flag != ESys_OK) { return flag; }
    }
    
    // Reset the properties of the event system
    sys->time = 0;
    sys->head = head;
    sys->fire = 0;
    sys->tnext = 0;
    sys->tdown = 0;
    sys->level = 0;
    
    return ESys_OK;
}

/*
 * Populates an event system using the events from a myokit.Protocol
 * Returns an error if the system already contains events.
 *
 * Arguments
 *  sys      : The pacing system to schedule the events in.
 *  protocol : A pacing protocol or NULL
 *
 * Returns a pacing error flag.
 */
ESys_Flag
ESys_Populate(ESys sys, PyObject* protocol)
{
    int i;
    int n;
    Event events;
    Event e;
    
    if(sys == 0) return ESys_INVALID_SYSTEM;
    if (sys->n_events != -1) return ESys_POPULATED_SYSTEM;

    // Default values    
    n = 0;
    events = 0;
    
    if (protocol != Py_None) {
    
        // Get PyList from protocol
        // Cast to (char*) happens because CallMethod accepts a mutable char*
        // This should have been const char* and has been fixed in python 3
        PyObject* list = PyObject_CallMethod(protocol, (char*)"events", NULL);
        if(list == NULL) return ESys_POPULATE_INVALID_PROTOCOL;
        if(!PyList_Check(list)) return ESys_POPULATE_INVALID_PROTOCOL;
        n = (int)PyList_Size(list);
        
        // Translate python pacing events
        // Note: A lot of the tests here shouldn't really make a difference,
        // since they are tested by the Python code already!
        if(n > 0) {
            PyObject *item, *attr;        
            events = (Event)malloc(n*sizeof(struct EventMem));
            e = events;
            for(i=0; i<n; i++) {
                item = PyList_GetItem(list, i); // Don't decref!
                // Level
                attr = PyObject_GetAttrString(item, "_level");
                if (attr == NULL) { // Not a string
                    free(events); Py_DECREF(list);
                    return ESys_POPULATE_MISSING_ATTR;
                }
                e->level = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events); Py_DECREF(list);
                    return ESys_POPULATE_INVALID_ATTR;
                }
                // duration
                attr = PyObject_GetAttrString(item, "_duration");
                if (attr == NULL) {
                    free(events);
                    Py_DECREF(list);
                    return ESys_POPULATE_MISSING_ATTR;
                }
                e->duration = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events); Py_DECREF(list);
                    return ESys_POPULATE_INVALID_ATTR;
                }
                // start
                attr = PyObject_GetAttrString(item, "_start");
                if (attr == NULL) {
                    free(events); Py_DECREF(list);
                    return ESys_POPULATE_MISSING_ATTR;
                }
                e->start = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events); Py_DECREF(list);
                    return ESys_POPULATE_INVALID_ATTR;
                }
                // Period
                attr = PyObject_GetAttrString(item, "_period");
                if (attr == NULL) {
                    free(events); Py_DECREF(list);
                    return ESys_POPULATE_MISSING_ATTR;
                }
                e->period = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events); Py_DECREF(list);
                    return ESys_POPULATE_INVALID_ATTR; 
                }
                // multiplier
                attr = PyObject_GetAttrString(item, "_multiplier");
                if (attr == NULL) {
                    free(events); Py_DECREF(list);
                    return ESys_POPULATE_MISSING_ATTR;
                }
                e->multiplier = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events); Py_DECREF(list);
                    return ESys_POPULATE_INVALID_ATTR;
                }
                // Original values
                e->ostart = e->start;
                e->operiod = e->period;
                e->omultiplier = e->multiplier;
                e->next = 0;
                if (e->period == 0 && e->multiplier != 0) {
                    free(events);
                    return ESys_POPULATE_NON_ZERO_MULTIPLIER;
                }
                if (e->period < 0) {
                    free(events);
                    return ESys_POPULATE_NEGATIVE_PERIOD;
                }
                if (e->multiplier < 0) {
                    free(events);
                    return ESys_POPULATE_NEGATIVE_MULTIPLIER;
                }
                e++;
            }
        }
    }
    
    // Add the events to the system
    sys->n_events = n;
    sys->events = events;

    // Set all remaining properties using reset
    return ESys_Reset(sys);
}

/*
 * Advances the pacing system to the next moment in time.
 *
 * Arguments
 *  sys      : The pacing system to advance.
 *  new_time : The time to increment the system to. Must be more than or equal
 *             to the current pacing system time.
 *  max_time : The maximum time to advance to.
 *
 * Returns a pacing error flag.
 */
ESys_Flag
ESys_AdvanceTime(ESys sys, double new_time, double max_time)
{
    if(sys == 0) return ESys_INVALID_SYSTEM;
    if(sys->n_events < 0) return ESys_UNPOPULATED_SYSTEM;
    if(sys->time > new_time) return ESys_NEGATIVE_TIME_INCREMENT;
    
    // Update internal time
    sys->time = new_time;
    if (new_time > max_time) max_time = new_time;
    
    // Advance
    ESys_Flag flag;
    while (sys->tnext <= sys->time && sys->tnext < max_time) {
        // Active event finished
        if (sys->fire != 0 && sys->tnext >= sys->tdown) {
            sys->fire = 0;
            sys->level = 0;
        }
        // New event starting
        if (sys->head != 0 && sys->tnext >= sys->head->start) {
            sys->fire = sys->head;
            sys->head = sys->head->next;
            sys->tdown = sys->fire->start + sys->fire->duration;
            sys->level = sys->fire->level;
            // Reschedule recurring event
            if (sys->fire->period > 0) {
                if (sys->fire->multiplier != 1) {
                    if (sys->fire->multiplier > 1) sys->fire->multiplier--;
                    sys->fire->start += sys->fire->period;
                    sys->head = ESys_ScheduleEvent(sys->head, sys->fire, &flag);
                    if (flag != ESys_OK) { return flag; }
                } else {
                    sys->fire->period = 0;
                }
            }
        }
        // Set next stopping time
        sys->tnext = max_time;
        if (sys->fire != 0 && sys->tnext > sys->tdown)
            sys->tnext = sys->tdown;
        if (sys->head != 0 && sys->tnext > sys->head->start)
            sys->tnext = sys->head->start;
    }
    return ESys_OK;
}

/*
 * Returns the next time a pacing event starts or finishes in the given system.
 *
 * Arguments
 *  sys : The pacing system to query for a time
 *  flag : The address of a pacing error flag or NULL
 *
 * Returns the next time a pacing event starts or finishes
 */
double
ESys_GetNextTime(ESys sys, ESys_Flag* flag)
{
    if(sys == 0) {
        if(flag != 0) *flag = ESys_INVALID_SYSTEM;
        return -1;
    }
    if(sys->n_events < 0) {
        if(flag != 0) *flag = ESys_UNPOPULATED_SYSTEM;
        return -1;
    }
    if(flag != 0) *flag = ESys_OK;
    return sys->tnext;
}

/*
 * Returns the current pacing level.
 *
 * Arguments
 *  sys : The pacing system to query for a time
 *  flag : The address of a pacing error flag or NULL
 *
 * Returns the next time a pacing event starts or finishes
 */
double
ESys_GetLevel(ESys sys, ESys_Flag* flag)
{
    if(sys == 0) {
        if(flag != 0) *flag = ESys_INVALID_SYSTEM;
        return -1;
    }
    if(sys->n_events < 0) {
        if(flag != 0) *flag = ESys_UNPOPULATED_SYSTEM;
        return -1;
    }
    if(flag != 0) *flag = ESys_OK;
    return sys->level;
}

/*
 *
 * Fixed-form code starts here
 *
 */

/*
 * Fixed-form pacing error flags
 */
typedef int FSys_Flag;
#define FSys_OK                             0
#define FSys_OUT_OF_MEMORY                  -1
// General
#define FSys_INVALID_SYSTEM                 -10
#define FSys_POPULATED_SYSTEM               -11
#define FSys_UNPOPULATED_SYSTEM             -12
// Populating the system
#define FSys_POPULATE_INVALID_TIMES         -20
#define FSys_POPULATE_INVALID_VALUES        -21
#define FSys_POPULATE_SIZE_MISMATCH         -22
#define FSys_POPULATE_NOT_ENOUGH_DATA       -23
#define FSys_POPULATE_INVALID_TIMES_DATA    -24
#define FSys_POPULATE_INVALID_VALUES_DATA   -25
#define FSys_POPULATE_DECREASING_TIMES_DATA -26

/*
 * Sets a python exception based on a fixed-form pacing error flag.
 *
 * Arguments
 *  flag : The python error flag to base the message on.
 */
void
FSys_SetPyErr(FSys_Flag flag)
{
    switch(flag) {
    case FSys_OK:
        break;
    case FSys_OUT_OF_MEMORY:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Memory allocation failed.");
        break;
    // General
    case FSys_INVALID_SYSTEM:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Invalid pacing system provided.");
        break;
    case FSys_POPULATED_SYSTEM:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Pacing system already populated.");
        break;
    case FSys_UNPOPULATED_SYSTEM:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Pacing system not populated.");
        break;
    // Populate
    case FSys_POPULATE_INVALID_TIMES:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Invalid times array passed.");
        break;
    case FSys_POPULATE_INVALID_VALUES:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Invalid values array passed.");
        break;
    case FSys_POPULATE_SIZE_MISMATCH:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Sizes of times and values arrays don't match.");
        break;
    case FSys_POPULATE_NOT_ENOUGH_DATA:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Time-series must contain at least two data points.");
        break;
    case FSys_POPULATE_INVALID_TIMES_DATA:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Times array must contain only floats.");
        break;
    case FSys_POPULATE_INVALID_VALUES_DATA:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Values array must contain only floats.");
        break;
    case FSys_POPULATE_DECREASING_TIMES_DATA:
        PyErr_SetString(PyExc_Exception, "F-Pacing error: Times array must be non-decreasing.");
        break;        
    // Unknown
    default:
    {
        int i = (int)flag;
        char buffer[1024];
        sprintf(buffer, "F-Pacing error: Unlisted error %d", i);
        PyErr_SetString(PyExc_Exception, buffer);
        break;
    }};
}

/*
 * Fixed-form pacing system
 */
struct FSys_Mem {
    int n_points;   // The number of entries in the time and pace arrays
    double* times;  // The time array
    double* values; // The values array
    int last_index; // The index of the most recently returned value
    //double level;   // The current output value
};
typedef struct FSys_Mem* FSys;

/*
 * Creates a fixed-form pacing system
 *
 * Arguments
 *  flag : The address of a fixed-form pacing error flag or NULL
 *
 * Returns the newly created fixed-form pacing system
 */
FSys
FSys_Create(FSys_Flag* flag)
{
    FSys sys = (FSys)malloc(sizeof(struct FSys_Mem));
    if (sys == 0) {
        if(flag != 0) *flag = FSys_OUT_OF_MEMORY;
        return 0;
    }
    
    sys->n_points = -1;
    sys->times = NULL;
    sys->values = NULL;
    sys->last_index = 0;
        
    if(flag != 0) *flag = FSys_OK;
    return sys;
}

/*
 * Destroys a fixed-form pacing system and frees the memory it occupies.
 *
 * Arguments
 *  sys : The fixed-form pacing system to destroy
 *
 * Returns a fixed-form pacing error flag.
 */
FSys_Flag
FSys_Destroy(FSys sys)
{
    if(sys == 0) return FSys_INVALID_SYSTEM;
    if(sys->times != NULL) {
        free(sys->times);
        sys->times = NULL;
    }
    if(sys->values != NULL) {
        free(sys->values);
        sys->values = NULL;
    }
    free(sys);
    return FSys_OK;
}

/*
 * Populates a fixed-form pacing system using two Python list objects
 * containing an equal number of floating point numbers.
 * Returns an error if the system already has data.
 *
 * Arguments
 *  sys    : The fixed-form pacing system to add the data to.
 *  times  : A Python list of (non-decreasing) floats.
 *  values : An equally sized Python list of floats.
 *
 * Returns a fixed-form pacing error flag.
 */
FSys_Flag
FSys_Populate(FSys sys, PyObject* times_list, PyObject* values_list)
{
    int i;
    int n;

    // Check ESys
    if(sys == 0) return FSys_INVALID_SYSTEM;
    if (sys->n_points != -1) return FSys_POPULATED_SYSTEM;
    
    // Check input lists
    if(!PyList_Check(times_list)) return FSys_POPULATE_INVALID_TIMES;
    if(!PyList_Check(values_list)) return FSys_POPULATE_INVALID_VALUES;
    n = PyList_Size(times_list);
    if (n != PyList_Size(values_list)) return FSys_POPULATE_SIZE_MISMATCH;
    if (n < 2) return FSys_POPULATE_NOT_ENOUGH_DATA;
    
    // Convert and check times list
    sys->times = (double*)malloc(n*sizeof(double));
    for(i=0; i<n; i++) {
        // GetItem and convert --> Borrowed reference so ok not to decref!
        sys->times[i] = PyFloat_AsDouble(PyList_GetItem(times_list, i));
    }
    if (PyErr_Occurred()) {
        free(sys->times); sys->times = NULL;
        return FSys_POPULATE_INVALID_TIMES_DATA;
    }
    for(i=1; i<n; i++) {
        if(sys->times[i] < sys->times[i-1]) {
            free(sys->times); sys->times = NULL;
            return FSys_POPULATE_DECREASING_TIMES_DATA;
        }
    }
    
    // Convert values list
    sys->values = (double*)malloc(n*sizeof(double));
    for(i=0; i<n; i++) {
        // GetItem and convert --> Borrowed reference so ok not to decref!
        sys->values[i] = PyFloat_AsDouble(PyList_GetItem(values_list, i));
    }
    if (PyErr_Occurred()) {
        free(sys->times); sys->times = NULL;
        free(sys->values); sys->values = NULL;
        return FSys_POPULATE_INVALID_VALUES_DATA;
    }
    
    // Update pacing system and return
    sys->n_points = n;
    sys->last_index = 0;
    return FSys_OK;
}

/*
 * Returns the pacing level at the given time.
 *
 * Arguments
 *  sys : The pacing system to query for a value.
 *  time : The time to find a value for.
 *  flag : The address of a pacing error flag or NULL.
 *
 * Returns the value of the pacing level at the given time.
 * Will return -1 if an error occurs, so errors should always be checked for
 * using the flag argument!
 */
double
FSys_GetLevel(FSys sys, double time, FSys_Flag* flag)
{
    // Index and time at left, mid and right point, plus guessed point
    int    ileft, imid, iright, iguess;
    double tleft, tmid, tright, tguess;
    double vleft;

    // Check system
    if(sys == 0) {
        if(flag != 0) *flag = FSys_INVALID_SYSTEM;
        return -1;
    }
    if(sys->n_points < 0) {
        if(flag != 0) *flag = FSys_UNPOPULATED_SYSTEM;
        return -1;
    }
    
    // Find the highest index `i` of sorted array `times` such that
    // `times[i] <= time`, or `-1` if no such indice can be found.
    // A guess can be given, which will be used to speed things up

    // Get left point, check value
    ileft = 0;
    tleft = sys->times[ileft];
    if (tleft > time) {
        // Out-of-bounds on the left, return left-most value
        if(flag != 0) *flag = FSys_OK;
        return sys->values[ileft];
    }
    
    // Get right point, check value
    iright = sys->n_points - 1;
    tright = sys->times[iright];
    if (tright <= time) {
        // Out-of-bounds on the right, return right-most value
        if(flag != 0) *flag = FSys_OK;
        return sys->values[iright];
    }
    
    // Have a quick guess at better boundaries, using last
    iguess = sys->last_index - 1; // -1 is heuristic! Could be smaller
    if (iguess > ileft) {
        tguess = sys->times[iguess];
        if (tguess <= time) {
            ileft = iguess;
            tleft = tguess;
        }
    }
    iguess = sys->last_index + 2;   // +2 is heuristic!
    if (iguess < iright) {
        tguess = sys->times[iguess];
        if (tguess > time) {
            iright = iguess;
            tright = tguess;
        }
    }
    
    // Start bisection
    imid = ileft + (iright - ileft) / 2;
    while (ileft != imid) {
        tmid = sys->times[imid];
        if (tmid < time) {
            ileft = imid;
            tleft = tmid;      
        } else {
            iright = imid;
            tright = tmid;
        }
        imid = ileft + (iright - ileft) / 2;
    }
    
    // At this stage, tleft < time <= tright
    
    // Handle special case of time == tright
    // (Because otherwise it can happen that tleft == tright, which would give
    //  a divide-by-zero in the interpolateion)
    if (time == tright) {
        if(flag != 0) *flag = FSys_OK;
        sys->last_index = iright;
        return sys->values[iright];
    }
    
    // Find the correct value using linear interpolation
    if(flag != 0) *flag = FSys_OK;
    sys->last_index = ileft;
    vleft = sys->values[ileft];
    return vleft + (sys->values[iright] - vleft) * (time - tleft) / (tright - tleft);
}

#endif
