/*
 * pacing.h
 * 
 * Implements the myokit pacing protocol in ansi C with Python wrappers.
 *
 * How to use:
 *  1. Create a pacing system using PSys_Create
 *  2. Populate it with events using PSys_PopulateFromPyList
 *  3. Set the time in the pacing system with PSys_AdvanceTime.
 *  4. Get the time of the first event with PSys_GetNextTime
 *  5. Get the initial pacing level with PSys_GetLevel
 *  6. Now at each step of a simulation
 *    - Advance the system to the simulation time with PSys_AdvanceTime
 *    - Get the time of the next event start or finish with PSys_GetNextTime
 *    - Get the pacing level using PSys_GetLevel
 *
 * Events must always start at t>=0, negative times are not supported.
 * 
 * This file is part of Myokit
 *  Copyright 2011-2016 Michael Clerx, Maastricht University
 *  Licensed under the GNU General Public License v3.0
 *  See: http://myokit.org
 * 
 */
#ifndef MyokitPacing
#define MyokitPacing

#include <Python.h>
#include <stdio.h>

/*
 * Pacing error flags
 */
typedef int PSys_Flag;
#define PSys_OK                              0
#define PSys_OUT_OF_MEMORY                  -1
// General
#define PSys_INVALID_SYSTEM                 -10
#define PSys_POPULATED_SYSTEM               -11
#define PSys_UNPOPULATED_SYSTEM             -12
// PSys_Populate
#define PSys_POPULATE_INVALID_PROTOCOL      -20
#define PSys_POPULATE_MISSING_ATTR          -21
#define PSys_POPULATE_INVALID_ATTR          -22
#define PSys_POPULATE_NON_ZERO_MULTIPLIER   -23
#define PSys_POPULATE_NEGATIVE_PERIOD       -24
#define PSys_POPULATE_NEGATIVE_MULTIPLIER   -25
// PSys_AdvanceTime
#define PSys_NEGATIVE_TIME_INCREMENT        -40
// PSys_ScheduleEvent
#define PSys_SIMULTANEOUS_EVENT             -50

/*
 * Sets a python exception based on a pacing error flag.
 *
 * Arguments:
 *  flag : The python error flag to base the message on.
 */
void
PSys_SetPyErr(PSys_Flag flag)
{
    PyObject *module, *dict, *exception;
    switch(flag) {
    case PSys_OK:
        break;
    case PSys_OUT_OF_MEMORY:
        PyErr_SetString(PyExc_Exception, "Pacing error: Memory allocation failed.");
        break;
    case PSys_INVALID_SYSTEM:
        PyErr_SetString(PyExc_Exception, "Pacing error: Invalid pacing system provided.");
        break;
    case PSys_POPULATED_SYSTEM:
        PyErr_SetString(PyExc_Exception, "Pacing error: Pacing system already populated.");
        break;
    case PSys_UNPOPULATED_SYSTEM:
        PyErr_SetString(PyExc_Exception, "Pacing error: Pacing system not populated.");
        break;
    // PSys_ScheduleEvent
    case PSys_SIMULTANEOUS_EVENT:
        module = PyImport_ImportModule("myokit");   // New ref
        dict = PyModule_GetDict(module);            // Borrowed ref
        exception = PyDict_GetItemString(dict, "SimultaneousProtocolEventError");   // Borrowed ref
        PyErr_SetString(exception, "Pacing error: Event scheduled or re-occuring at the same time as another event.");
        Py_DECREF(module);
        break;
    // PSys_Populate
    case PSys_POPULATE_INVALID_PROTOCOL:
        PyErr_SetString(PyExc_Exception, "Pacing error: Protocol.events() failed to return a list.");
        break;
    case PSys_POPULATE_MISSING_ATTR:
        PyErr_SetString(PyExc_Exception, "Pacing error: Missing event attribute.");
        break;        
    case PSys_POPULATE_INVALID_ATTR:
        PyErr_SetString(PyExc_Exception, "Pacing error: Failed to convert event attribute to double.");
        break;
    case PSys_POPULATE_NON_ZERO_MULTIPLIER:
        PyErr_SetString(PyExc_Exception, "Pacing error: Non-zero multiplier found for non-periodic stimulus.");
        break;
    case PSys_POPULATE_NEGATIVE_PERIOD:
        PyErr_SetString(PyExc_Exception, "Pacing error: Pacing event period cannot be negative.");
        break;
    case PSys_POPULATE_NEGATIVE_MULTIPLIER:
        PyErr_SetString(PyExc_Exception, "Pacing error: Pacing event multiplier cannot be negative.");
        break;
    // PSys_AdvanceTime
    case PSys_NEGATIVE_TIME_INCREMENT:
        PyErr_SetString(PyExc_Exception, "Pacing error: New time is before current time.");
        break;
    // Unknown
    default:
    {
        int i = (int)flag;
        char buffer[1024];
        sprintf(buffer, "Pacing error: Unlisted error %d", i);
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
PSys_ScheduleEvent(Event head, Event add, PSys_Flag* flag)
{
    Event e;    // Needs to be declared here for visual C
    *flag = PSys_OK;
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
        *flag = PSys_SIMULTANEOUS_EVENT;
    }
    add->next = e->next;
    e->next = add;
    return head;
}

/* *  flag : The address of a pacing error flag or NULL
 * Pacing system 
 */
struct PSys_Mem {
    double time;    // The current time
    int n_events;   // The number of events in this system
    Event events;   // The events, stored as an array
    Event head;     // The head of the event queue
    Event fire;     // The currently active event
    double tnext;   // The time of the next event start or finish
    double tdown;   // The time the active event is over
    double level;   // The current output value
};
typedef struct PSys_Mem* PSys;

/*
 * Creates a pacing system
 *
 * Arguments
 *  flag : The address of a pacing error flag or NULL
 *
 * Returns the newly created pacing system
 */
PSys
PSys_Create(PSys_Flag* flag)
{
    PSys sys = (PSys)malloc(sizeof(struct PSys_Mem));
    if (sys == 0) {
        if(flag != 0) *flag = PSys_OUT_OF_MEMORY;
        return 0;
    }
    
    sys->time = 0;
    sys->n_events = -1; // Used to indicate unpopulated system
    sys->events = 0;
    sys->head = 0;
    sys->fire = 0;    
    sys->tnext = 0;
    sys->tdown = 0;
    sys->level = 0;
    
    if(flag != 0) *flag = PSys_OK;
    return sys;
}

/*
 * Destroys a pacing system and frees the memory it occupies.
 * Arguments:
 *  sys : The pacing system to destroy
 * Returns a pacing error flag.
 */
PSys_Flag
PSys_Destroy(PSys sys)
{
    if(sys == 0) return PSys_INVALID_SYSTEM;
    if(sys->events != 0) {
        free(sys->events);
        sys->events = NULL;
    }
    free(sys);
    return PSys_OK;
}

/*
 * Resets this pacing system to time=0.
 *
 * Arguments:
 *  sys : The pacing system to reset
 *
 * Returns a pacing error flag.
 */
PSys_Flag
PSys_Reset(PSys sys)
{
    Event next;     // Need to be declared here for C89 Visual C
    Event head;
    int i;
    
    if(sys == 0) return PSys_INVALID_SYSTEM;
    if(sys->n_events < 0) return PSys_UNPOPULATED_SYSTEM;

    // Reset all events
    next = sys->events;
    for(i=0; i<sys->n_events; i++) {
        next->start = next->ostart;
        next->period = next->operiod;
        next->multiplier = next->omultiplier;
        next->next = 0;
    }

    // Set up the event queue
    PSys_Flag flag;
    head = sys->events;
    next = head + 1;
    for(i=1; i<sys->n_events; i++) {
        head = PSys_ScheduleEvent(head, next++, &flag);
        if (flag != PSys_OK) { return flag; }
    }
    
    // Reset the properties of the event system
    sys->time = 0;
    sys->head = head;
    sys->fire = 0;
    sys->tnext = 0;
    sys->tdown = 0;
    sys->level = 0;
    
    return PSys_OK;
}

/*
 * Populates an event system using the events from a myokit.Protocol
 * Returns an error if the system already contains events.
 *
 * Arguments:
 *  sys      : The pacing system to schedule the events in.
 *  protocol : A pacing protocol or NULL
 *
 * Returns a pacing error flag.
 */
PSys_Flag
PSys_Populate(PSys sys, PyObject* protocol)
{
    int i;
    int n;
    Event events;
    Event e;
    
    if(sys == 0) return PSys_INVALID_SYSTEM;
    if (sys->n_events != -1) return PSys_POPULATED_SYSTEM;

    // Default values    
    n = 0;
    events = 0;
    
    if (protocol != Py_None) {
    
        // Get PyList from protocol
        // Cast to (char*) happens because CallMethod accepts a mutable char*
        // This should have been const char* and has been fixed in python 3
        PyObject* list = PyObject_CallMethod(protocol, (char*)"events", NULL);
        if(list == NULL) return PSys_POPULATE_INVALID_PROTOCOL;
        if(!PyList_Check(list)) return PSys_POPULATE_INVALID_PROTOCOL;
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
                if (attr == NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_MISSING_ATTR; }
                e->level = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_INVALID_ATTR; }
                // duration
                attr = PyObject_GetAttrString(item, "_duration");
                if (attr == NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_MISSING_ATTR; }
                e->duration = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_INVALID_ATTR; }
                // start
                attr = PyObject_GetAttrString(item, "_start");
                if (attr == NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_MISSING_ATTR; }
                e->start = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_INVALID_ATTR; }
                // Period
                attr = PyObject_GetAttrString(item, "_period");
                if (attr == NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_MISSING_ATTR; }
                e->period = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_INVALID_ATTR; }
                // multiplier
                attr = PyObject_GetAttrString(item, "_multiplier");
                if (attr == NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_MISSING_ATTR; }
                e->multiplier = PyFloat_AsDouble(attr);
                Py_DECREF(attr); attr = NULL;
                if (PyErr_Occurred() != NULL) {
                    free(events);
                    Py_DECREF(list);
                    return PSys_POPULATE_INVALID_ATTR; }
                // Original values
                e->ostart = e->start;
                e->operiod = e->period;
                e->omultiplier = e->multiplier;
                e->next = 0;
                if (e->period == 0 && e->multiplier != 0) {
                    free(events);
                    return PSys_POPULATE_NON_ZERO_MULTIPLIER;
                }
                if (e->period < 0) {
                    free(events);
                    return PSys_POPULATE_NEGATIVE_PERIOD;
                }
                if (e->multiplier < 0) {
                    free(events);
                    return PSys_POPULATE_NEGATIVE_MULTIPLIER;
                }
                e++;
            }
        }
    }
    
    // Add the events to the system
    sys->n_events = n;
    sys->events = events;

    // Set all remaining properties using reset
    return PSys_Reset(sys);
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
PSys_Flag
PSys_AdvanceTime(PSys sys, double new_time, double max_time)
{
    if(sys == 0) return PSys_INVALID_SYSTEM;
    if(sys->n_events < 0) return PSys_UNPOPULATED_SYSTEM;
    if(sys->time > new_time) return PSys_NEGATIVE_TIME_INCREMENT;
    
    // Update internal time
    sys->time = new_time;
    if (new_time > max_time) max_time = new_time;
    
    // Advance
    PSys_Flag flag;
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
                    sys->head = PSys_ScheduleEvent(sys->head, sys->fire, &flag);
                    if (flag != PSys_OK) { return flag; }
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
    return PSys_OK;
}

/*
 * Returns the next time a pacing event starts or finishes in the given system.
 *
 * Arguments
 *  sys : The pacing system to query for a time
 *  flag : The address of a pacing error flag or NULL
 * Returns the next time a pacing event starts or finishes
 */
double
PSys_GetNextTime(PSys sys, PSys_Flag* flag)
{
    if(sys == 0) {
        if(flag != 0) *flag = PSys_INVALID_SYSTEM;
        return -1;
    }
    if(sys->n_events < 0) {
        if(flag != 0) *flag = PSys_UNPOPULATED_SYSTEM;
        return -1;
    }
    if(flag != 0) *flag = PSys_OK;
    return sys->tnext;
}

/*
 * Returns the current pacing level.
 *
 * Arguments
 *  sys : The pacing system to query for a time
 *  flag : The address of a pacing error flag or NULL
 * Returns the next time a pacing event starts or finishes
 */
double
PSys_GetLevel(PSys sys, PSys_Flag* flag)
{
    if(sys == 0) {
        if(flag != 0) *flag = PSys_INVALID_SYSTEM;
        return -1;
    }
    if(sys->n_events < 0) {
        if(flag != 0) *flag = PSys_UNPOPULATED_SYSTEM;
        return -1;
    }
    if(flag != 0) *flag = PSys_OK;
    return sys->level;
}

#endif
