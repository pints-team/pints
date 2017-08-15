<?
# opencl_info.c
#
# A pype template for an opencl information object
#
# Required variables
# -----------------------------------------------------------------------------
# module_name       A module name
# -----------------------------------------------------------------------------
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
import myokit.formats.opencl as opencl

tab = '    '
?>
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "mcl.h"

/*
 * Returns a tuple with information about the available opencl platforms and
 * devices.
 */
static PyObject*
info(PyObject *self, PyObject *args)
{
    mcl_device_info();
    return mcl_device_info();
}

/*
 * Methods in this module
 */
static PyMethodDef SimMethods[] = {
    {"info", info, METH_VARARGS, "Get some information about OpenCL devices."},
    {NULL},
};

/*
 * Module definition
 */
PyMODINIT_FUNC
init<?=module_name?>(void) {
    (void) Py_InitModule("<?= module_name ?>", SimMethods);
}
