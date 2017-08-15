#
# This file is part of Myokit
#  Copyright 2017      University of Oxford
#  Copyright 2011-2016 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
"""
Myokit: The Maastricht Modeling Toolkit

This module provides a gateway to the main myokit components. For example, to
load a model use myokit.load_model('filename.mmt'), create a myokit.Protocol
and then a myokit.Simulation which you can .run() to obtain simulated results.
"""
#__all__ =
#
# __all__ should NOT be provided! Doing so removes all methods below from
# the content imported by "from myokit import *".
#
# Without an explicit __all__, importing * will result in importing all
# functions and classes described below. No submodules of myokit will be
# loaded!
#
# GUI and graphical modules should not be auto-included because they define a
# matplotlib backend to use. If the user requires a different backend, this
# will generate an error.
#
# Check python version
import sys
if sys.hexversion < 0x02070000:
    print('-- ERROR --')
    print('Myokit requires Python version 2.7.0 or higher.')
    print('Detected Python version: ')
    print(sys.version)
    print()
    sys.exit(1)
if sys.hexversion > 0x03000000:
    print('-- ERROR --')
    print('Myokit is not compatible with Python 3.')
    print('Detected Python version: ')
    print(sys.version)
    print()
    sys.exit(1)
del(sys)
# Constants
# Version information
VERSION_INT = 1,25,2
VERSION = '.'.join([str(x) for x in VERSION_INT]); del(x)
RELEASE = ''
# Licensing
# Full license text
LICENSE = """
Myokit
Copyright 2011-2017 Maastricht University
michael@myokit.org

Myokit is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your
option) any later version.

Myokit is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

For a copy of the GNU General Public License,
see http://www.gnu.org/licenses/.
""".strip()
# Full license text, html
LICENSE_HTML = """
<h1>Myokit</h1>
<p>
    Copyright 2011-2017 Maastricht University
    <br /><a href="mailto:michael@myokit.org">michael@myokit.org</a>
</p>
<p>
    Myokit is free software: you can redistribute it and/or modify it under the
    terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version.
</p>
<p>
    Myokit is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
    details.
</p>
<p>
    For a copy of the GNU General Public License, see
    <a href="http://www.gnu.org/licenses/">http://www.gnu.org/licenses/</a>.
</p>
""".strip()
# Example license header that should appear in most files
LICENSE_HEADER = """
This file is part of Myokit
 Copyright 2011-2017 Maastricht University
 Licensed under the GNU General Public License v3.0
 See: http://myokit.org

Authors:
""".strip()
# Single-line copyright notice
COPYRIGHT = '(C) 2011-2017, Maastricht University'
# Myokit paths
import os
import inspect
try:
    frame = inspect.currentframe()
    DIR_MYOKIT = os.path.dirname(inspect.getfile(frame))
finally:
    # Always manually delete frame
    # https://docs.python.org/2/library/inspect.html#the-interpreter-stack
    del(frame)
DIR_DATA   = os.path.join(DIR_MYOKIT, '_bin')
DIR_CFUNC  = os.path.join(DIR_MYOKIT, '_sim')
# Location of myokit user info
DIR_USER = os.path.join(os.path.expanduser('~'), '.myokit')
if os.path.exists(DIR_USER):
    if not os.path.isdir(DIR_USER):
        raise Exception('File or link found in place of user directory: '
            + str(DIR_USER))
else:
    os.makedirs(DIR_USER)
# Location of example mmt file
EXAMPLE = os.path.join(DIR_DATA, 'example.mmt')
# Prevent standard libraries being represented as part of Myokit
del(os, inspect)
# Debugging mode: Simulation code will be shown, not executed
DEBUG = False
# Data logging flags (bitmasks)
LOG_NONE  = 0
LOG_STATE = 1
LOG_BOUND = 2
LOG_INTER = 4
LOG_DERIV = 8
LOG_ALL   = LOG_STATE + LOG_INTER + LOG_BOUND + LOG_DERIV
# Floating point precision
SINGLE_PRECISION = 32
DOUBLE_PRECISION = 64
# Unit checking modes
UNIT_TOLERANT = 1
UNIT_STRICT   = 2
# Maximum precision float output
SFDOUBLE = '{:< 1.17e}'
SFSINGLE = '{:< 1.9e}'
# Date and time formats to use throughout Myokit
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
TIME_FORMAT = '%H:%M:%S'
# Add line numbers to debug output of simulations
DEBUG_LINE_NUMBERS = True
# Favor PySide or PyQt
FORCE_PYQT5 = False
FORCE_PYQT4 = False
FORCE_PYSIDE = False
# Location of the Sundials (CVODE) shared library objects (.dll or .so)
SUNDIALS_LIB = []
# Location of the Sundials (CVODE) header files (.h)
SUNDIALS_INC = []
# Location of the OpenCL shared library objects (.dll or .so)
OPENCL_LIB = []
# Location of the OpenCL header files (.h)
OPENCL_INC = []
# Load settings
import _config
del(_config)
# Myokit version
def version(raw=False):
    """
    Returns the current Myokit version.
    """
    if raw:
        return VERSION
    else:
        return '\n Myokit version ' + VERSION + ' '*(15-len(VERSION)) \
            + '|/\\\n_______________________________|  |______'
# Exceptions
from ._err import MyokitError
from ._err import IntegrityError, InvalidBindingError, InvalidLabelError
from ._err import DuplicateName, InvalidNameError, IllegalAliasError
from ._err import UnresolvedReferenceError, IllegalReferenceError
from ._err import UnusedVariableError, CyclicalDependencyError
from ._err import MissingRhsError, MissingTimeVariableError
from ._err import NonLiteralValueError, NumericalError
from ._err import IncompatibleUnitError, InvalidMetaDataNameError
from ._err import DuplicateFunctionName, DuplicateFunctionArgument
from ._err import InvalidFunction
from ._err import ParseError, SectionNotFoundError
from ._err import ProtocolParseError, ProtocolEventError
from ._err import SimultaneousProtocolEventError
from ._err import SimulationError, FindNanError, SimulationCancelledError
from ._err import InvalidDataLogError, DataLogReadError, DataBlockReadError
from ._err import GenerationError, CompilationError
from ._err import ImportError, ExportError
from ._err import IncompatibleModelError
# Check if all errors imported
# Dynamically importing them doesn't seem to be possible, and forgetting to
#  import an error creates a hard to debug bug (something needs to go wrong
#  before the interpreter reaches the code raising the error and notices it's
#  not there).
import inspect
_globals = globals()
ex, name, clas = None, None, None
for ex in inspect.getmembers(_err):
    name, clas = ex
    if type(clas) == type(MyokitError) and issubclass(clas, MyokitError):
        if not name in _globals:
            raise Exception('Failed to import exception: ' + name)
del(ex, name, clas, _globals, inspect) # Prevent public visibility
# Model structure
from ._core import ModelPart, Model, Component, Variable, check_name
from ._core import Equation, EquationList, UserFunction
# Expressions and units
from ._expr import Expression
from ._expr import LhsExpression, Derivative, Name, Number
from ._expr import PrefixExpression, PrefixPlus, PrefixMinus
from ._expr import InfixExpression, Plus, Minus, Multiply, Divide
from ._expr import Quotient, Remainder, Power
from ._expr import Function, Sqrt, Sin, Cos, Tan, ASin, ACos, ATan, Exp
from ._expr import Log, Log10, Floor, Ceil, Abs
from ._expr import If, Condition, PrefixCondition, Not, And, Or, InfixCondition
from ._expr import Equal, NotEqual, More, Less, MoreEqual, LessEqual
from ._expr import Piecewise, OrderedPiecewise, Polynomial, Spline
from ._expr import UnsupportedFunction
from ._expr import Unit, Quantity
# Pacing protocol
from ._protocol import Protocol, ProtocolEvent, PacingSystem
# Parser functions
from ._parser import KEYWORDS
from ._parser import parse, split, format_parse_error
from ._parser import parse_model, parse_protocol, parse_state
from ._parser import parse_unit_string as parse_unit
from ._parser import parse_number_string as parse_number
from ._parser import parse_expression_string as parse_expression
from ._parser import strip_expression_units
# Global date and time formats
from ._aux import date, time
# Reading, writing
from ._aux import load, load_model, load_protocol, load_script
from ._aux import save, save_model, save_protocol, save_script
from ._aux import load_state, save_state
from ._aux import load_state_bin, save_state_bin
# Test step
from ._aux import step
# Output masking
from ._aux import PyCapture, SubCapture
# Sorting
from ._aux import natural_sort_key
# Data logging
from ._datalog import DataLog, LoggedVariableInfo
from ._datalog import dimco, split_key, prepare_log
from ._datablock import DataBlock1d, DataBlock2d, ColorMap
# Simulations
from ._sim import ProgressReporter, ProgressPrinter
from ._sim import CModule, CppModule
from ._sim.cvode import Simulation
from ._sim.cable import Simulation1d
from ._sim.opencl import OpenCL
from ._sim.opencl import OpenCLInfo, OpenCLPlatformInfo, OpenCLDeviceInfo
from ._sim.openclsim import SimulationOpenCL
#from ._sim.openmp import SimulationOpenMP
from ._sim.fiber_tissue import FiberTissueSimulation
from ._sim.rhs import RhsBenchmarker
from ._sim.jacobian import JacobianTracer, JacobianCalculator
from ._sim.icsim import ICSimulation
from ._sim.psim import PSimulation
# Auxillary functions
from ._aux import pywriter, numpywriter
from ._aux import ModelComparison
from ._aux import Benchmarker, TextLogger
from ._aux import lvsd, format_path, strfloat, format_float_dict
from ._aux import pack_snapshot
# Running scripts
from ._aux import run
# Import whole modules
# This allows these modules to be used after myokit was imported, without
# importing the modules specifically (like os and os.path).
# All modules imported here must report so in their docs
import mxml
import pacing
import units # Also loads all common unit names
# Globally shared progress reporter
_Simulation_progress = None
# Default mmt file parts
def default_protocol():
    """
    Provides a default protocol to use when no embedded one is available.
    """
    p = Protocol()
    p.schedule(1, 100, 0.5, 1000, 0)
    return p
def default_script():
    """
    Provides a default script to use when no embedded script is available.
    """
    return \
"""[[script]]
import matplotlib.pyplot as pl
import myokit

# Get model and protocol, create simulation
m = get_model()
p = get_protocol()
s = myokit.Simulation(m, p)

# Run simulation
d = s.run(1000)

# Get the first state variable's name
first_state = m.states().next()
var = first_state.qname()

# Display the results
pl.figure()
pl.plot(d.time(), d[var])
pl.title(var)
pl.show()
"""
