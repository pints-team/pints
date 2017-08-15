#
# Provides Ansi-C support
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
# Importers
#
# Exporters
from _exporter import AnsiCExporter, AnsiCCableExporter, AnsiCEulerExporter
_exporters = {
    'ansic' : AnsiCExporter,
    'ansic-cable' : AnsiCCableExporter,
    'ansic-euler' : AnsiCEulerExporter,
    }
def exporters():
    """
    Returns a dict of all exporters available in this module.
    """
    return dict(_exporters)
# Expression writers
from _ewriter import AnsiCExpressionWriter
_ewriters = {
    'ansic' : AnsiCExpressionWriter,
    }
def ewriters():
    """
    Returns a dict of all expression writers available in this module.
    """
    return dict(_ewriters)
# Language keywords
keywords = [
    # Basic language keywords
    'auto',
    'break',
    'case',
    'char',
    'const',
    'continue',
    'default',
    'do',
    'double',
    'else',
    'enum',
    'extern',
    'float',
    'for',
    'goto',
    'if',
    'inline',
    'int',
    'long',
    'register',
    'return',
    'short',
    'signed',
    'sizeof',
    'static',
    'struct',
    'switch',
    'typedef',
    'union',
    'unsigned',
    'void',
    'volatile',
    'while',
# From stddef.h
    'NULL',
    'offsetof',
    'ptrdiff_t',
    'wchar_t',
    'size_t',
# From limits.h
# TODO?
# From math.h
    'M_E',
    'M_LOG2E',
    'M_LOG10E',
    'M_LN2',
    'M_LN10',
    'M_PI',
    'M_PI_2',
    'M_PI_4',
    'M_1_PI',
    'M_2_PI',
    'M_2_SQRTPI',
    'M_SQRT2',
    'M_SQRT1_2',
    'MAXFLOAT',
    'HUGE_VAL',
    'acos',
    'asin',
    'atan',
    'atan2',
    'ceil',
    'cos',
    'cosh',
    'exp',
    'fabs',
    'floor',
    'fmod',
    'frexp',
    'ldexp',
    'log',
    'log10',
    'modf',
    'pow',
    'sin',
    'sinh',
    'sqrt',
    'tan',
    'tanh',
    'erf',
    'erfc',
    'gamma',
    'hypot',
    'j0',
    'j1',
    'jn',
    'lgamma',
    'y0',
    'y1',
    'yn',
    'isnan',
    'acosh',
    'asinh',
    'atanh',
    'cbrt',
    'expm1',
    'ilogb',
    'log1p',
    'logb',
    'nextafter',
    'remainder',
    'rint',
    'scalb',
# From sys/wait.h
# TODO?
# From stdlib.h
# http://pubs.opengroup.org/onlinepubs/7908799/xsh/stdlib.h.html
# Says: "Inclusion of the <stdlib.h> header may also make visible all
#        symbols from <stddef.h>, <limits.h>, <math.h> and <sys/wait.h>."
    'EXIT_FAILURE',
    'EXIT_SUCCESS',
    'RAND_MAX',
    'MB_CUR_MAX',
    'div_t',
    'ldiv_t',
    'a64l',
    'abort',
    'abs',
    'atexit',
    'atof',
    'atoi',
    'atol',
    'bsearch',
    'calloc',
    'div',
    'drand48',
    'ecvt',
    'erand48',
    'exit',
    'fcvt',
    'free',
    'gcvt',
    'getenv',
    'getsubopt',
    'grantpt',
    'initstate',
    'jrand48',
    'l64a',
    'labs',
    'lcong48',
    'ldiv',
    'lrand48',
    'malloc',
    'mblen',
    'mbstowcs',
    'mbtowc',
    'mktemp',
    'mkstemp',
    'mrand48',
    'nrand48',
    'ptsname',
    'putenv',
    'qsort',
    'rand',
    'rand_r',
    'random',
    'realloc',
    'realpath',
    'seed48',
    'setkey',
    'setstate',
    'srand',
    'srand48',
    'srandom',
    'strtod',
    'strtol',
    'strtoul',
    'system',
    'ttyslot',
    'unlockpt',
    'valloc',
    'wcstombs',
    'wctomb',
    ]
