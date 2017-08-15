#
# Provides Stan support (see: http://mc-stan.org/)
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
from _exporter import StanExporter
_exporters = {
    'stan' : StanExporter,
    }
def exporters():
    """
    Returns a dict of all exporters available in this module.
    """
    return dict(_exporters)
# Expression writers
from _ewriter import StanExpressionWriter
_ewriters = {
    'stan' : StanExpressionWriter,
    }
def ewriters():
    """
    Returns a dict of all expression writers available in this module.
    """
    return dict(_ewriters)
# Language keywords
# Variable names can overlap with built-in function names in stan, so that
# `real abs = abs(-5)` is a perfectly valid statement.
keywords = [
    # Data types, constraints
    'cholesky_factor_corr',
    'cholesky_factor_cov',
    'corr_matrix',
    'cov_matrix',
    'int',
    'lower',
    'matrix',
    'ordered',
    'positive_ordered',
    'real',
    'row_vector',
    'simplex',
    'unit_vector',
    'upper',
    'vector',
    #not a number
    #+inf
    #-inf
    # Block identifiers
    'data',
    'functions',
    'generated_quantities',
    'model',
    'parameters',
    'transformed data',
    'transformed_parameters',
    # Language
    'break',
    'continue',
    'else',
    'false',
    'for',
    'if',
    'in',
    'repeat',
    'return',
    'then',
    'true',
    'until',
    'while',
    # Reserved names from C++ implementation
    'fvar',
    'STAN_MAJOR',
    'STAN_MATH_MAJOR',
    'STAN_MATH_MINOR',
    'STAN_MATH_PATH',
    'STAN_MINOR',
    'STAN_PATCH',
    'var',
    # Further C++ keywords
    'alignas',
    'alignof',
    'and',
    'and_eq',
    'asm',
    'auto',
    'bitand',
    'bitor',
    'bool',
    'break',
    'case',
    'catch',
    'char',
    'char16_t',
    'char32_t',
    'class',
    'compl',
    'const',
    'const_cast',
    'constexpr',
    'continue',
    'decltype',
    'default',
    'delete',
    'do',
    'double',
    'dynamic_cast',
    'else',
    'enum',
    'explicit',
    'export',
    'extern',
    'false',
    'float',
    'for',
    'friend',
    'goto',
    'if',
    'inline',
    'int',
    'long',
    'mutable',
    'namespace',
    'new',
    'noexcept',
    'not',
    'not_eq',
    'nullptr',
    'operator',
    'or',
    'or_eq',
    'private',
    'protected',
    'public',
    'register',
    'reinterpret_cast',
    'return',
    'short',
    'signed',
    'sizeof',
    'static',
    'static_assert',
    'static_cast',
    'struct',
    'switch',
    'template',
    'this',
    'thread_local',
    'throw',
    'true',
    'try',
    'typedef',
    'typeid',
    'typename',
    'union',
    'unsigned',
    'using',
    'virtual',
    'void',
    'volatile',
    'wchar_t',
    'while',
    'xor',
    'xor_eq',
    ]
