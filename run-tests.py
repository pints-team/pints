#!/usr/bin/env python3
#
# Runs all unit tests included in Pints.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import re
import os
import sys
import argparse
import unittest
import subprocess


def run_unit_tests():
    """
    Runs unit tests (without subprocesses).
    """
    tests = os.path.join('pints', 'tests')
    suite = unittest.defaultTestLoader.discover(tests, pattern='test*.py')
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


def run_flake8():
    """
    Runs flake8 in a subprocess, exits if it doesn't finish.
    """
    print('Running flake8 ... ')
    sys.stdout.flush()
    p = subprocess.Popen(
        [sys.executable, '-m', 'flake8'], stderr=subprocess.PIPE
    )
    try:
        ret = p.wait()
    except KeyboardInterrupt:
        try:
            p.terminate()
        except OSError:
            pass
        p.wait()
        print('')
        sys.exit(1)
    if ret == 0:
        print('ok')
    else:
        print('FAILED')
        sys.exit(ret)


def run_doctests():
    """
    Runs a number of tests related to documentation
    """

    print('\n{}\n# Starting doctests... #\n{}\n'.format('#' * 24, '#' * 24))

    # Check documentation can be built with sphinx
    doctest_sphinx()

    # Check all example notebooks are in the index
    doctest_examples_readme()

    # Check all classes and methods are documented in rst files, and no
    # unintended modules are exposed via a public interface
    doctest_rst_and_public_interface()

    # Check all .slow-books exist (they haven't been, e.g., renamed or removed)
    doctest_slow_books()

    print('\n{}\n# Doctests passed. #\n{}\n'.format('#' * 20, '#' * 20))


def doctest_sphinx():
    """
    Runs sphinx-build in a subprocess, checking that it can be invoked without
    producing errors.
    """
    print('Checking if docs can be built.')
    p = subprocess.Popen([
        'sphinx-build',
        '-b',
        'doctest',
        'docs/source',
        'docs/build/html',
        '-W',
    ])
    try:
        ret = p.wait()
    except KeyboardInterrupt:
        try:
            p.terminate()
        except OSError:
            pass
        p.wait()
        print('')
        sys.exit(1)
    if ret != 0:
        print('FAILED')
        sys.exit(ret)


def doctest_examples_readme():
    """
    Checks that every ipynb in the examples directory is included in the index
    `examples/README.md`.
    """
    print('\nChecking that example notebooks are indexed.')

    # Index file is in ./examples/README.md
    index_file = os.path.join('examples', 'README.md')
    assert(os.path.isfile(index_file))

    with open(index_file, 'r') as f:
        index_contents = f.read()

    # Get a list of all notebooks in the examples directory
    notebooks = [x[9:] for x in list_notebooks('examples')]
    assert(len(notebooks) > 10)

    # Find which are not indexed
    not_indexed = [nb for nb in notebooks if nb not in index_contents]

    # Report any failures
    if len(not_indexed) > 0:
        print('The following notebooks are not indexed in %s:' % index_file)
        for nb in sorted(not_indexed):
            print(nb)
        print('FAILED')
        sys.exit(1)
    else:
        print('All ' + str(len(notebooks)) + ' example notebooks are indexed.')


def doctest_rst_and_public_interface():
    """
    Check that every class and method is documented in an rst file and that
    no unintended modules are exposed via a public interface
    """
    print('\nChecking that all classes and methods are documented in an RST '
          'file and that public interfaces are clean.')

    # Import all public pints modules. Importing everything here is not
    # strictly needed: `import pints` happens to import `pints.noise`, and
    # `import pints.toy` will import `pints` and `pints.toy`, but we list
    # everything here for completeness. If a new module is added to pints
    # it should be imported here for this doctest.
    import pints
    import pints.io
    import pints.noise
    import pints.plot
    import pints.toy

    # If any modules other than these are exposed it may indicate that a module
    # has been inadvertently exposed in a public context, or that a new module
    # has been added to pints and should be imported above and included in this
    # list.
    pints_submodules = ['pints.io', 'pints.noise', 'pints.plot', 'pints.toy']

    doc_symbols = get_all_documented_symbols()

    check_exposed_symbols(pints, pints_submodules, doc_symbols)
    check_exposed_symbols(pints.io, [], doc_symbols)
    check_exposed_symbols(pints.noise, [], doc_symbols)
    check_exposed_symbols(pints.plot, [], doc_symbols)
    check_exposed_symbols(pints.toy, [], doc_symbols)

    print('All classes and methods are documented in an RST file, and all '
          'public interfaces are clean.')


def check_exposed_symbols(module, submodule_names, doc_symbols):
    """
    Check ``module`` for any classes and methods not contained in
    ``doc_symbols``, and check for any modules not contained in
    ``submodule_names``.

    Arguments:

    ``module``
        The module to check
    ``submodule_names``
        List of submodules expected to be exposed by ``module``
    ``doc_symbols``
        Dictionary containing lists of documented classes and functions
    """

    import inspect
    exposed_symbols = [x for x in dir(module) if not x.startswith('_')]
    symbols = [getattr(module, x) for x in exposed_symbols]

    classes = [x for x in symbols if inspect.isclass(x)]
    functions = [x for x in symbols if inspect.isfunction(x)]

    # Check for modules: these should match perfectly with _submodule_names
    exposed_modules = [x for x in symbols if inspect.ismodule(x)]
    unexpected_modules = [m for m in exposed_modules if
                          m.__name__ not in submodule_names]

    if len(unexpected_modules) > 0:
        print('The following modules are unexpectedly exposed in the public '
              'interface of %s:' % module.__name__)
        for m in sorted(unexpected_modules):
            print('  unexpected module: ' + m.__name__)

        print('For python modules such as numpy you may need to confine the '
              'import to the function scope. If you have created a new pints '
              'submodule, you will need to make %s (doctest) aware of this.'
              % __file__)
        print('FAILED')
        sys.exit(1)

    # Check that all classes are documented
    undocumented_classes = []
    for _class in classes:
        class_name = module.__name__ + '.' + _class.__name__
        if class_name not in doc_symbols['classes']:
            undocumented_classes.append(class_name)

    if len(undocumented_classes) > 0:
        print('The following classes do not appear in any RST file:')
        for m in sorted(undocumented_classes):
            print('  undocumented class: ' + m)
        print('FAILED')
        sys.exit(1)

    # Check that all functions are documented
    undocumented_functions = []
    for _funct in functions:
        funct_name = module.__name__ + '.' + _funct.__name__
        if funct_name not in doc_symbols['functions']:
            undocumented_functions.append(funct_name)

    if len(undocumented_functions) > 0:
        print('The following functions do not appear in any RST file:')
        for m in sorted(undocumented_functions):
            print('  undocumented function: ' + m)
        print('FAILED')
        sys.exit(1)


def get_all_documented_symbols():
    """
    Recursively traverse docs/source and identify all autoclass and
    autofunction declarations.

    Returns: A dict containing a list of classes and a list of functions
    """

    doc_files = []
    for root, dirs, files in os.walk(os.path.join('docs', 'source')):
        for file in files:
            if file.endswith('.rst'):
                doc_files.append(os.path.join(root, file))

    # Regular expression that would find either 'module' or 'currentmodule':
    # this needs to be prepended to the symbols as x.y.z != x.z
    regex_module = re.compile(r'\.\.\s*\S*module\:\:\s*(\S+)')

    # Regular expressions to find autoclass and autofunction specifiers
    regex_class = re.compile(r'\.\.\s*autoclass\:\:\s*(\S+)')
    regex_funct = re.compile(r'\.\.\s*autofunction\:\:\s*(\S+)')

    # Identify all instances of autoclass and autofunction in all rst files
    documented_symbols = {'classes': [], 'functions': []}
    for doc_file in doc_files:
        with open(doc_file, 'r') as f:
            # We need to identify which module each class or function is in
            module = ''
            for line in f.readlines():
                m_match = re.search(regex_module, line)
                c_match = re.search(regex_class, line)
                f_match = re.search(regex_funct, line)
                if m_match:
                    module = m_match.group(1) + '.'
                elif c_match:
                    documented_symbols['classes'].append(
                        module + c_match.group(1))
                elif f_match:
                    documented_symbols['functions'].append(
                        module + f_match.group(1))

    # Validate the list for any duplicate documentation
    for symbols in documented_symbols.values():
        if len(set(symbols)) != len(symbols):
            print('The following symbols are unexpectedly documented multiple '
                  'times in rst files:')

            dupes = set([d for d in symbols if symbols.count(d) > 1])
            for d in dupes:
                print('  multiple entries in docs: ' + d)

            print('FAILED')
            sys.exit(1)

    return documented_symbols


def doctest_slow_books():
    """
    Check that all notebooks listed in .slow-books actually exist. This
    prevents cron jobs falling over if slow notebooks are renamed without
    editing the list.
    """

    print('\nChecking that all notebooks listed in .slow-books exist.')

    with open('.slow-books', 'r') as f:
        slow_books = [b.strip() for b in f.readlines() if
                      b.strip().endswith('.ipynb')]

    if len(slow_books) < 1:
        print('No slow books found in .slow-books. Did something change?')
        print('FAILED')
        sys.exit(1)

    with open('.slow-books', 'r') as f:
        other_lines = [b.strip() for b in f.readlines() if not (
            b.strip().startswith('#') or
            b.strip().endswith('.ipynb') or
            b.strip() == ''
        )]

    if len(other_lines) > 0:
        print('The following entries are in .slow-books but are not ipynb:')
        print('  {}'.format('\n  '.join(other_lines)))
        print('FAILED')
        sys.exit(1)

    examples = [b[9:] for b in list_notebooks('examples')]
    undocumented = [b for b in slow_books if b not in examples]

    if len(undocumented) > 0:
        print('The following ipynb files are in .slow-books but are not in the'
              ' examples directory:')
        print('  {}'.format('\n  '.join(undocumented)))
        print('FAILED')
        sys.exit(1)

    print('All notebooks listed in .slow-books exist.')


def run_notebook_tests(skip_slow_books=False):
    """
    Runs Jupyter notebook tests. Exits if they fail.
    """
    debug = True

    # Ignore books with deliberate errors and books that are too slow for
    # fast testing.
    ignore_list = []
    if os.path.isfile('.error-books'):
        with open('.error-books', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line or line[:1] == '#':
                    continue
                if not line.startswith('examples/'):
                    line = 'examples/' + line
                if not line.endswith('.ipynb'):
                    line = line + '.ipynb'
                if not os.path.isfile(line):
                    raise Exception('Error notebook not found: ' + line)
                ignore_list.append(line)
    if skip_slow_books and os.path.isfile('.slow-books'):
        with open('.slow-books', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line or line[:1] == '#':
                    continue
                if not line.startswith('examples/'):
                    line = 'examples/' + line
                if not line.endswith('.ipynb'):
                    line = line + '.ipynb'
                if not os.path.isfile(line):
                    raise Exception('Slow notebook not found: ' + line)
                ignore_list.append(line)

    # Scan and run
    print('Testing notebooks')
    ok = True
    for notebook in list_notebooks('examples', True, ignore_list):
        if debug:
            print(notebook)
        else:
            ok &= test_notebook(notebook)
    if not ok:
        print('\nErrors encountered in notebooks')
        sys.exit(1)
    print('\nOK')


def list_notebooks(root, recursive=True, ignore_list=[], notebooks=[]):
    """
    Returns a list of all notebooks in a directory.
    """
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if path in ignore_list:
            print('Skipping slow/error book: ' + path)
            continue

        # Add notebooks
        if os.path.splitext(path)[1] == '.ipynb':
            notebooks.append(path)

        # Recurse into subdirectories
        elif recursive and os.path.isdir(path):
            # Ignore hidden directories
            if filename[:1] == '.':
                continue
            list_notebooks(path, recursive, ignore_list, notebooks)

    return notebooks


def test_notebook(path):
    """
    Tests a notebook in a subprocess, exists if it doesn't finish.
    """
    import nbconvert
    import pints
    b = pints.Timer()
    print('Running ' + path + ' ... ', end='')
    sys.stdout.flush()

    # Load notebook, convert to python
    e = nbconvert.exporters.PythonExporter()
    code, __ = e.from_filename(path)

    # Remove coding statement, if present
    code = '\n'.join([x for x in code.splitlines() if x[:9] != '# coding'])

    # Tell matplotlib not to produce any figures
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Template'

    # Run in subprocess
    cmd = [sys.executable, '-c', code]
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        stdout, stderr = p.communicate()
        # TODO: Use p.communicate(timeout=3600) if Python3 only
        if p.returncode != 0:
            # Show failing code, output and errors before returning
            print('ERROR')
            print('-- script ' + '-' * (79 - 10))
            for i, line in enumerate(code.splitlines()):
                j = str(1 + i)
                print(j + ' ' * (5 - len(j)) + line)
            print('-- stdout ' + '-' * (79 - 10))
            print(stdout)
            print('-- stderr ' + '-' * (79 - 10))
            print(stderr)
            print('-' * 79)
            return False
    except KeyboardInterrupt:
        p.terminate()
        print('ABORTED')
        sys.exit(1)

    # Sucessfully run
    print('ok (' + b.format() + ')')
    return True


def export_notebook(ipath, opath):
    """
    Exports the notebook at `ipath` to a python file at `opath`.
    """
    import nbconvert
    from traitlets.config import Config

    # Create nbconvert configuration to ignore text cells
    c = Config()
    c.TemplateExporter.exclude_markdown = True

    # Load notebook, convert to python
    e = nbconvert.exporters.PythonExporter(config=c)
    code, __ = e.from_filename(ipath)

    # Remove "In [1]:" comments
    r = re.compile(r'(\s*)# In\[([^]]*)\]:(\s)*')
    code = r.sub('\n\n', code)

    # Store as executable script file
    with open(opath, 'w') as f:
        f.write('#!/usr/bin/env python')
        f.write(code)
    os.chmod(opath, 0o775)


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run unit tests for Pints.',
        epilog='To run individual unit tests, use e.g.'
               ' $ pints/tests/test_toy_logistic_model.py',
    )
    # Unit tests
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run all unit tests using the `python` interpreter.',
    )
    # Notebook tests
    parser.add_argument(
        '--books',
        action='store_true',
        help='Test only the fast Jupyter notebooks in `examples`.',
    )
    parser.add_argument(
        '--allbooks',
        action='store_true',
        help='Test all Jupyter notebooks in `examples`.',
    )
    parser.add_argument(
        '-debook',
        nargs=2,
        metavar=('in', 'out'),
        help='Export a Jupyter notebook to a Python file for manual testing.',
    )
    # Doctests
    parser.add_argument(
        '--doctest',
        action='store_true',
        help='Run any doctests, check if docs can be built',
    )
    # Combined test sets
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick checks (unit tests, flake8, docs)',
    )

    # Parse!
    args = parser.parse_args()

    # Run tests
    has_run = False
    # Unit tests
    if args.unit:
        has_run = True
        run_unit_tests()
    # Doctests
    if args.doctest:
        has_run = True
        run_doctests()
    # Notebook tests
    if args.allbooks:
        has_run = True
        run_notebook_tests()
    elif args.books:
        has_run = True
        run_notebook_tests(True)
    if args.debook:
        has_run = True
        export_notebook(*args.debook)
    # Combined test sets
    if args.quick:
        has_run = True
        run_flake8()
        run_unit_tests()
        run_doctests()
    # Help
    if not has_run:
        parser.print_help()
