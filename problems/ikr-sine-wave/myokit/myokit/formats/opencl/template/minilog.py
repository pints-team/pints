#
# Stripped version of myokit.DataLog
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import os
from array import array
from collections import OrderedDict
class DataLog(OrderedDict):
    """
    Extends an ordered ``dict`` type with a number of methods specifically for
    working with logged simulation data.
    """
    def get(self, name, *cell):
        """
        Convenience method for access to multi-cell data. For example, in a 0d
        (single cell) simulation, a log could look like this::

            d = myokit.DataLog()
            d['engine.time'] = numpy.range(0, 10)
            d['membrane.V'] = 10 * d['engine.time']

        In this case, the ``get`` method just returns whatever is stored at the
        given name::

            d.get('membrane.V') == d['membrane.V']

        In a 1d cable situation, a log can be built like this::

            d = myokit.DataLog()
            d['engine.time'] = numpy.range(0, 10)
            d['0.membrane.V'] = 10 * d['engine.time']       # First cell
            d['1.membrane.V'] = 20 * d['engine.time']       # Second cell

        In this case, the ``get`` method can be used to access the different
        cells::

            d.get('membrane.V', 0) == d['0.membrane.V']

        Similarly, in a 2d tissue log ``d.get('membrane.V', 1, 2)`` will return
        the entry stored at ``d['1.2.membrane.V']``.
        """
        key = [str(x) for x in cell]
        key.append(str(name))
        key = '.'.join(key)
        return self[key]
    @staticmethod
    def load_csv(filename):
        """
        Loads a CSV file from disk and returns it as a :class:`DataLog`.

        The CSV file must start with a header line indicating the variable names,
        separated by commas. Each subsequent row should contain the values at a
        single point in time for all logged variables.
        """
        # Check filename
        filename = os.path.expanduser(filename)
        # Error method
        def e(line, char, msg):
            raise Exception('Syntax error on line ' + str(line)
                + ', character ' + str(1 + char) + ': ' + msg)
         # Delimiters
        quote = '"'
        delim = ','
        with open(filename, 'rb') as f:
            # Read header
            keys = []
            try:
                line = f.readline()
            except EOFError:
                e(0, 0, 'Empty file, expecting header.')
            # Trim end of line
            if len(line) > 1 and line[-2:] == '\r\n':
                eol = 2
                line = line[:-2]
            else:
                eol = 1
                line = line[:-1]
            # Trim ; at end of line if given
            if line[-1] == ';':
                eol += 1
                line = line[:-1]
            # Get enumerated iterator over characters
            line = enumerate(line)
            try:
                i, c = line.next()
            except StopIteration:
                e(1, i, 'Empty line, expecting header.')
            run1 = True
            while run1:
                text = []
                if c == quote:
                    # Read quoted field + delimiter or eol
                    run2 = True
                    while run2:
                        try:
                            i, c = line.next()
                        except StopIteration:
                            e(1, i, 'Unexpected end-of-line inside quoted string.')
                        if c == quote:
                            try:
                                i, c = line.next()
                                if c == quote:
                                    text.append(quote)
                                elif c == delim:
                                    run2 = False
                                else:
                                    e(1, i, 'Expecting double quote, delimiter'
                                        ' or end-of-line. Found "' + c + '".')
                            except StopIteration:
                                run1 = run2 = False
                        else:
                            text.append(c)
                else:
                    # Read unquoted field + delimiter or eol
                    while run1 and c != delim:
                        try:
                            text.append(c)
                            i, c = line.next()
                        except StopIteration:
                            run1 = False
                # Append new field to list
                keys.append(''.join(text))
                # Read next character
                try:
                    i, c = line.next()
                except StopIteration:
                    run1 = False
            if c == delim:
                e(1, i, 'Empty field in header.')
            # Create data structure
            m = len(keys)
            log = DataLog()
            lists = []
            for key in keys:
                x = array('d')
                lists.append(x)
                log[key] = x
            # Read remaining data
            try:
                n = 0
                while True:
                    row = f.readline()
                    if row.strip() == '':
                        break
                    row = row[:-eol]
                    row = row.split(delim)
                    n += 1
                    if len(row) != m:
                        raise Exception('Wrong number of columns found in row '
                            + str(n) + '. Expecting ' + str(m) + ', found '
                            + str(len(row)) +'.')
                    for k, v in enumerate(row):
                        lists[k].append(float(v))
            except StopIteration:
                pass
            # Return log
            return log
    def npview(self):
        """
        Returns an ordered dict of numpy arrays pointing to the data stored in
        this log.
        """
        import numpy as np
        out = DataLog()
        for k, d in self.iteritems():
            out[k] = np.array(d, copy=False)
        return out
    def save_csv(self, filename, pad=None):
        """
        Writes this ``DataLog`` to a CSV file, following the syntax
        outlined in RFC 4180 and with a header indicating the field names.

        The resulting file will consist of:

          - A header line containing the names of all logged variables,
            separated by commas.
          - Each following line will be a comma separated list of values in the
            same order as the header line. A line is added for each time point
            logged.
        """
        # Check filename
        filename = os.path.expanduser(filename)
        # EOL: CSV files use DOS line endings '\r\n'. In windows, if you try
        # to write a '\n' in mode 'w' it automatically writes '\r\n', so '\r\n'
        # gets converted to '\r\r\n'. To circumvent this, open the file in mode
        # 'wb'.
        # Save
        eol = '\r\n'
        delim = ','
        quote = '"'
        escape = '""'
        with open(filename, 'wb') as f:
            # Convert dict structure to ordered sequences
            keys = []
            data = []
            n = []
            for key, dat in sorted(self.iteritems()):
                keys.append(key)
                data.append(iter(dat))
                n.append(len(dat))
            m = len(keys)
            n = set(n)
            if len(n) > 1:
                # Padding needed, check if provided
                if pad is None:
                    raise Exception('Data passed to save_csv contains lists of'
                        ' unequal length (' + str(n) + '). Please ensure all'
                        ' lists have equal length or set a padding value.')
            else:
                # Padding not needed
                pad = None
            n = max(n)
            # Write header
            line = []
            for key in keys:
                # Escape quotes within strings
                line.append(quote + key.replace(quote, escape) + quote)
            f.write(delim.join(line))
            f.write(eol)
            # Write data
            if pad is None:
                for i in xrange(0, n):
                    line = []
                    for d in data:
                        line.append(myokit.strfloat(d.next()))
                    f.write(delim.join(line) + eol)
            else:
                for i in xrange(0, n):
                    line = []
                    for d in data:
                        try:
                            line.append(myokit.strfloat(d.next()))
                        except StopIteration:
                            line.append('0')
                    f.write(delim.join(line) + eol)
