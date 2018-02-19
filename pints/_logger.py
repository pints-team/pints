#
# Logger class
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import sys
import numpy as np
import collections

# Value types
_COUNTER = 0
_FLOAT = 1
_INT = 2


class Logger(object):
    """
    Logs numbers to screen and/or a file.

    Example::

        log = pints.Logger()
        log.add_counter('id', width=2)
        log.add_float('Length')
        log.log(1, 1.23456)
        log.log(2, 7.8901)

    """
    def __init__(self):
        super(Logger, self).__init__()

        # Log to screen
        self._stream = sys.stdout

        # Log to file (disabled)
        self._filename = None

        # Log to file in csv mode
        self._csv_mode = False

        # Started writing?
        self._have_logged = False

        # Logging fields

        # List of field names
        self._field_names = []

        # List of field formatting options, specified as a tuple:
        #    (width, type, format1, format2)
        # Where format2 is a format to be used if format1 is too wide.
        # For the format specification, see:
        #   https://docs.python.org/3/library/string.html#formatspec
        self._field_formats = []

        # List of field indices to write to stream
        self._stream_fields = []

        # Buffer of data to log
        self._buffer = collections.deque()

    def add_counter(self, name, width=5, max_value=None, file_only=False):
        """
        Adds a field for positive integers.

        Arguments:

        ``name``
            This field's name. Will be displayed in the header.
        ``width``
            A hint for the width of this column. If numbers exceed this width
            layout will break, but no information will be lost.
        ``max_value``
            A hint for the maximum number this field will need to display.
        ``file_only``
            If set to ``True``, this field will not be shown on screen.

        Returns this :class:`Logger` object.
        """
        if self._have_logged:
            raise ValueError('Cannot add fields after logging has started.')

        # Check name & width
        name = str(name)
        width = int(width)

        # Determine field width
        width = max(width, len(name), 1)
        if max_value is not None:
            max_value = float(max_value)
            width = max(width, int(np.ceil(np.log10(max_value))))

        # Create format
        f1 = f2 = '{:<' + str(width) + 'd}'

        # Add field
        self._field_names.append(name)
        self._field_formats.append((width, _COUNTER, f1, f2))
        if not file_only:
            self._stream_fields.append(len(self._field_names) - 1)

        # Return self to allow for chaining
        return self

    def add_float(self, name, width=7, file_only=False):
        """
        Adds a field for floating point number.

        Arguments:

        ``name``
            This field's name. Will be displayed in the header.
        ``width``
            A hint for the field's width. The minimum width is 7.
        ``file_only``
            If set to ``True``, this field will not be shown on screen.

        Returns this :class:`Logger` object.
        """
        if self._have_logged:
            raise ValueError('Cannot add fields after logging has started.')

        # Example: 5 digits => width 11
        # -1.234e-299
        # 12345678901
        #    12345

        # Example: 5 digits => 7
        # -1.2345

        # Example: 1 digit => 7
        # -1e-299

        # Check name & width
        name = str(name)
        width = int(width)

        # Determine field width
        width = max(width, len(name), 7)

        # Create format
        # 'g' is for general floating point number, formatting depends on
        # magnitude
        f1 = '{: .' + str(width - 2) + 'g}'
        f2 = '{: .' + str(width - 6) + 'g}'

        # Add field
        self._field_names.append(name)
        self._field_formats.append((width, _FLOAT, f1, f2))
        if not file_only:
            self._stream_fields.append(len(self._field_names) - 1)

        # Return self to allow for chaining
        return self

    def add_int(self, name, width=5, file_only=False):
        """
        Adds a field for a (positive or negative) integer.

        Arguments:

        ``name``
            This field's name. Will be displayed in the header.
        ``width``
            A hint for the width of this column. If numbers exceed this width
            layout will break, but no information will be lost.
        ``file_only``
            If set to ``True``, this field will not be shown on screen.

        Returns this :class:`Logger` object.
        """
        if self._have_logged:
            raise ValueError('Cannot add fields after logging has started.')

        # Check name & width
        name = str(name)
        width = int(width)

        # Determine field width
        width = int(max(width, len(name), 1))

        # Create format
        f1 = f2 = '{:< ' + str(width) + 'd}'

        # Add field
        self._field_names.append(name)
        self._field_formats.append((width, _INT, f1, f2))
        if not file_only:
            self._stream_fields.append(len(self._field_names) - 1)

        # Return self to allow for chaining
        return self

    def add_long_float(self, name, file_only=False):
        """
        Adds a field for a maximum precision floating point number.

        Arguments:

        ``name``
            This field's name. Will be displayed in the header.
        ``file_only``
            If set to ``True``, this field will not be shown on screen.

        Returns this :class:`Logger` object.
        """
        if self._have_logged:
            raise ValueError('Cannot add fields after logging has started.')

        # Example: 17 digits = width 25
        # -1.23456699999999992e-299
        # 1234567890123456789012345
        #  1 23456789012345678

        # Example: 17 digits = width 24
        # -1.23456699999999997e+00
        # 123456789012345678901234
        #  1 23456789012345678

        # Check name
        name = str(name)

        # Determine field width
        width = max(len(name), 24)

        # Create format
        f1 = '{: .17e}'
        f2 = '{: .16e}'

        # Add field
        self._field_names.append(name)
        self._field_formats.append((width, _FLOAT, f1, f2))
        if not file_only:
            self._stream_fields.append(len(self._field_names) - 1)

        # Return self to allow for chaining
        return self

    def log(self, *data):
        """
        Logs a new row of data.
        """
        # Ignore data if no logging specified
        if self._stream is None and self._filename is None:
            return

        # Check number of fields
        nfields = len(self._field_names)
        if nfields < 1:
            raise ValueError('Unable to log: No fields specified.')

        # Exactly one row given? Then log, else store in buffer
        rows = []
        if len(self._buffer) == 0 and len(data) == nfields:
            rows.append(data)
        else:
            self._buffer.extend(data)
            while len(self._buffer) >= nfields:
                rows.append([self._buffer.popleft() for i in range(nfields)])

            # Nothing to print? Then return
            if not rows:
                return

        # Log in CSV format
        if self._csv_mode and self._filename is not None:

            mode = 'a' if self._have_logged else 'w'
            with open(self._filename, mode) as f:

                # Write names
                if not self._have_logged:
                    f.write(','.join(
                        ['"' + x + '"' for x in self._field_names]) + '\n')

                # Write data
                for row in rows:
                    line = []
                    i = iter(row)
                    for width, dtype, f1, f2 in self._field_formats:
                        if dtype == _FLOAT:
                            x = '{:<.17e}'.format(next(i))
                        else:
                            x = str(int(next(i)))
                        line.append(x)
                    f.write(','.join(line) + '\n')

            # No need to log to screen? Then skip line formatting and return
            if not self._stream:
                self._have_logged = True
                return

        # Format fields
        formatted_rows = []

        # Add headers
        if not self._have_logged:
            headers = []
            for i, name in enumerate(self._field_names):
                width = self._field_formats[i][0]
                headers.append(name + ' ' * (width - len(name)))
            formatted_rows.append(headers)

        # Add data
        for row in rows:
            column = iter(row)
            formatted_row = []
            for width, dtype, f1, f2 in self._field_formats:
                if dtype == _FLOAT:
                    v = next(column)
                    x = f1.format(v)
                    if len(x) > width:
                        x = f2.format(v)
                    x += ' ' * (width - len(x))
                else:
                    x = f1.format(int(next(column)))
                formatted_row.append(x)
            formatted_rows.append(formatted_row)

        # Log to screen
        if self._stream is not None:
            lines = []
            for row in formatted_rows:
                lines.append(' '.join([row[i] for i in self._stream_fields]))
            self._stream.write('\n'.join(lines) + '\n')

        # Log to file (non csv)
        if self._filename is not None and not self._csv_mode:
            lines = []
            for row in formatted_rows:
                lines.append(' '.join([x for x in row]))
            with open(self._filename, 'a' if self._have_logged else 'w') as f:
                f.write('\n'.join(lines) + '\n')

        # Have logged!
        self._have_logged = True

    def set_filename(self, filename=None, csv=False):
        """
        Enables logging to a file if a ``filename`` is passed in. Logging to
        file can be disabled by passing ``filename=None``.

        Usually, file logging happens in the same format as logging to screen.
        To obtain csv logs instead, set `csv=True`
        """
        if self._have_logged:
            raise ValueError('Cannot configure after logging has started.')

        if filename is None:
            self._filename = None
        else:
            self._filename = str(filename)
        self._csv_mode = True if csv else False

    def set_stream(self, stream=sys.stdout):
        """
        Enables logging to screen if an output ``stream`` is passed in. Logging
        to screen can be disabled by passing ``stream=None``.
        """
        if self._have_logged:
            raise ValueError('Cannot configure after logging has started.')

        self._stream = stream

