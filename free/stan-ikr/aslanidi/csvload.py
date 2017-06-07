# Blatant rip from myokit
import array
import numpy as np
import os

def load(filename):
    """
    Loads a CSV file.
    """
    log = {}
    # Check filename
    filename = os.path.expanduser(filename)
    # Typecode dependent on precision
    typecode = 'd'
    # Error raising function
    def e(line, char, msg):
        raise myokit.DataLogReadError('Syntax error on line ' + str(line)
             + ', character ' + str(1 + char) + ': ' + msg)
    quote = '"'
    delim = ','
    with open(filename, 'r') as f:
        # Read header
        keys = [] # The log keys, in order of appearance
        try:
            line = f.readline()
            # Ignore lines commented with #
            while line.lstrip()[:1] == '#':
                line = f.readline()
        except EOFError:
            # Empty file
            return log
        # Trim end of line
        if len(line) > 1 and line[-2:] == '\r\n':
            eol = 2
            line = line[:-2]
        else:
            eol = 1
            line = line[:-1]
        # Trim ; at end of line if given
        if line[-1:] == ';':
            eol += 1
            line = line[:-1]
        # Get enumerated iterator over characters
        line = enumerate(line)
        try:
            i, c = line.next()
        except StopIteration:
            # Empty line
            return log
        # Whitespace characters to ignore
        whitespace = ' \f\t'
        # Start parsing header fields
        run1 = True
        while run1:
            text = []
            # Skip whitespace
            try:
                while c in whitespace:
                    i, c = line.next()
            except StopIteration:
                break
            if c == quote:
                # Read quoted field + delimiter or eol
                run2 = True
                while run2:
                    try:
                        i, c = line.next()
                    except StopIteration:
                        e(1, i, 'Unexpected end-of-line inside quoted'
                            ' string.')
                    if c == quote:
                        try:
                            i, c = line.next()
                            if c == quote:
                                text.append(quote)
                            elif c == delim or c in whitespace:
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
        lists = []
        for key in keys:
            x = array.array(typecode)
            lists.append(x)
            log[key] = x
        # Read remaining data
        try:
            n = 0
            while True:
                row = f.readline()
                # Ignore blank lines
                if row.strip() == '':
                    break
                # Ignore lines commented with #
                if row.lstrip()[:1] == '#':
                    continue
                row = row[:-eol]
                row = row.split(delim)
                n += 1
                if len(row) != m:
                    e(n, 0, 'Wrong number of columns found in row '
                        + str(n) + '. Expecting ' + str(m) + ', found '
                        + str(len(row)) +'.')
                try:
                    for k, v in enumerate(row):
                        lists[k].append(float(v))
                except ValueError:
                    e(n, 0, 'Unable to convert found data to floats.')
        except StopIteration:
            pass
        # Create numpy view of data
        for key in keys:
            log[key] = np.array(log[key], copy=False)
        # Return log
        return log
