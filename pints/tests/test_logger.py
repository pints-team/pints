#!/usr/bin/env python3
#
# Tests the Logger class.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import os
import sys
import pints
import unittest

from shared import StreamCapture, TemporaryDirectory


data = [
    1, 4, 1.234567890987654321, 12, 10, 0, 'yes',
    None, 3.234, -2.234567890987654321e12, 230, 100, 7.9, 'yes',
    30, -2.23456789, -3.234567890987654321e-12, None, 1000, 179.99999, 'no',
    40, 1.23456789, 4.234567890987654321e-123, -12, 10000, 12345.6, None,
]
out1 = (
    '#  Latitude Number                   Val  Count Time     Q  \n' +
    '1   4        1.23456789098765429e+00  12  10      0:00.0 yes\n' +
    '    3.234   -2.23456789098765430e+12  230 100     0:07.9 yes\n' +
    '30 -2.23457 -3.23456789098765439e-12      1000    3:00.0 no \n' +
    '40  1.23457  4.2345678909876540e-123 -12  10000 205:45.6    \n'
)
out2 = (
    '#  Lat.    Val  Count Time     Q  \n' +
    '1   4       12  10      0:00.0 yes\n' +
    '    3.234   230 100     0:07.9 yes\n' +
    '30 -2.2346      1000    3:00.0 no \n' +
    '40  1.2346 -12  10000 205:45.6    \n'
)
out3 = (
    '#  Lat.    Number                   Val  Count Time     Q  \n' +
    '1   4       1.23456789098765429e+00  12  10      0:00.0 yes\n' +
    '    3.234  -2.23456789098765430e+12  230 100     0:07.9 yes\n' +
    '30 -2.2346 -3.23456789098765439e-12      1000    3:00.0 no \n' +
    '40  1.2346  4.2345678909876540e-123 -12  10000 205:45.6    \n'
)
out4 = (
    '"#","Lat.","Number","Val","Count","Time","Q"\n' +
    '1,4.00000000000000000e+00,1.23456789098765429e+00,12,10,0,"yes"\n' +
    ',3.23399999999999999e+00,-2.23456789098765430e+12,230,100,7.9,"yes"\n' +
    '30,-2.23456789000000011e+00,-3.23456789098765439e-12,,1000,' +
    '179.99999,"no"\n' +
    '40,1.23456788999999989e+00,4.23456789098765400e-123,-12,10000,' +
    '12345.6,\n'
)


class TestLogger(unittest.TestCase):
    """
    Tests the Logger class.
    """
    def test_all_simultaneously(self):
        # Normal use, all data at once
        with StreamCapture() as c:
            # Test logger with no fields
            log = pints.Logger()
            self.assertRaises(ValueError, log.log, 1)

            # Test logging output
            log.add_counter('#', width=2)
            log.add_float('Latitude', width=1)
            log.add_long_float('Number')
            log.add_int('Val', width=4)
            log.add_counter('Count', max_value=12345)
            log.add_time('Time')
            log.add_string('Q', 3)

            # Add all data in one go
            log.log(*data)
        self.assertOutput(expected=out1, returned=c.text())

        # Can't configure once logging
        self.assertRaises(RuntimeError, log.add_counter, 'a')
        self.assertRaises(RuntimeError, log.add_int, 'a')
        self.assertRaises(RuntimeError, log.add_float, 'a')
        self.assertRaises(RuntimeError, log.add_long_float, 'a')
        self.assertRaises(RuntimeError, log.add_time, 'a')
        self.assertRaises(RuntimeError, log.add_string, 'a', 3)
        self.assertRaises(RuntimeError, log.set_filename, 'a')
        self.assertRaises(RuntimeError, log.set_stream, sys.stdout)

    def test_partial_row_not_shown(self):
        # Normal use, all data at once, plus extra bit
        with StreamCapture() as c:
            log = pints.Logger()
            log.add_counter('#', width=2)
            log.add_float('Latitude', width=1)
            log.add_long_float('Number')
            log.add_int('Val', width=4)
            log.add_counter('Count', max_value=12345)
            log.add_time('Time')
            log.add_string('Q', 3)

            log.log(*data)
            log.log(1, 2, 3)    # not enough for more output!
        self.assertOutput(expected=out1, returned=c.text())

    def test_row_by_row(self):
        # Normal use, data row by row
        with StreamCapture() as c:
            log = pints.Logger()
            log.add_counter('#', width=2)
            log.add_float('Latitude', width=1)
            log.add_long_float('Number')
            log.add_int('Val', width=4)
            log.add_counter('Count', max_value=12345)
            log.add_time('Time')
            log.add_string('Q', 3)

            # Add data row by row
            n = 7
            for i in range(len(data) // n):
                log.log(*data[i * n:(i + 1) * n])
        self.assertOutput(expected=out1, returned=c.text())

    def test_field_by_field(self):
        # Normal use, data field by field
        with StreamCapture() as c:
            log = pints.Logger()
            log.add_counter('#', width=2)
            log.add_float('Latitude', width=1)
            log.add_long_float('Number')
            log.add_int('Val', width=4)
            log.add_counter('Count', max_value=12345)
            log.add_time('Time')
            log.add_string('Q', 3)

            # Add data cell by cell
            for d in data:
                log.log(d)
        self.assertOutput(expected=out1, returned=c.text())

    def test_various_chunks(self):
        # Log in different sized chunks
        order = [3, 2, 1, 1, 4, 6, 3, 2, 6]
        self.assertEqual(sum(order), len(data))
        with StreamCapture() as c:
            log = pints.Logger()
            log.add_counter('#', width=2)
            log.add_float('Latitude', width=1)
            log.add_long_float('Number')
            log.add_int('Val', width=4)
            log.add_counter('Count', max_value=12345)
            log.add_time('Time')
            log.add_string('Q', 3)

            # Add data in different sized chunks
            offset = 0
            for n in order:
                log.log(*data[offset:offset + n])
                offset += n
        self.assertOutput(expected=out1, returned=c.text())

    def test_file_only_fields_hidden_on_screen(self):
        # Log with file-only fields, and shorter name
        with StreamCapture() as c:
            log = pints.Logger()
            log.add_counter('#', width=2)
            log.add_float('Lat.', width=1)
            log.add_long_float('Number', file_only=True)
            log.add_int('Val', width=4)
            log.add_counter('Count', max_value=12345)
            log.add_time('Time')
            log.add_string('Q', 3)
            log.log(*data)
        self.assertOutput(expected=out2, returned=c.text())

    def test_file_writing_txt(self):
        # Log with file-only fields, and shorter name, and file
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                log = pints.Logger()
                log.set_filename(filename)
                log.add_counter('#', width=2)
                log.add_float('Lat.', width=1)
                log.add_long_float('Number', file_only=True)
                log.add_int('Val', width=4)
                log.add_counter('Count', max_value=12345)
                log.add_time('Time')
                log.add_string('Q', 3)
                log.log(*data)
                with open(filename, 'r') as f:
                    out = f.read()
        self.assertOutput(expected=out2, returned=c.text())
        self.assertOutput(expected=out3, returned=out)

    def test_file_writing_csv(self):
        # Repeat in csv mode
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.csv')
                log = pints.Logger()
                log.set_filename(filename, csv=True)
                log.add_counter('#', width=2)
                log.add_float('Lat.', width=1)
                log.add_long_float('Number', file_only=True)
                log.add_int('Val', width=4)
                log.add_counter('Count', max_value=12345)
                log.add_time('Time')
                log.add_string('Q', 3)
                log.log(*data)
                with open(filename, 'r') as f:
                    out = f.read()
        self.assertOutput(expected=out2, returned=c.text())
        self.assertOutput(expected=out4, returned=out)

    def test_file_writing_no_screen_csv(self):
        # Repeat without screen output
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.csv')
                log = pints.Logger()
                log.set_filename(filename, csv=True)
                log.set_stream(None)
                log.add_counter('#', width=2)
                log.add_float('Lat.', width=1)
                log.add_long_float('Number', file_only=True)
                log.add_int('Val', width=4)
                log.add_counter('Count', max_value=12345)
                log.add_time('Time')
                log.add_string('Q', 3)
                log.log(*data)
                with open(filename, 'r') as f:
                    out = f.read()
        self.assertOutput(expected='', returned=c.text())
        self.assertOutput(expected=out4, returned=out)

    def test_file_writing_no_screen_txt(self):
        # Repeat without screen output, outside of csv mode
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.csv')
                log = pints.Logger()
                log.set_filename(filename, csv=False)
                log.set_stream(None)
                log.add_counter('#', width=2)
                log.add_float('Lat.', width=1)
                log.add_long_float('Number', file_only=True)
                log.add_int('Val', width=4)
                log.add_counter('Count', max_value=12345)
                log.add_time('Time')
                log.add_string('Q', 3)
                log.log(*data)
                with open(filename, 'r') as f:
                    out = f.read()
        self.assertOutput(expected='', returned=c.text())
        self.assertOutput(expected=out3, returned=out)

        # Unset file output
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.csv')
                log = pints.Logger()
                log.set_filename(filename, csv=False)
                log.set_filename(None)
                log.set_stream(None)
                log.add_counter('#', width=2)
                log.log(1)
                self.assertFalse(os.path.isfile(filename))
        self.assertOutput(expected='', returned=c.text())
        self.assertOutput(expected=out3, returned=out)

    def test_no_output(self):
        # Repeat without any output
        with StreamCapture() as c:
            log = pints.Logger()
            log.set_stream(None)
            log.add_counter('#', width=2)
            log.add_float('Lat.', width=1)
            log.add_long_float('Number', file_only=True)
            log.add_int('Val', width=4)
            log.add_counter('Count', max_value=12345)
            log.add_time('Time')
            log.add_string('Q', 3)
            log.log(*data)
        self.assertOutput(expected='', returned=c.text())

        # Repeat on stderr
        with StreamCapture(stdout=True, stderr=True) as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.csv')
                log = pints.Logger()
                log.set_filename(filename, csv=False)
                log.set_stream(sys.stderr)
                log.add_counter('#', width=2)
                log.add_float('Lat.', width=1)
                log.add_long_float('Number', file_only=True)
                log.add_int('Val', width=4)
                log.add_counter('Count', max_value=12345)
                log.add_time('Time')
                log.add_string('Q', 3)
                log.log(*data)
                with open(filename, 'r') as f:
                    out = f.read()
        self.assertOutput(expected='', returned=c.text()[0])
        self.assertOutput(expected=out2, returned=c.text()[1])
        self.assertOutput(expected=out3, returned=out)

    def assertOutput(self, expected, returned):
        """
        Checks if 2 strings are equal.
        """
        if expected != returned:
            expected = expected.splitlines()
            returned = returned.splitlines()
            ne = len(expected)
            nr = len(returned)
            for k in range(max(ne, nr)):
                print('exp: ' + (expected[k] if k < ne else '') + '|')
                print('ret: ' + (returned[k] if k < nr else '') + '|')
            sys.stdout.flush()
        self.assertEqual(expected, returned)


if __name__ == '__main__':
    unittest.main()
