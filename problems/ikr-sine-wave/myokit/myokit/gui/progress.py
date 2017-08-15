#
# Progress bar for Myokit
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
# Future stuff
from __future__ import division
from __future__ import print_function
# Standard library imports
# Myokit
import myokit
# Qt imports
from myokit.gui import Qt, QtCore, QtGui, QtWidgets
# GUI components
# Constants
N = 100000000
class ProgressBar(QtWidgets.QProgressDialog):
    """
    Progress bar dialog for Myokit. Has a method :meth:`reporter()` that will
    return a :class:`myokit.ProgressReporter` for interfacing with simulations
    and other tasks implementing the ``ProgressReporter`` interface.
    """
    def __init__(self, parent, message):
        super(ProgressBar, self).__init__(message, 'Cancel', 0, N,
            parent=parent)
        self.setWindowModality(Qt.WindowModal)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self._reporter = ProgressBarReporter(self)
    def reporter(self):
        """
        Returns a :class:`ProgressReporter` for his progress bar.
        """
        return self._reporter
    def was_cancelled(self):
        """
        Pythonic version of Qt class.
        """
        return self.wasCanceled()
class ProgressBarReporter(myokit.ProgressReporter):
    """
    A :class:`myokit.ProgressReporter` that sends updates to a
    :class:`ProgressBar`. To use, create a ``ProgressBar`` and then call its
    :meth:`reporter() <ProgressBar.reporter>` method to obtain a linked
    ``ProgressBarReporter``.
    """
    def __init__(self, pd):
        self._pd = pd
    def enter(self, msg=None):
        self._pd.setEnabled(True)
        self._pd.reset()
        if msg is not None:
            self._pd.setLabelText(str(msg))
        self._pd.setValue(0)
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ExcludeUserInputEvents)
    def exit(self):
        self._pd.setEnabled(False)
    def update(self, f):
        self._pd.setValue((int)(N * f))
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ExcludeUserInputEvents)
        return not self._pd.wasCanceled()
