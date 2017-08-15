#
# Temporary Qt explorer for Myokit
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
import myokit.gui
import progress
# Matplotlib (must be imported _after_ gui has had chance to set backend)
import matplotlib
import matplotlib.figure
from myokit.gui import matplotlib_backend as backend
# Constants
# Classes & methods
class Explorer(QtWidgets.QDialog):
    """
    *Extends:* ``QtWidgets.QDialog``

    Runs simulations with a model and allows all variables to be graphed.
    
    Arguments:
    
    ``parent``
        The parent window calling this one..
    ``sim_method``
        A method that returns a tuple (:class:`myokit.Model`,
        :class:`myokit.Protocol`, :class:`myokit.Simulation`).
    ``output_stream``
        Something with a write() method.

    """
    def __init__(self, parent, sim_method, output_stream, duration=1000):
        super(Explorer, self).__init__(parent)
        self.setWindowTitle('Myokit Explorer')
        self._sim_method = sim_method
        self._stream = output_stream
        # Set guess for run times
        guess_pre = 0
        guess_run = duration
        # Explorer data
        self._data = None
        self._keys = None
        # Fix background color of line edits
        self.setStyleSheet('QLineEdit{background: white;}')
        # Create top widgets
        label1 = QtWidgets.QLabel('Run unlogged for ')
        label2 = QtWidgets.QLabel( ' and then log for ')
        self._pre_field = QtWidgets.QLineEdit(str(guess_pre))
        self._pre_valid = QtGui.QDoubleValidator()
        self._pre_valid.setBottom(0)
        self._pre_field.setValidator(self._pre_valid)
        self._run_field = QtWidgets.QLineEdit(str(guess_run))
        self._run_valid = QtGui.QDoubleValidator()
        self._run_valid.setBottom(0)
        self._run_field.setValidator(self._run_valid)
        self._clear_button = QtWidgets.QPushButton('Clear graphs')
        self._clear_button.clicked.connect(self.action_clear)
        self._run_button = QtWidgets.QPushButton('Run')
        self._run_button.clicked.connect(self.action_run)
        # Create graph widgets
        self._axes = None
        self._figure = matplotlib.figure.Figure()
        self._canvas = backend.FigureCanvasQTAgg(self._figure)
        self._toolbar = backend.NavigationToolbar2QT(self._canvas, self)
        self._select_x = QtWidgets.QComboBox()
        self._select_x.currentIndexChanged.connect(self.combo_changed)
        self._select_y = QtWidgets.QComboBox()
        self._select_y.currentIndexChanged.connect(self.combo_changed)
        # Create bottom widgets
        self._close_button = QtWidgets.QPushButton('Close')
        self._close_button.clicked.connect(self.action_close)
        # Create button layout
        button_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
        button_layout.addWidget(label1)
        button_layout.addWidget(self._pre_field)
        button_layout.addWidget(label2)
        button_layout.addWidget(self._run_field)
        button_layout.addWidget(self._clear_button)
        button_layout.addWidget(self._run_button)
        # Create graph options layout
        graph_option_layout = QtWidgets.QBoxLayout(
            QtWidgets.QBoxLayout.LeftToRight)
        graph_option_layout.addWidget(self._select_x)
        graph_option_layout.addWidget(self._select_y)
        # Create bottom layout
        bottom_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
        bottom_layout.addWidget(self._close_button)
        # Create central layout
        layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        layout.addLayout(button_layout)
        layout.addLayout(graph_option_layout)
        layout.addWidget(self._canvas)
        layout.addWidget(self._toolbar)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)
    def action_clear(self):
        """
        Clears the current graphs & data (leaves the keys alone).
        """
        self._data = None
        self.action_draw()
    def action_close(self):
        """
        Closes this window.
        """
        self.close()
    def action_draw(self):
        """
        Draws the current graphs.
        """
        if not self._data:
            self._figure.clear()
            self._canvas.draw()
            self.update()
            return
        x = self._select_x.currentText()
        y = self._select_y.currentText()
        if x < 0 or y < 0:
            return
        if self._axes is None:
            # Create new axes
            self._axes = self._figure.add_subplot(1,1,1)
        else:
            # Reset color cycle on axes
            try:
                self._axes.set_prop_cycle(None)
            except AttributeError:
                # Matplotlib < 1.5
                self._axes.set_color_cycle(None)
        for d in self._data:
            self._axes.plot(d[x], d[y])
        self._canvas.draw()
        self.update()
    def action_run(self):
        """
        Runs a simulation and updates the explorer data.
        """
        pbar = progress.ProgressBar(self, 'Creating simulation')
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ExcludeUserInputEvents)
        try:
            pbar.show()
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
            # Create and run simulation
            out = self._sim_method()
            if type(out) == str:
                self._stream.write(out)
                return
            else:
                m, p, s = out
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
            pre = float(self._pre_field.text())
            run = float(self._run_field.text())
            if pre:
                s.pre(pre, progress=pbar.reporter())
            d = s.run(run, progress=pbar.reporter()).npview()
            self._stream.write('Final state: \n' + m.format_state(s.state()))
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
        except myokit.SimulationCancelledError:
            return
        except myokit.MyokitError as e:
            self._stream.write(e.message)
            return
        except Exception as e:
            self._stream.write(str(e))
            return
        finally:
            pbar.close()
            pbar.deleteLater()
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
        # Reset combo-box keys?
        reset_keys = True
        if self._keys:
            # Only reset keys if the old set differs from the new
            if self._keys == set(d.iterkeys()):
                reset_keys = False
        # Append or reset old data
        reset_plot = True
        if self._data:
            # Only reset if we can't append because keys have changed
            reset_plot = reset_keys
        # Perform plot/key reset (order is important!)
        if reset_plot:
            self._data = []
            self._figure.clear()
            self._axes = None
        if reset_keys:
            x = self._select_x.currentText()
            y = self._select_y.currentText()
            self._select_x.clear()
            self._select_y.clear()
            for k in sorted(d.iterkeys()):
                self._select_x.addItem(k)
                self._select_y.addItem(k)
            self._keys = set(d.iterkeys())
            # Attempt to keep current variables
            x = self._select_x.findText(x)
            if x < 0:
                # Guess: Use the time variable
                x = self._select_x.findText(m.time().qname())
            self._select_x.setCurrentIndex(x)
            y = self._select_y.findText(y)
            if y < 0:
                # Guess: First log entry (first state)
                y = self._select_y.findText(d.iterkeys().next())
            self._select_y.setCurrentIndex(y)
        # Add new data
        self._data.append(d)
        # Draw
        self.action_draw()   
    def combo_changed(self, x):
        """
        Combo box changed.
        """
        if self._data:
            if self._axes:
                self._figure.clear()
                self._axes = None
            self.action_draw()
