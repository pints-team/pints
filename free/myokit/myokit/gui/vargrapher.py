#
# Source code editor for Myokit.
#
# This code is based in part on examples provided by the PyQt project.
#
# This file is part of Myokit
#  Copyright 2011-2016 Michael Clerx, Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Future stuff
from __future__ import division
from __future__ import print_function
# Standard library imports
# Myokit
import myokit
# Qt imports
from myokit.gui import Qt, QtCore, QtGui, QtWidgets
# Numpy, scipy
import numpy as np
# GUI components
import myokit.gui
# Matplotlib (must be imported _after_ gui has had chance to set backend)
import matplotlib
import matplotlib.figure
from myokit.gui import matplotlib_backend as backend
from mpl_toolkits.mplot3d import axes3d
# Constants
# Classes & methods
class VarGrapher(QtWidgets.QDialog):
    """
    *Extends:* ``QtWidgets.QDialog``

    This widget is used to display graphs of model variables.
    
    Arguments:
    
    ``parent``
        The parent window calling this one.
    ``title``
        A title for this window
    ``var``
        A :class:`myokit.Variable`.
    ``func``
        The python function that calculates ``var``
    ``args``
        The arguments to func, i.e. the LhsExpressions that ``var`` depends on.
        
    
    """
    def __init__(self, parent, title, var, func, args):
        super(VarGrapher, self).__init__(parent)
        self.setFixedSize(700, 600)
        self.setWindowTitle(title)
        # Figure panel
        self._figure = matplotlib.figure.Figure()
        self._canvas = backend.FigureCanvasQTAgg(self._figure)
        self._toolbar = backend.NavigationToolbar2QT(self._canvas, self)
        # Variable panel
        self._variable_widget = QtWidgets.QWidget()
        # Button panel
        self._button_widget = QtWidgets.QWidget()
        # Central widget
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._canvas)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._variable_widget)
        layout.addWidget(self._button_widget)
        self.setLayout(layout)
        # Get function handle, information object
        self._func = func
        self._args = args
        self._var = var
        # Variable ranges
        grid = QtWidgets.QGridLayout()
        self._bounds = {}
        n = len(self._args)
        for k, lhs in enumerate(self._args):
            var = lhs.var()
            # Guess appropriate bounds
            if var.label() == 'membrane_potential' or \
                    var.name().lower() in ['v', 'voltage', 'potential']:
                lohi = (-100.0, 100.0)
            else:
                v = lhs.eval()
                if v >= 0 and v <= 1:
                    lohi = (0.0, 1.0)
                elif v < 0:
                    lohi = (-50, 50)
                else:
                    lohi = (0, 100)
            # Row and column of first widget in grid
            row = k // 2
            col = (k % 2) * 3
            # Add label
            label = QtWidgets.QLabel(var.qname())
            grid.addWidget(label, row, col)
            # Add lower and upper bound or single value
            if k < 2:
                # Lower
                editlo = QtWidgets.QLineEdit()
                editlo.setValidator(QtGui.QDoubleValidator())
                editlo.setText(str(lohi[0]))
                grid.addWidget(editlo, row, col + 1)
                # Upper
                edithi = QtWidgets.QLineEdit()
                edithi.setValidator(QtGui.QDoubleValidator())
                edithi.setText(str(lohi[1]))
                grid.addWidget(edithi, row, col + 2)
                self._bounds[lhs] = (editlo, edithi)
            else:
                # Single, fixed value
                v = 0.5 * (lohi[0] + lohi[1])
                edit = QtWidgets.QLineEdit(str(v))
                edit.setReadOnly(True)
                grid.addWidget(edit, row, col + 1)
                self._bounds[lhs] = (edit, edit)
        self._variable_widget.setLayout(grid)
        # Buttons
        layout = QtWidgets.QHBoxLayout()
        # Graph button
        button = QtWidgets.QPushButton('Refresh')
        button.clicked.connect(self.action_draw)
        layout.addWidget(button)
        # Close button
        button = QtWidgets.QPushButton('Close')
        button.clicked.connect(self.close)
        layout.addWidget(button)
        self._button_widget.setLayout(layout)
        # Draw!
        self.action_draw()
    def action_draw(self):
        """
        (re-)draws the graph
        """
        n = len(self._args)
        if n == 0:
            raise Exception('Cannot graph constant')
        elif n == 1:
            steps = 10000
        else:
            steps = 100
        # Get boundaries
        vrs = []
        for arg in self._args:
            lohi = self._bounds[arg]
            lo = float(lohi[0].text())
            hi = float(lohi[1].text())
            if hi < lo:
                lo, hi = hi, lo
            if lo == hi:
                var = lo
            else:
                var = np.linspace(lo, hi, steps)
            vrs.append(var)
        def label(lhs):
            label = str(lhs)
            unit = lhs.var().unit()
            return label if unit is None else label + ' ' + str(unit)
        # Redraw figure
        try:
            self._figure.clear()
        except Exception:
            pass
        if n == 1:
            ax = self._figure.add_subplot(111)
            x = vrs[0]
            with np.errstate(all='ignore'):
                y = self._func(x)
            ax.plot(x, y)
            ax.grid(True)
            ax.set_xlabel(label(self._args[0]))
            ax.set_ylabel(label(self._var.lhs()))
        else:
            ax = self._figure.add_subplot(111, projection='3d')
            x = vrs[0]
            y = vrs[1]
            x, y = np.meshgrid(x, y)
            vrs[0] = x
            vrs[1] = y
            with np.errstate(all='ignore'):
                z = self._func(*vrs)
            ax.plot_surface(x, y, z)
            ax.grid(True)
            ax.set_xlabel(label(self._args[0]))
            ax.set_ylabel(label(self._args[1]))
            ax.set_zlabel(label(self._var.lhs()))
        # Refresh
        self._canvas.draw()
        self.update()
