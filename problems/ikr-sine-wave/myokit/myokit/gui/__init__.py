#
# This hidden module contains the GUI elements used throughout Myokit.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
# Library imports
import os
import sys
import signal
# Detect platform
import platform
platform = platform.system()
# Myokit imports
import myokit
# Select Qt library to use
pyqt4 = False
pyqt5 = False
pyside = False
# Allow overriding automatic selection
if myokit.FORCE_PYQT5:
    pyqt5 = True
elif myokit.FORCE_PYQT4:
    pyqt4 = True
elif myokit.FORCE_PYSIDE:
    pyside = True
else:
    # Automatic selection
    try:
        import PyQt5
        pyqt5 = True
    except ImportError:
        try:
            import PyQt4
            pyqt4 = True
        except ImportError:
            try:
                import PySide
                pyside = True
            except ImportError:
                raise ImportError('Unable to find PyQt5, PyQt4 or PySide')
# Import and configure Qt
if pyqt5:
    # Load PyQt5
    from PyQt5 import QtGui, QtWidgets, QtCore
    from PyQt5.QtCore import Qt
    # Fix PyQt naming issues
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.Property = QtCore.pyqtProperty
    import matplotlib
    matplotlib.use('Qt5Agg')
    matplotlib.rcParams['backend.qt5']='PyQt5'
    import matplotlib.backends.backend_qt5agg as matplotlib_backend
    # Set backend variables
    backend = 'PyQt5'
    qtversion = 5
elif pyqt4:
    # Load PyQt4
    # Set PyQt to "API 2"
    import sip
    sip.setapi('QString', 2)
    sip.setapi('QVariant', 2)
    sip.setapi('QDate', 2)
    sip.setapi('QDateTime', 2)
    sip.setapi('QTextStream', 2)
    sip.setapi('QTime', 2)
    sip.setapi('QUrl', 2)
    # Load main classes
    from PyQt4 import QtGui, QtCore
    from PyQt4.QtCore import Qt
    # Qt5 compatibility
    QtWidgets = QtGui
    QtCore.QItemSelection = QtGui.QItemSelection
    QtCore.QItemSelectionModel = QtGui.QItemSelectionModel
    QtCore.QItemSelectionRange = QtGui.QItemSelectionRange
    QtCore.QSortFilterProxyModel = QtGui.QSortFilterProxyModel
    QtWidgets.QStyleOptionViewItem = QtWidgets.QStyleOptionViewItemV4
    # Fix Qt4 location issue
    import PyQt4.Qt
    QtGui.QKeySequence = PyQt4.Qt.QKeySequence
    QtGui.QTextCursor = PyQt4.Qt.QTextCursor
    del(PyQt4.Qt)
    # Fix PyQt4 naming issues
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.Property = QtCore.pyqtProperty
    # Fix QFileDialog.getOpenFileName return type issues:
    # Return type in PyQt4 is not a tuple (call ...AndFilter methods instead)
    # Fix getOpenFileName
    def gofn(parent=None, caption='', directory='', filter='',
            initialFilter='', options=0):
        if options == 0:
            options = QtWidgets.QFileDialog.Options()
        return QtWidgets.QFileDialog.getOpenFileNameAndFilter(parent, caption,
            directory, filter, initialFilter, options)
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(gofn)
    del(gofn)
    # Fix getOpenFileNames
    def gofns(parent=None, caption='', directory='', filter='',
            initialFilter='', options=0):
        if options == 0:
            options = QtWidgets.QFileDialog.Options()
        return QtWidgets.QFileDialog.getOpenFileNamesAndFilter(parent, caption,
            directory, filter, initialFilter, options)
    QtWidgets.QFileDialog.getOpenFileNames = staticmethod(gofns)
    del(gofns)
    # Fix getSaveFileName
    def gsfn(parent=None, caption='', directory='', filter='',
            initialFilter='', options=0):
        if options == 0:
            options = QtWidgets.QFileDialog.Options()
        return QtWidgets.QFileDialog.getSaveFileNameAndFilter(parent, caption,
            directory, filter, initialFilter, options)
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(gsfn)
    del(gsfn)
    # Configure Matplotlib for use with PyQt4
    import matplotlib
    matplotlib.use('Qt4Agg')
    matplotlib.rcParams['backend.qt4']='PyQt4'
    import matplotlib.backends.backend_qt4agg as matplotlib_backend    
    # Set backend variables
    backend = 'PyQt4'
    qtversion = 4
elif pyside:
    # Load PySide
    # Load main classes
    from PySide import QtGui, QtCore
    from PySide.QtCore import Qt
    # Qt5 compatibility
    QtWidgets = QtGui
    QtCore.QItemSelection = QtGui.QItemSelection
    QtCore.QItemSelectionModel = QtGui.QItemSelectionModel
    QtCore.QItemSelectionRange = QtGui.QItemSelectionRange
    QtCore.QSortFilterProxyModel = QtGui.QSortFilterProxyModel
    QtWidgets.QStyleOptionViewItem = QtWidgets.QStyleOptionViewItemV4
    # Fix QFileDialog.getOpenFileName signature issues (is different in PySide,
    #  which causes issues when using keyword arguments. This is fixed simply
    #  by simply wrapping the methods.)
    # Signature in PySide
    #  parent
    #  caption
    #  dir --> directory PyQt4/5
    #  filter
    #  selectedFilter --> initialFilter in PyQt4/5
    #  options
    # Fix getOpenFileName
    gofn_org = QtWidgets.QFileDialog.getOpenFileName
    def gofn(parent=None, caption='', directory='', filter='',
            initialFilter='', options=0):
        if options == 0:
            options = QtWidgets.QFileDialog.Options()
        return gofn_org(parent, caption, directory, filter, initialFilter,
            options)
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(gofn)
    del(gofn)
    # Fix getOpenFileNames
    gofns_org = QtWidgets.QFileDialog.getOpenFileNames
    def gofns(parent=None, caption='', directory='', filter='',
            initialFilter='', options=0):
        if options == 0:
            options = QtWidgets.QFileDialog.Options()
        return gofns_org(parent, caption, directory, filter, initialFilter,
            options)
    QtWidgets.QFileDialog.getOpenFileNames = staticmethod(gofns)
    del(gofns)
    # Fix getSaveFileName
    gsfn_org = QtWidgets.QFileDialog.getSaveFileName
    def gsfn(parent=None, caption='', directory='', filter='',
            initialFilter='', options=0):
        if options == 0:
            options = QtWidgets.QFileDialog.Options()
        return gsfn_org(parent, caption, directory, filter, initialFilter,
            options)
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(gsfn)
    del(gsfn)
    # Configure Matplotlib for use with PySide
    import matplotlib
    matplotlib.use('Qt4Agg')
    matplotlib.rcParams['backend.qt4']='PySide'
    import matplotlib.backends.backend_qt4agg as matplotlib_backend
    # Set backend variables
    backend = 'PySide'
    qtversion = 4
else:
    raise Exception('Selection of qt version failed.')
# Delete temporary variables
del(pyqt4, pyqt5, pyside)
# Icons with fallback for apple and windows
ICON_PATH = os.path.join(myokit.DIR_DATA, 'gui')
ICONS = {
    'document-new' : 'new.png',
    'document-open' : 'open.png',
    'document-save' : 'save.png',
    'edit-undo' : 'undo.png',
    'edit-redo' : 'redo.png',
    'edit-find' : 'find.png',
    'media-playback-start' : 'run.png',
    }
for k, v in ICONS.iteritems():
    ICONS[k] = os.path.join(ICON_PATH, v)
# Toolbar style suitable for platform
TOOL_BUTTON_STYLE = Qt.ToolButtonTextUnderIcon
if platform == 'Windows':
    TOOL_BUTTON_STYLE = Qt.ToolButtonIconOnly
elif platform == 'Darwin':
    TOOL_BUTTON_STYLE = Qt.ToolButtonTextOnly
# Stand alone applications
class MyokitApplication(QtWidgets.QMainWindow):
    """
    *Extends*: ``QtWidgets.QMainWindow``.
    
    Base class for Myokit applications.
    """
def icon(name):
    """
    Returns a QtIcon created either from the theme or from one of the fallback
    icons.
    
    Raises a ``KeyError`` if no such icon is available.
    """
    return QtGui.QIcon.fromTheme(name, QtGui.QIcon(ICONS[name]))
def qtMonospaceFont():
    """
    Attempts to create and return a monospace font.
    """
    font = QtGui.QFont('monospace')
    #font.setStyleHint(QtGui.QFont.TypeWriter)
    font.setStyleHint(QtGui.QFont.Monospace)
    font.setHintingPreference(QtGui.QFont.PreferVerticalHinting) # Qt5
    return font
def run(app, *args):
    """
    Runs a Myokit gui app as a stand-alone application.
    
    Arguments:
    
    ``app``
        The application to run, specified as a class object (not an instance).
    ``*args``
        Any arguments to pass to the app's constructor.
    
    Example usage:
    
        load(myokit.gui.MyokitIDE, 'model.mmt')
        
    
    """
    # Test application class
    if not issubclass(app, MyokitApplication):
        raise ValueError('Application must be specified as a type extending'
            ' MyokitApplication.')
    # Create Qt app
    a = QtWidgets.QApplication([])
    # Apply custom styling if required
    #_style_application(a)
    # Close with last window
    a.lastWindowClosed.connect(a.quit)
    # Close on Ctrl-C
    def int_signal(signum, frame):
        a.closeAllWindows()
    signal.signal(signal.SIGINT, int_signal)
    # Create app and show
    app = app(*args)
    app.show()
    # For some reason, Qt needs focus to handle the SIGINT catching...
    timer = QtCore.QTimer()
    timer.start(500) # Flags timeout every 500ms
    timer.timeout.connect(lambda: None)
    # Wait for app to exit
    sys.exit(a.exec_())
#def _style_application(app):
#    """
#    Applies some custom styling to a Qt application.
#    
#    (Not required if running the app with :meth:`run`)
#    """
