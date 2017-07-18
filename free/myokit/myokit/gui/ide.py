#
# Graphical interface to myokit. Allows mmt files to be created, modified and
# run.
#
# This file is part of Myokit
#  Copyright 2011-2016 Michael Clerx, Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Future stuff
from __future__ import division
# Standard library imports
import gc
import os
import platform
import traceback
import ConfigParser
# Myokit
import myokit
import myokit.formats
import myokit.lib.deps
# Qt imports
from myokit.gui import QtWidgets, QtGui, QtCore, Qt
# GUI components
import myokit.gui
import source
import explorer
import progress
import vargrapher
# Matplotlib imports
# Matplotlib.pyplot must be imported _after_ myokit.gui has set the backend
import matplotlib
matplotlib.interactive(True) # Allows pl.show()
import matplotlib.pyplot as pl
# Application title
TITLE = 'Myokit IDE'
# Application icon
def icon():
    icons = [
        'icon-ide.ico',
        'icon-ide-16.xpm',
        'icon-ide-24.xpm',
        'icon-ide-32.xpm',
        'icon-ide-48.xpm',
        'icon-ide-64.xpm',
        'icon-ide-96.xpm',
        'icon-ide-128.xpm',
        'icon-ide-256.xpm',
        ]
    icon = QtGui.QIcon()
    for i in icons:
        icon.addFile(os.path.join(myokit.DIR_DATA, 'gui', i))
    return icon
# Settings file
SETTINGS_FILE = os.path.join(myokit.DIR_USER, 'myokit-ide.ini')
# Number of recent files to display
N_RECENT_FILES = 5
# About
ABOUT = '<h1>' + TITLE + '</h1>' + """
<p>
    The Myokit IDE provides a user-friendly environment in which mmt files can
    be created, imported, modified and run.
</p>
<p>
    <a href="http://myokit.org">http://myokit.org</a>
</p>
<p>
    (Currently running on the BACKEND backend.)
</p>
""".replace('BACKEND', myokit.gui.backend)
# File filters
# Note: Using the special filter MMT_SAVE with only one extension specified,
# the file save dialog will add the extension if the user didn't specify one.
FILTER_ALL = 'All files (*.*)'
FILTER_MMT_SAVE = 'Myokit mmt files (*.mmt)'
FILTER_MMT = 'Myokit mmt files (*.mmt);;' + FILTER_ALL
FILTER_CELLML = 'CellML File (*.cellml *.xml);;' + FILTER_ALL
FILTER_HTML = 'Html File (*.html *.htm);;' + FILTER_ALL
FILTER_LATEX = 'Tex File (*.tex)' + FILTER_ALL
FILTER_ABF = 'Axon Binary File (*.abf);;Axon Protocol File (*.pro);;' \
    + FILTER_ALL
# Classes & functions
class MyokitIDE(myokit.gui.MyokitApplication):
    """
    New GUI for editing ``.mmt`` files.
    """
    def __init__(self, filename=None):
        super(MyokitIDE, self).__init__()
        # Set application icon
        self.setWindowIcon(icon())
        # Set size, center
        self.resize(950,720)
        self.setMinimumSize(600, 440)
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        # Status bar
        self._label_cursor = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self._label_cursor)
        self.statusBar().showMessage('Ready')
        # Menu bar
        self.create_menu()
        # Tool bar
        self.create_toolbar() 
        # Create editors and highlighters: The highlighters need to be stored:
        # without a reference to them the python part will be deleted and pyqt
        # (not pyside) gets confused.
        # Create model editor
        self._model_editor = source.Editor()
        self._model_highlighter = source.ModelHighlighter(
            self._model_editor.document())
        self._model_editor.find_action.connect(self.statusBar().showMessage)
        # Create protocol editor
        self._protocol_editor = source.Editor()
        self._protocol_highlighter = source.ProtocolHighlighter(
            self._protocol_editor.document())
        self._protocol_editor.find_action.connect(self.statusBar().showMessage)
        # Create script editor
        self._script_editor = source.Editor()
        self._script_highlighter = source.ScriptHighlighter(
            self._script_editor.document())
        self._script_editor.find_action.connect(self.statusBar().showMessage)
        # Create editor tabs
        self._editor_tabs = QtWidgets.QTabWidget()
        self._editor_tabs.addTab(self._model_editor, 'Model definition')
        self._editor_tabs.addTab(self._protocol_editor, 'Protocol definition')
        self._editor_tabs.addTab(self._script_editor, 'Embedded script')
        # Track changes in mmt file
        self._have_changes = False
        self._model_changed = False
        self._protocol_changed = False
        self._script_changed = False
        self._model_editor.modificationChanged.connect(
            self.change_modified_model)
        self._protocol_editor.modificationChanged.connect(
            self.change_modified_protocol)
        self._script_editor.modificationChanged.connect(
            self.change_modified_script)
        # Track undo/redo button state
        self._editor_tabs.currentChanged.connect(self.change_editor_tab)
        self._model_editor.undoAvailable.connect(self.change_undo_model)
        self._protocol_editor.undoAvailable.connect(self.change_undo_protocol)
        self._script_editor.undoAvailable.connect(self.change_undo_script)
        self._model_editor.redoAvailable.connect(self.change_redo_model)
        self._protocol_editor.redoAvailable.connect(self.change_redo_protocol)
        self._script_editor.redoAvailable.connect(self.change_redo_script)
        # Create console
        self._console = Console()
        self._console.write('Loading Myokit IDE')
        # Create central layout
        self._central_splitter = QtWidgets.QSplitter(Qt.Vertical)
        self._central_splitter.addWidget(self._editor_tabs)
        self._central_splitter.addWidget(self._console)
        self._central_splitter.setSizes([580, 120])
        self.setCentralWidget(self._central_splitter)
        # Code navigator dock
        self._navigator = QtWidgets.QDockWidget("Model components", self)
        self._navigator.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        self._navigator.setAllowedAreas(
            Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._navigator_items = QtWidgets.QListWidget()
        self._navigator.setWidget(self._navigator_items)
        self._navigator_list = []
        self.addDockWidget(Qt.RightDockWidgetArea, self._navigator)
        self._navigator_items.currentItemChanged.connect(
            self.navigator_item_changed)
        self._navigator.hide()
        # Timer to bundle operations after the model text has changed
        self._model_changed_timer = QtCore.QTimer()
        self._model_changed_timer.setSingleShot(True)
        self._model_changed_timer.timeout.connect(self.change_model_timeout)
        # Model explorer (with simulations)
        self._explorer = None
        # Current path, current file, recent files
        self._path = QtCore.QDir.currentPath()
        self._file = None
        self._recent_files = []
        # Load settings from ini file
        self.load_config()
        # Cached validated model and protocol
        self._valid_model = None
        self._valid_protocol = None
        # Last-found model error
        self._last_model_error = None
        # React to changes to model and protocol
        # (For example devalidate model and protocol upon any changes)
        self._model_editor.textChanged.connect(self.change_model)
        self._protocol_editor.textChanged.connect(self.change_protocol)
        # Open select file, recent file or start new
        if filename is not None:
            # Attempt to open selected file. If it doesn't work, show an error
            # message, as this is something the user explicitly requested.
            try:
                self.load_file(filename)
            except Exception:
               self._console.write('Error loading file: ' + str(filename))
               self.show_exception() 
        else:
            if self._file is not None:
                # Try loading the last file, but if it goes wrong continue
                # without error messages
                try:
                    self.load_file(self._file)
                except Exception:
                    self._file = None
            if self._file is None:
                # Create a new file
                self.new_file()
        # Set focus
        self._model_editor.setFocus()
    def action_about(self):
        """
        Displays the about dialog.
        """
        QtWidgets.QMessageBox.about(self, TITLE, ABOUT)
    def action_check_units_tolerant(self):
        """
        Perform a unit check in tolerant mode.
        """
        try:
            model = self.model(errors_in_console=True)
            if not model:
                return
            model.check_units(mode=myokit.UNIT_TOLERANT)
            self._console.write('Units ok! (checked in tolerant mode)')
        except myokit.IncompatibleUnitError as e:
            self._console.write(e.message)
        except Exception:
            self.show_exception()
    def action_check_units_strict(self):
        """
        Perform a unit check in strict mode.
        """
        try:
            model = self.model(errors_in_console=True)
            if not model:
                return
            model.check_units(mode=myokit.UNIT_STRICT)
            self._console.write('Units ok! (checked in strict mode)')
        except myokit.IncompatibleUnitError as e:
            self._console.write(e.message)
        except Exception:
            self.show_exception()
    def action_clear_units(self):
        """
        Remove all units from expressions in this model.
        """
        try:
            # Ask are you sure?
            msg = 'Remove all units from expressions in model?'
            box = QtWidgets.QMessageBox
            options = box.Yes | box.No
            reply = box.question(self, TITLE, msg, options)
            if reply == box.No:
                return
            # Strip units
            # Note: lines are used in error handling!
            lines = self._model_editor.get_text().splitlines()
            text = myokit.strip_expression_units(lines)
            self._model_editor.replace(text)
            self._console.write('Removed all expression units.')
        except myokit.ParseError as e:
            self.statusBar().showMessage('Error parsing model')
            self._console.write(myokit.format_parse_error(e, lines))
        except myokit.IntegrityError as e:
            self.statusBar().showMessage('Model integrity error')
            self._console.write('Model integrity error:')
            self._console.write(e.message)
        except Exception:
            self.show_exception()
    def action_comment(self):
        """
        Comments or uncomments the currently selected lines.
        """
        self._editor_tabs.currentWidget().toggle_comment()            
    def action_component_cycles(self):
        """
        Checks for interdependency-cycles amongst the components and displays
        them if found.
        """
        try:
            # Validate model
            model = self.model(errors_in_console=True)
            if not model:
                return
            # Check for component cycles
            if model.has_interdependent_components():
                cycles = model.component_cycles()
                cycles = ['  ' + ' > '.join([x.name() for x in c])
                            for c in cycles]
                cycles = ['Found component cycles:'] + cycles
                self._console.write('\n'.join(cycles))
            else:
                self._console.write('No component cycles found.')
        except Exception:
            self.show_exception()
    def action_component_dependency_graph(self):
        """
        Displays a component dependency graph
        """
        try:
            model = self.model(errors_in_console=True)
            if not model:
                return
            f = pl.figure()
            a = f.add_subplot(1,1,1)
            myokit.lib.deps.plot_component_dependency_graph(model, axes=a,
                omit_states=True, omit_constants=True)
            pl.show()
        except Exception:
            self.show_exception()
    def action_explore(self):
        """
        Opens the explorer
        """
        # Simulation creation method
        def sim():
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
            try:
                # Get model and protocol
                m = self.model(errors_in_console=True)
                if m == False:
                    return 'Errors in model'
                if m is None:
                    return 'Empty model definition'
                QtWidgets.QApplication.processEvents(
                    QtCore.QEventLoop.ExcludeUserInputEvents)
                p = self.protocol(errors_in_console=True)
                if p == False:
                    return 'Errors in protocol'
                QtWidgets.QApplication.processEvents(
                    QtCore.QEventLoop.ExcludeUserInputEvents)
                # Create and return simulation
                self.statusBar().showMessage('Creating simulation...')
                return m, p, myokit.Simulation(m, p)
            except Exception:
                self.show_exception()
        try:
            # Create explorer or update existing one
            if self._explorer is None:
                self._explorer = explorer.Explorer(self, sim, self._console)
            self._explorer.show()
        except Exception:
            self.show_exception()
    def action_export_model(self, name, glob=None):
        """
        Exports the model to a file.
        
        Arguments:
        
        ``name``
            The exporter name
        ``filter``
            A filter for the file selection method.
        """
        try:
            m = self.model(errors_in_console=True)
            if m == False:
                return
            e = myokit.formats.exporter(name)
            if not e.supports_model():
                raise Exception('Exporter does not support export of model')
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 
                'Select file to export to', self._path, filter=glob)[0]
            if not filename:
                return
            try:
                e.model(filename, m)
                msg = 'Export successful.'
                e.log(e.info())
            except myokit.ExportError:
                msg = 'Export failed.'
            self._console.write(msg + '\n' + e.text())
        except Exception:
            self.show_exception()
    def action_export_runnable(self, name):
        """
        Exports the model and optionally the protocol to a directory.
        
        Arguments:
        
        ``name``
            The exporter name.
        
        """
        try:
            # Get model & protocol
            m = self.model(errors_in_console=True)
            if m is False:
                return
            p = self.protocol(errors_in_console=True)
            if p is False:
                return
            # Create exporter & test compatibility
            e = myokit.formats.exporter(name)
            if not e.supports_runnable():
                raise Exception('Exporter does not support export of runnable')
            # Select dir
            path = QtWidgets.QFileDialog.getSaveFileName(self,
                'Create directory', self._path)[0]
            if not path:
                return
            try:
                e.runnable(path, m, p)
                msg = 'Export successful.'
                e.log(e.info())
            except myokit.ExportError:
                msg = 'Export failed.'
            self._console.write(msg + '\n' + e.text())
        except Exception:
            self.show_exception()
    def action_find(self):
        """
        Display a find/replace dialog for the current editor
        """
        current = self._editor_tabs.currentWidget()
        es = (self._model_editor, self._protocol_editor, self._script_editor)
        for editor in es:
            if editor != current:
                editor.hide_find_dialog()
        current.activate_find_dialog()
    def action_import_abf_protocol(self):
        """
        Imports a protocol from an abf (v2) file.
        """
        try:
            if not self.prompt_save_changes(cancel=True):
                return
            filename = QtWidgets.QFileDialog.getOpenFileName(self, 
                'Open ABF file', self._path, filter=FILTER_ABF)[0]
            if not filename:
                return
            # Load file
            i = myokit.formats.importer('abf')
            try:
                protocol = i.protocol(filename)
                # Import okay, update interface
                self.new_file()
                self._protocol_editor.setPlainText(protocol.code())
                self._console.write('Protocol imported successfully.\n'
                    + i.text())
                # Set working directory to file's path
                self._path = os.path.dirname(filename)
                os.chdir(self._path)
                # Save settings file
                try:
                    self.save_config()
                except Exception:
                    pass
                # Update interface
                self._tool_save.setEnabled(True)
                self.update_window_title()
            except myokit.ImportError:
                self._console.write('Protocol import failed.\n' + i.text())
                self.statusBar().showMessage('Protocol import failed.')
        except Exception:
            self.show_exception()
    def action_import_cellml(self):
        """
        Imports a CellML model
        """
        try:
            if not self.prompt_save_changes(cancel=True):
                return
            filename = QtWidgets.QFileDialog.getOpenFileName(self,
                'Open CellML file', self._path, filter=FILTER_CELLML)[0]
            if not filename:
                return
            # Load file
            i = myokit.formats.importer('cellml')
            try:
                model = i.model(filename)
                # Import okay, update interface
                self.new_file()
                self._model_editor.setPlainText(model.code())
                # Write log to console
                i.log_warnings()
                self._console.write('Model imported successfully.\n'+i.text())
                # Set working directory to file's path
                self._path = os.path.dirname(filename)
                os.chdir(self._path)
                # Save settings file
                try:
                    self.save_config()
                except Exception:
                    pass
                # Update interface
                self._tool_save.setEnabled(True)
                self.update_window_title()
            except myokit.ImportError as e:
                # Write output to console
                i.log_warnings()
                self._console.write('Model import failed.\n' + i.text()
                    + '\n\nModel import failed.\n' + e.message)
                self.statusBar().showMessage('Model import failed.')
        except Exception:
            self.show_exception()
    def action_jump_to_error(self):
        """
        Jump tot the last error in the model tab.
        """
        try:
            # Check for error
            self.model(console=True)
            if self._last_model_error is None:
                return
            # Switch to model tab if required
            self._editor_tabs.setCurrentWidget(self._model_editor)
            # Show error
            line = self._last_model_error.line
            char = self._last_model_error.char
            self.statusBar().showMessage('Jumping to (' + str(line) + ','
                + str(char) + ').')
            self._model_editor.jump_to(line - 1, char)
        except Exception:
            self.show_exception()
    def action_license(self):
        """
        Displays this program's licensing information.
        """
        QtWidgets.QMessageBox.about(self, TITLE, myokit.LICENSE_HTML)
    def action_model_stats(self):
        """
        Gathers and displays some basic information about the current model.
        """
        try:
            self.statusBar().showMessage('Gathering model statistics')
            # Get model and editor code
            model = self.model()
            code = self._model_editor.get_text()
            # Create text
            text = []
            text.append('Model statistics')
            text.append('----------------')
            # Add statistics about the model code
            if model:
                text.append('Name: ' + model.name())
            text.append('Number of lines: ' + str(len(code.splitlines())))
            code = code.replace('\n', '')
            text.append('Number of characters: ' + str(len(code)))
            code = code.replace(' ', '')
            text.append('  without whitespace: ' + str(len(code)))
            # Add statistics about the model
            if model is None:
                text.append('No model to parse')
            elif model is False:
                text.append('Unable to parse model')
            else:
                text.append('Number of components: '
                    + str(model.count_components()))
                text.append('Number of variables: '
                    + str(model.count_variables(deep=True)))
                text.append('              bound: '
                    + str(model.count_variables(bound=True, deep=True)))
                text.append('              state: '
                    + str(model.count_variables(state=True, deep=True)))
                text.append('       intermediary: '
                    + str(model.count_variables(inter=True, deep=True)))
                text.append('           constant: '
                    + str(model.count_variables(const=True, deep=True)))
            self._console.write('\n'.join(text))
        except Exception:
            self.statusBar().showMessage('"New file" failed.')
            self.show_exception()
    def action_new(self):
        """
        Create a new model, closing any current one
        """
        try:
            # Attempt to save changes, allow user to cancel
            if not self.prompt_save_changes(cancel=True):
                return
            self.new_file()
        except Exception:
            self.statusBar().showMessage('"New file" failed.')
            self.show_exception()
    def action_open(self):
        """
        Select and open an existing file.
        """
        try:
            if not self.prompt_save_changes(cancel=True):
                return
            filename = QtWidgets.QFileDialog.getOpenFileName(self,
                'Open mmt file', self._path, filter=FILTER_MMT)[0]
            if filename:
                self.load_file(filename)
        except Exception:
            self.statusBar().showMessage('"Open file" failed')
            self.show_exception()
    def action_open_recent(self):
        """
        Opens a recent file.
        """
        try:
            if not self.prompt_save_changes(cancel=True):        
                return
            action = self.sender()
            if action:
                filename = str(action.data())
                if not os.path.isfile(filename):
                    self._console.write('Failed to load file. The selected'
                        ' file can not be found: ' + str(filename))
                else:
                    self.load_file(filename)
        except Exception:
            self.statusBar().showMessage('"Open recent file" failed')
            self.show_exception()
    def action_preview_protocol(self):
        """
        Displays a preview of the current protocol.
        """
        try:
            p = self.protocol(errors_in_console=True)
            if p is False:
                self._console.write('Can\'t display preview: Errors in'
                    ' protocol.')
            elif p is None:
                self._console.write('Can\'t display preview: No protocol'
                    ' specified.')
            else:
                a = 0
                b = p.characteristic_time()
                if b == 0:
                    b = 1000
                d = p.create_log_for_interval(a, b, for_drawing=True)
                pl.figure()
                pl.plot(d['time'], d['pace'])
                lo, hi = p.range()
                if lo == 0 and hi == 1:
                    pl.ylim(-0.1, 1.1)
                else:
                    r = (hi - lo) * 0.1
                    pl.ylim(lo - r, hi + r)
                pl.show()
        except Exception:
            self.show_exception()
    def action_redo(self):
        """
        Redoes the previously undone text edit operation.
        """
        self._editor_tabs.currentWidget().redo()
    def action_run(self):
        """
        Runs the embedded script.
        """
        pbar = None
        try:
            # Prepare interface
            self.setEnabled(False)
            self._console.write('Running embedded script.')
            QtWidgets.QApplication.setOverrideCursor(
                QtGui.QCursor(Qt.WaitCursor))
            # Create progress bar
            pbar = progress.ProgressBar(self, 'Running embedded script')
            pbar.show()
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
            # Get model and protocol
            m = self.model(errors_in_console=True)
            if m is False:
                return
            p = self.protocol(errors_in_console=True)
            if p is False:
                return
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
            # Clone model & protocol: the script may modify them!
            if m:
                m = m.clone()
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
            if p:
                p = p.clone()
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
            # Get embedded script
            x = self._script_editor.get_text()
            # Run
            try:
                myokit.run(m, p, x,
                    stdout = self._console,
                    stderr =self._console,
                    progress = pbar.reporter(),
                    )
                # Update user
                self._console.write('Done.')
                # Garbage collection
                gc.collect()
            except myokit.SimulationCancelledError:
                self._console.write('Simulation cancelled by user.')
            except Exception:
                self._console.write('An error has occurred')
                self._console.write(traceback.format_exc())
        finally:
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents)
            # Hide progress bar
            if pbar is not None:
                pbar.close()
                pbar.deleteLater()
            # Work-around for cursor bug on linux
            pos = QtGui.QCursor.pos()
            QtGui.QCursor.setPos(0, 0)
            QtGui.QCursor.setPos(pos)
            # Re-enable
            self.setEnabled(True)
            # Set focus on editor
            self._editor_tabs.currentWidget().setFocus()
            # Fix cursor
            QtWidgets.QApplication.restoreOverrideCursor()
    def action_save(self):
        """
        Save the current file.
        """
        self.save_file(save_as=False)
    def action_save_as(self):
        """
        Save the current file under a different name.
        """
        self.save_file(save_as=True)
    def action_show_or_hide_navigator(self):
        """
        Show or hide the model navigator.
        """
        if (self._tool_view_navigator.isChecked() and
                self._editor_tabs.currentWidget() == self._model_editor):
            # Update navigator and show
            self.update_navigator()
            self._navigator.show()
        else:
            self._navigator.hide()
    def action_state_derivatives(self):
        """
        Evaluates all the state derivatives and displays the results. Numerical
        erorrs raised will be displayed.
        """
        self._action_state_derivatives_inner(False)
    def action_state_derivatives2(self):
        """
        Evaluates all the state derivatives and displays the results. Numerical
        errors are ignored.
        """
        self._action_state_derivatives_inner(True)
    def _action_state_derivatives_inner(self, ignore_errors):
        """
        Evaluates all the state derivatives and displays the results.
        """
        try:
            m = self.model(errors_in_console=True)
            if m is None:
                self._console.write('No model found')
                return
            elif m == False:
                self._console.write('Errors found in model. Please fix any'
                    ' remaining issues before using the method.')
                return
            try:
                self._console.write(myokit.step(m,
                    ignore_errors=ignore_errors))
            except myokit.NumericalError as ee:
                self._console.write('A numerical error occurred:')
                self._console.write(ee.message)
        except Exception:
            self.show_exception()
    def action_state_matrix(self):
        """
        Displays a state dependency matrix.
        """
        try:
            # Validate model
            model = self.model(errors_in_console=True)
            if not model:
                return
            # Show graph
            f = pl.figure()
            a = f.add_subplot(1,1,1)
            myokit.lib.deps.plot_state_dependency_matrix(model, axes=a)
            # Tweak figure margins (doesn't always work)
            f.subplots_adjust(
                left = 0.1,
                right = 1.0,
                bottom = 0.08,
                top = 0.77,
                wspace=0,
                hspace=0)
            pl.show()
        except Exception:
            self.show_exception()
    def action_trim_whitespace(self):
        """
        Trims any trailing whitespace from the current editor.
        """
        self._editor_tabs.currentWidget().trim_trailing_whitespace()
        self._console.write('Trailing whitespace removed.')
        self.statusBar().showMessage('Trailing whitespace removed.')
    def action_undo(self):
        """
        Undoes the previous text edit operation.
        """
        self._editor_tabs.currentWidget().undo()
    def action_validate(self):
        """
        Validates the model and, if the model is valid, the protocol.
        """
        try:
            if not self.model(console=True):
                return
            self.protocol(console=True)
        except Exception:
            self.show_exception()
    def action_variable_dependencies(self):
        """
        Finds the variable pointed at by the cursor in the model editor and
        displays all expressions required for its calculation.
        """
        try:
            if self._editor_tabs.currentWidget() != self._model_editor:
                self._console.write('Variable info can only be displayed'
                    ' for model variables.')
                return
            var = self.selected_variable()
            if var == False:
                return # Model error
            elif var is None:
                self._console.write('No variable selected. Please select a'
                    ' variable in the model editing tab.')
                return
            self._console.write(var.model().show_expressions_for(var))
        except Exception:
            self.show_exception()
    def action_variable_dependency_graph(self):
        """
        Displays a variable dependency graph
        """
        try:
            model = self.model(errors_in_console=True)
            if not model:
                return
            f = pl.figure()
            a = f.add_subplot(1,1,1)
            myokit.lib.deps.plot_variable_dependency_graph(model, axes=a)
            pl.show()
        except Exception:
            self.show_exception()
    def action_variable_evaluation(self):
        """
        Finds the variable pointed at by the cursor in the model editor and
        displays its calculation.
        """
        try:
            if self._editor_tabs.currentWidget() != self._model_editor:
                self._console.write('Variable info can only be displayed'
                    ' for model variables.')
                return
            var = self.selected_variable()
            if var == False:
                return # Model error
            elif var is None:
                self._console.write('No variable selected. Please select a'
                    ' variable in the model editing tab.')
                return
            self._console.write(var.model().show_evaluation_of(var))
        except Exception:
            self.show_exception()
    def action_variable_graph(self):
        """
        Attempts to graph the variable pointed at by the cursor in the model
        editor.
        """
        try:
            if self._editor_tabs.currentWidget() != self._model_editor:
                self._console.write('Only variables on the model editing tab'
                    ' can be graphed.')
            var = self.selected_variable()
            if var == False:
                return # Model editor
            elif var is None:
                self._console.write('No variable selected. Please select a'
                    ' variable in the model editing tab.')
                return
            if var.is_constant():
                self._console.write('Cannot graph constants.')
                return
            elif var.is_bound():
                self._console.write('Cannot graph bound variables.')
                return
            f, a = var.pyfunc(arguments=True)
            title = 'Graphing ' + var.lhs().var().qname()
            grapher = vargrapher.VarGrapher(self, title, var, f, a)
            grapher.show()
        except Exception:
            self.show_exception()
    def action_variable_info(self):
        """
        Finds the variable pointed at by the cursor and displays its type and
        the line on which it is defined.
        """
        try:
            if self._editor_tabs.currentWidget() != self._model_editor:
                self._console.write('Variable info can only be displayed'
                    ' for model variables.')
                return
            var = self.selected_variable()
            if var == False:
                return # Model error
            elif var is None:
                self._console.write('No variable selected. Please select a'
                    ' variable in the model editing tab.')
                return
            self._console.write(var.model().show_line(var))
        except Exception:
            self.show_exception()
    def action_variable_users(self):
        """
        Finds the variable pointed at by the cursor in the model editor and
        displays all variables that depend on it.
        """
        try:
            if self._editor_tabs.currentWidget() != self._model_editor:
                self._console.write('Variable info can only be displayed'
                    ' for model variables.')
                return
            var = self.selected_variable()
            if var == False:
                return # Model error
            elif var is None:
                self._console.write('No variable selected. Please select a'
                    ' variable in the model editing tab.')
                return
            out = []
            if var.is_state():
                name = str(var.lhs())
                users = list(var.refs_by(state_refs=False))
                if users:
                    out.append('The following variables depend on '+ name +':')
                    for v in sorted([v.qname() for v in users]):
                        out.append('  ' + v)
                else:
                    out.append('No variables depend on ' + name + '.')
            name = var.qname()
            users = list(var.refs_by(state_refs=var.is_state()))
            if users:
                out.append('The following variables depend on ' + name + ':')
                for v in sorted([v.qname() for v in users]):
                    out.append('  ' + v)
            else:
                out.append('No variables depend on ' + name + '.')
            self._console.write('\n'.join(out))
        except Exception:
            self.show_exception()
    def action_view_model(self):
        """
        View the model tab.
        """
        self._editor_tabs.setCurrentWidget(self._model_editor)
        self._model_editor.setFocus()
    def action_view_protocol(self):
        """
        View the protocol tab.
        """
        self._editor_tabs.setCurrentWidget(self._protocol_editor)
        self._protocol_editor.setFocus()
    def action_view_script(self):
        """
        View the script tab.
        """
        self._editor_tabs.setCurrentWidget(self._script_editor)
        self._script_editor.setFocus()
    def add_recent_file(self, filename):
        """
        Adds the given filename to the list of recent files.
        """
        try:
            # Remove filename from recent files list
            i = self._recent_files.index(filename)
            self._recent_files = self._recent_files[:i] \
                + self._recent_files[i+1:]
        except ValueError:
            pass
        self._recent_files.insert(0, filename)
        self._recent_files = self._recent_files[:N_RECENT_FILES]
        self.update_recent_files_menu()
    def change_editor_tab(self, index):
        """
        Qt slot: Called when the editor tab is changed.
        """
        # Update undo/redo
        d = self._editor_tabs.currentWidget().document()
        self._tool_undo.setEnabled(d.isUndoAvailable())
        self._tool_redo.setEnabled(d.isRedoAvailable())
        # Hide find/replace dialogs
        es = (self._model_editor, self._protocol_editor, self._script_editor)
        for editor in es:
            editor.hide_find_dialog()
        # Show/hide model navigator
        if index == 0 and self._tool_view_navigator.isChecked():
            self._navigator.show()
        else:
            self._navigator.hide()
    def change_model(self):
        """
        Qt slot: Called whenever the model is changed.
        """
        self._valid_model = None
        # Bundle events in one-shot timer that calls change_model_timeout
        # Successive calls will restart the timer!
        self._model_changed_timer.start(100) # in ms
    def change_model_timeout(self):
        """
        Called with a slight delay after a change to the model.
        """
        if self._tool_view_navigator.isChecked():
            self.update_navigator()
    def change_modified_model(self, have_changes):
        """
        Qt slot: Called when the model modified state is changed.
        """
        # Update have_changes status
        self._model_changed = have_changes
        self._have_changes = (self._model_changed or self._protocol_changed or
            self._script_changed)
        # Update button states
        self._tool_save.setEnabled(self._have_changes)
        # Update window title
        self.update_window_title()
    def change_modified_protocol(self, have_changes):
        """
        Qt slot: Called when the protocol modified state is changed.
        """
        # Update have_changes status
        self._protocol_changed = have_changes
        self._have_changes = (self._model_changed or self._protocol_changed or
            self._script_changed)
        # Update button states
        self._tool_save.setEnabled(self._have_changes)
        # Update window title
        self.update_window_title()
    def change_modified_script(self, have_changes):
        """
        Qt slot: Callend when the script modified state is changed.
        """
        # Update have_changes status
        self._script_changed = have_changes
        self._have_changes = (self._model_changed or self._protocol_changed or
            self._script_changed)
        # Update button states
        self._tool_save.setEnabled(self._have_changes)
        # Update window title
        self.update_window_title()
    def change_protocol(self):
        """
        Qt slot: Called whenever the protocol is changed.
        """
        self._valid_protocol = None     
    def change_redo_model(self, enabled):
        """
        Qt slot: Redo state of model editor changed.
        """
        if self._editor_tabs.currentWidget() == self._model_editor:
            self._tool_redo.setEnabled(enabled)
    def change_redo_protocol(self, enabled):
        """
        Qt slot: Redo state of protocol editor changed.
        """
        if self._editor_tabs.currentWidget() == self._protocol_editor:
            self._tool_redo.setEnabled(enabled)
    def change_redo_script(self, enabled):
        """
        Qt slot: Redo state of script editor changed.
        """
        if self._editor_tabs.currentWidget() == self._script_editor:
            self._tool_redo.setEnabled(enabled)
    def change_undo_model(self, enabled):
        """
        Qt slot: Undo state of model editor changed.
        """
        if self._editor_tabs.currentWidget() == self._model_editor:
            self._tool_undo.setEnabled(enabled)
    def change_undo_protocol(self, enabled):
        """
        Qt slot: Undo state of protocol editor changed.
        """
        if self._editor_tabs.currentWidget() == self._protocol_editor:
            self._tool_undo.setEnabled(enabled)
    def change_undo_script(self, enabled):
        """
        Qt slot: Undo state of script editor changed.
        """
        if self._editor_tabs.currentWidget() == self._script_editor:
            self._tool_undo.setEnabled(enabled)
    def closeEvent(self, event=None):
        """
        Called when window is closed. To force a close (and trigger this
        function, call self.close())
        """
        try:
            self.save_config()
        except Exception:
            pass
        # Save changes?
        if not self.prompt_save_changes(cancel=False):
            # Something went wrong when saving changes or use wants to abort
            if event:
                event.ignore()
            return
        # Close all windows, including matplotlib plots
        QtWidgets.qApp.closeAllWindows()
        # Accept event, closing this window
        if event:
            event.accept()
    def close_explorer(self):
        """
        Closes the explorer, if any.
        """
        if self._explorer is None:
            return
        self._explorer.close()
        self._explorer.deleteLater()
        self._explorer = None
    def create_menu(self):
        """
        Creates this widget's menu.
        """
        self._menu = self.menuBar()
        # File menu
        self._menu_file = self._menu.addMenu('&File')
        # File > New
        self._tool_new = QtWidgets.QAction('&New', self)
        self._tool_new.setShortcut('Ctrl+N')
        self._tool_new.setStatusTip('Create a new mmt file.')
        self._tool_new.setIcon(myokit.gui.icon('document-new'))
        self._tool_new.triggered.connect(self.action_new)
        self._menu_file.addAction(self._tool_new)
        # File > Open
        self._tool_open = QtWidgets.QAction('&Open', self)
        self._tool_open.setShortcut('Ctrl+O')
        self._tool_open.setStatusTip('Open an existing mmt file.')
        self._tool_open.setIcon(myokit.gui.icon('document-open'))
        self._tool_open.triggered.connect(self.action_open)
        self._menu_file.addAction(self._tool_open)
        # File > ----
        self._menu_file.addSeparator()
        # File > Save
        self._tool_save = QtWidgets.QAction('&Save', self)
        self._tool_save.setShortcut('Ctrl+S')
        self._tool_save.setStatusTip('Save the current file')
        self._tool_save.setIcon(myokit.gui.icon('document-save'))
        self._tool_save.triggered.connect(self.action_save)
        self._tool_save.setEnabled(False)
        self._menu_file.addAction(self._tool_save)
        # File > Save as
        self._tool_save_as = QtWidgets.QAction('Save &as', self)
        self._tool_save_as.setShortcut('Ctrl+Shift+S')
        self._tool_save_as.setStatusTip('Save the current file under a'
            ' different name.')
        self._tool_save_as.triggered.connect(self.action_save_as)
        self._menu_file.addAction(self._tool_save_as)
        # File > ----
        self._menu_file.addSeparator()
        # File > Recent files
        self._recent_file_tools = []
        for i in xrange(N_RECENT_FILES):
            tool = QtWidgets.QAction(self, visible=False)
            tool.triggered.connect(self.action_open_recent)
            self._recent_file_tools.append(tool)
            self._menu_file.addAction(tool)
        # File > ----
        self._menu_file.addSeparator()
        # File > Quit
        self._tool_exit = QtWidgets.QAction('&Quit', self)
        self._tool_exit.setShortcut('Ctrl+Q')
        self._tool_exit.setStatusTip('Exit application.')
        self._tool_exit.triggered.connect(self.close)
        self._menu_file.addAction(self._tool_exit)
        #
        # Edit menu
        #
        self._menu_edit = self._menu.addMenu('&Edit')
        # Edit > Undo
        self._tool_undo = QtWidgets.QAction('&Undo', self)
        self._tool_undo.setShortcut('Ctrl+Z')
        self._tool_undo.setStatusTip('Undo the last edit.')
        self._tool_undo.setIcon(myokit.gui.icon('edit-undo'))
        self._tool_undo.triggered.connect(self.action_undo)
        self._tool_undo.setEnabled(False)
        self._menu_edit.addAction(self._tool_undo)
        # Edit > Redo
        self._tool_redo = QtWidgets.QAction('&Redo', self)
        self._tool_redo.setShortcut('Ctrl+Shift+Z')
        self._tool_redo.setStatusTip('Redo the last undone edit.')
        self._tool_redo.setIcon(myokit.gui.icon('edit-redo'))
        self._tool_redo.triggered.connect(self.action_redo)
        self._tool_redo.setEnabled(False)
        self._menu_edit.addAction(self._tool_redo)
        # Edit > ----
        self._menu_edit.addSeparator()
        # Edit > Find and replace
        self._tool_find = QtWidgets.QAction('&Find and replace', self)
        self._tool_find.setShortcut('Ctrl+F')
        self._tool_find.setStatusTip('Find and/or replace some text.')
        self._tool_find.setIcon(myokit.gui.icon('edit-find'))
        self._tool_find.triggered.connect(self.action_find)
        self._menu_edit.addAction(self._tool_find)
        # Edit > ----
        self._menu_edit.addSeparator()
        # Edit > Comment or uncomment
        self._tool_comment = QtWidgets.QAction(
            '&Comment/uncomment selected lines', self)
        self._tool_comment.setShortcut('Ctrl+;')
        self._tool_comment.setStatusTip('Comments or uncomments the currently'
            ' selected lines.')
        self._tool_comment.triggered.connect(self.action_comment)
        self._menu_edit.addAction(self._tool_comment)
        # Edit > Remove units from expressions
        self._tool_remove_units = QtWidgets.QAction('Remove units from'
            ' &expressions', self)
        self._tool_remove_units.setStatusTip('Remove all units inside'
            ' expressions.')
        self._tool_remove_units.triggered.connect(self.action_clear_units)
        self._menu_edit.addAction(self._tool_remove_units)
        # Edit > Trim whitespace
        self._tool_trim_whitespace = QtWidgets.QAction(
            'Trim trailing &whitespace', self)
        self._tool_trim_whitespace.setStatusTip('Remove trailing whitespace'
            ' from each line.')
        self._tool_trim_whitespace.triggered.connect(
            self.action_trim_whitespace)
        self._menu_edit.addAction(self._tool_trim_whitespace)
        #
        # View menu
        #
        self._menu_view = self._menu.addMenu('&View')
        # View > View model definition
        self._tool_view_model = QtWidgets.QAction('View &model definition',
            self)
        self._tool_view_model.setShortcut('Alt+1')
        self._tool_view_model.setStatusTip('View the model definition tab')
        self._tool_view_model.triggered.connect(self.action_view_model)
        self._menu_view.addAction(self._tool_view_model)
        # View > View protocol definition
        self._tool_view_protocol = QtWidgets.QAction(
            'View &protocol definition', self)
        self._tool_view_protocol.setShortcut('Alt+2')
        self._tool_view_protocol.setStatusTip(
            'View the protocol definition tab')
        self._tool_view_protocol.triggered.connect(self.action_view_protocol)
        self._menu_view.addAction(self._tool_view_protocol)
        # View > View embedded script
        self._tool_view_script = QtWidgets.QAction('View embedded &script',
            self)
        self._tool_view_script.setShortcut('Alt+3')
        self._tool_view_script.setStatusTip('View the embedded script tab')
        self._tool_view_script.triggered.connect(self.action_view_script)
        self._menu_view.addAction(self._tool_view_script)
        # View > ----
        self._menu_view.addSeparator()
        # View > Show model components (navigator)
        self._tool_view_navigator = QtWidgets.QAction('Show model &components',
            self)
        self._tool_view_navigator.setCheckable(True)
        self._tool_view_navigator.setStatusTip('Shows or hides the model'
            + ' navigator pane.')
        self._tool_view_navigator.triggered.connect(
            self.action_show_or_hide_navigator)
        self._menu_view.addAction(self._tool_view_navigator)
        # View > ----
        self._menu_view.addSeparator()
        # View > Preview protocol
        self._tool_preview_protocol = QtWidgets.QAction('&Preview protocol',
            self)
        self._tool_preview_protocol.setShortcut('Ctrl+P')
        self._tool_preview_protocol.setStatusTip('Show a preview of the'
            ' current protocol.')
        self._tool_preview_protocol.triggered.connect(
            self.action_preview_protocol)
        self._menu_view.addAction(self._tool_preview_protocol)
        #
        # Convert menu
        #
        self._menu_convert = self._menu.addMenu('&Convert')
        # Convert > Import CellML
        self._tool_import_cellml = QtWidgets.QAction(
            'Import model from CellML', self)
        self._tool_import_cellml.setStatusTip('Import a model definition from'
            ' a CellML file.')
        self._tool_import_cellml.triggered.connect(self.action_import_cellml)
        self._menu_convert.addAction(self._tool_import_cellml)
        # Convert > Import ABF
        self._tool_import_abf = QtWidgets.QAction('Import protocol from ABF',
            self)
        self._tool_import_abf.setStatusTip('Import a protocol definition from'
            ' an ABF file.')
        self._tool_import_abf.triggered.connect(
            self.action_import_abf_protocol)
        self._menu_convert.addAction(self._tool_import_abf)
        # Convert > ----
        self._menu_convert.addSeparator()
        # Convert > Export CellML
        self._tool_export_cellml = QtWidgets.QAction('Export model to CellML',
            self)
        self._tool_export_cellml.setStatusTip('Export a model definition to a'
            ' CellML document')
        self._tool_export_cellml.triggered.connect(lambda:
            self.action_export_model('cellml', FILTER_CELLML))
        self._menu_convert.addAction(self._tool_export_cellml)
        # Convert > Export HTML
        self._tool_export_html = QtWidgets.QAction('Export model to HTML',
            self)
        self._tool_export_html.setStatusTip('Export a model definition to an'
            ' HTML document using presentation MathML.')
        self._tool_export_html.triggered.connect(lambda:
            self.action_export_model('html', FILTER_HTML))
        self._menu_convert.addAction(self._tool_export_html)
        # Convert > Export Latex
        self._tool_export_latex = QtWidgets.QAction('Export model to Latex',
            self)
        self._tool_export_latex.setStatusTip('Export a model definition to a'
            ' Latex document.')
        self._tool_export_latex.triggered.connect(lambda:
            self.action_export_model('latex-article', FILTER_LATEX))
        self._menu_convert.addAction(self._tool_export_latex)
        # Convert > ----
        self._menu_convert.addSeparator()
        # Convert > Ansic
        self._tool_export_ansic = QtWidgets.QAction('Export to Ansi C', self)
        self._tool_export_ansic.setStatusTip('Export to a runnable Ansi C'
            ' program.')
        self._tool_export_ansic.triggered.connect(lambda:
            self.action_export_runnable('ansic'))
        self._menu_convert.addAction(self._tool_export_ansic)
        # Convert > Matlab
        self._tool_export_matlab = QtWidgets.QAction('Export to Matlab/Octave',
            self)
        self._tool_export_matlab.setStatusTip('Export to a runnable'
            ' Matlab/Octave script.')
        self._tool_export_matlab.triggered.connect(lambda:
            self.action_export_runnable('matlab'))
        self._menu_convert.addAction(self._tool_export_matlab)
        # Convert > OpenCL
        self._tool_export_opencl = QtWidgets.QAction('Export to OpenCL', self)
        self._tool_export_opencl.setStatusTip('Export a model definition to an'
            ' OpenCL kernel program.')
        self._tool_export_opencl.triggered.connect(lambda:
            self.action_export_runnable('opencl'))
        self._menu_convert.addAction(self._tool_export_opencl)
        # Convert > Python
        self._tool_export_python = QtWidgets.QAction('Export to Python', self)
        self._tool_export_python.setStatusTip('Export a model definition to a'
            ' runnable Python script.')
        self._tool_export_python.triggered.connect(lambda:
            self.action_export_runnable('python'))
        self._menu_convert.addAction(self._tool_export_python)
        #
        # Analysis menu
        #
        self._menu_analysis = self._menu.addMenu('&Analysis')
        # Analysis > Model statistics
        self._tool_stats = QtWidgets.QAction('Show model statistics', self)
        self._tool_stats.setStatusTip('Displays some basic statistics about'
            ' the current model.')
        self._tool_stats.triggered.connect(self.action_model_stats)
        self._menu_analysis.addAction(self._tool_stats)
        # Analysis > ----
        self._menu_analysis.addSeparator()
        # Analysis > Check units strict
        self._tool_units_strict = QtWidgets.QAction('Check units (&strict)',
            self)
        self._tool_units_strict.setStatusTip('Check this model\'s units in'
            ' strict mode.')
        self._tool_units_strict.triggered.connect(
            self.action_check_units_strict)
        self._menu_analysis.addAction(self._tool_units_strict)
        # Analysis > Check units tolerant
        self._tool_units_tolerant = QtWidgets.QAction(
            'Check units (&tolerant)', self)
        self._tool_units_tolerant.setStatusTip('Check this model\'s units in'
            ' tolerant mode.')
        self._tool_units_tolerant.triggered.connect(
            self.action_check_units_tolerant)
        self._menu_analysis.addAction(self._tool_units_tolerant)
        # Analysis > ----
        self._menu_analysis.addSeparator()
        # Analysis > Show variable info
        self._tool_variable_info = QtWidgets.QAction(
            'Show quick variable info', self)
        self._tool_variable_info.setShortcut('Ctrl+R')
        self._tool_variable_info.setStatusTip('Shows this variable\'s type and'
            ' where it is defined.')
        self._tool_variable_info.triggered.connect(self.action_variable_info)
        self._menu_analysis.addAction(self._tool_variable_info)
        # Analysis > Show variable evaluation
        self._tool_variable_evaluation = QtWidgets.QAction('Show variable'
            ' evaluation', self)
        self._tool_variable_evaluation.setShortcut('Ctrl+E')
        self._tool_variable_evaluation.setStatusTip('Show how the selected'
            ' variable is evaluated.')
        self._tool_variable_evaluation.triggered.connect(
            self.action_variable_evaluation)
        self._menu_analysis.addAction(self._tool_variable_evaluation)
        # Analysis > Show variable dependencies
        self._tool_variable_dependencies = QtWidgets.QAction('Show variable'
            ' dependencies', self)
        self._tool_variable_dependencies.setShortcut('Ctrl+D')
        self._tool_variable_dependencies.setStatusTip('Show all expressions'
            ' needed to calculate the selected variable.')
        self._tool_variable_dependencies.triggered.connect(
            self.action_variable_dependencies)
        self._menu_analysis.addAction(self._tool_variable_dependencies)
        # Analysis > Show variable users
        self._tool_variable_users = QtWidgets.QAction('Show variable users',
            self)
        self._tool_variable_users.setShortcut('Ctrl+U')
        self._tool_variable_users.setStatusTip('Show all expressions dependent'
            ' on the selected variable.')
        self._tool_variable_users.triggered.connect(
            self.action_variable_users)
        self._menu_analysis.addAction(self._tool_variable_users)
        # Analysis > Graph variable
        self._tool_variable_graph = QtWidgets.QAction(
            'Graph selected variable', self)
        self._tool_variable_graph.setShortcut('Ctrl+G')
        self._tool_variable_graph.setStatusTip('Display a graph of the'
            ' selected variable.')
        self._tool_variable_graph.triggered.connect(self.action_variable_graph)
        self._menu_analysis.addAction(self._tool_variable_graph)
        # Analysis > ----
        self._menu_analysis.addSeparator()
        # Analysis > Evaluate state derivatives
        self._tool_state_derivatives = QtWidgets.QAction('Evaluate state'
            ' derivatives', self)
        self._tool_state_derivatives.setShortcut('F7')
        self._tool_state_derivatives.setStatusTip('Evaluate all state'
            ' derivatives and display the results.')
        self._tool_state_derivatives.triggered.connect(
            self.action_state_derivatives)
        self._menu_analysis.addAction(self._tool_state_derivatives)
        # Analysis > Evaluate state derivatives without error checking
        self._tool_state_derivatives2 = QtWidgets.QAction('Evaluate state'
            ' derivatives (no error checking)', self)
        self._tool_state_derivatives2.setShortcut('F8')
        self._tool_state_derivatives2.setStatusTip('Evaluate all state'
            ' derivatives without checking for numerical errors.')
        self._tool_state_derivatives2.triggered.connect(
            self.action_state_derivatives2)
        self._menu_analysis.addAction(self._tool_state_derivatives2)
        # Analysis > ----
        self._menu_analysis.addSeparator()
        # Analysis > Show component dependency graph
        self._tool_component_dependency_graph = QtWidgets.QAction(
            'Show component  dependency graph', self)
        self._tool_component_dependency_graph.setStatusTip('Display a graph of'
            ' the dependencies between components.')
        self._tool_component_dependency_graph.triggered.connect(
            self.action_component_dependency_graph)
        self._menu_analysis.addAction(self._tool_component_dependency_graph)
        # Analysis > Show variable dependency graph
        self._tool_variable_dependency_graph = QtWidgets.QAction(
            'Show variable dependency graph', self)
        self._tool_variable_dependency_graph.setStatusTip('Display a graph of'
            ' the dependencies between variables.')
        self._tool_variable_dependency_graph.triggered.connect(
            self.action_variable_dependency_graph)
        self._menu_analysis.addAction(self._tool_variable_dependency_graph)
        # Analysis > Show state dependency matrix
        self._tool_state_matrix = QtWidgets.QAction(
            'Show state dependency matrix', self)
        self._tool_state_matrix.setStatusTip('Display a matrix graph of'
            ' the dependencies between states.')
        self._tool_state_matrix.triggered.connect(self.action_state_matrix)
        self._menu_analysis.addAction(self._tool_state_matrix)
        # Analysis > Show component dependency cycles
        self._tool_component_cycles = QtWidgets.QAction(
            'Show cyclical component dependencies', self)
        self._tool_component_cycles.setStatusTip('Display a list of cyclical'
            ' dependencies between components.')
        self._tool_component_cycles.triggered.connect(
            self.action_component_cycles)
        self._menu_analysis.addAction(self._tool_component_cycles)
        #
        # Run menu
        #
        self._menu_run = self._menu.addMenu('&Run')
        # Run > Validate
        self._tool_validate = QtWidgets.QAction('&Validate model and protocol',
            self)
        self._tool_validate.setShortcut('Ctrl+B')
        self._tool_validate.setStatusTip('Validate the model and protocol')
        self._tool_validate.triggered.connect(self.action_validate)
        self._menu_run.addAction(self._tool_validate)
        # Run > Jump to error
        self._tool_jump_to_error = QtWidgets.QAction('&Jump to last error',
            self)
        self._tool_jump_to_error.setShortcut('Ctrl+Space')
        self._tool_jump_to_error.setStatusTip('Jump to the last model error'
            ' found.')
        self._tool_jump_to_error.triggered.connect(self.action_jump_to_error)
        self._menu_run.addAction(self._tool_jump_to_error)
        # Run > Run embedded script
        self._tool_run = QtWidgets.QAction('&Run embedded script', self)
        self._tool_run.setShortcut('F5')
        self._tool_run.setStatusTip('Run the embedded script.')
        self._tool_run.setIcon(myokit.gui.icon('media-playback-start'))
        self._tool_run.triggered.connect(self.action_run)
        self._menu_run.addAction(self._tool_run)
        # Run > Run explorer
        self._tool_explore = QtWidgets.QAction('&Run explorer', self)
        self._tool_explore.setShortcut('F6')
        self._tool_explore.setStatusTip('Run a simulation and display the'
            ' results in the explorer.')
        self._tool_explore.setIcon(myokit.gui.icon('media-playback-start'))
        self._tool_explore.triggered.connect(self.action_explore)
        self._menu_run.addAction(self._tool_explore)
        #
        # Help menu
        #
        self._menu_help = self._menu.addMenu('&Help')
        # Help > About
        self._tool_about = QtWidgets.QAction('&About', self)
        self._tool_about.setStatusTip('View information about this program.')
        self._tool_about.triggered.connect(self.action_about)
        self._menu_help.addAction(self._tool_about)
        # Help > License
        self._tool_license = QtWidgets.QAction('&License', self)
        self._tool_license.setStatusTip('View this program\'s license info.')
        self._tool_license.triggered.connect(self.action_license)
        self._menu_help.addAction(self._tool_license)
    def create_toolbar(self):
        """
        Creates the shared toolbar.
        """
        self._toolbar = self.addToolBar('tools')
        self._toolbar.setFloatable(False)
        self._toolbar.setMovable(False)
        self._toolbar.setToolButtonStyle(myokit.gui.TOOL_BUTTON_STYLE)
        self._toolbar.addAction(self._tool_new)
        self._toolbar.addAction(self._tool_open)
        self._toolbar.addAction(self._tool_save)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._tool_undo)
        self._toolbar.addAction(self._tool_redo)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._tool_find)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._tool_explore)
        self._toolbar.addAction(self._tool_run)
    def load_config(self):
        """
        Loads the user configuration from an ini file.
        """
        # Read ini file
        inifile = os.path.expanduser(SETTINGS_FILE)
        if not os.path.isfile(inifile):
            return
        config = ConfigParser.ConfigParser()
        config.read(inifile)
        def getor(section, name, alt):
            """ Get or use alternative """
            kind = type(alt)
            if config.has_option(section, name):
                return kind(config.get(section, name))
            return alt
        # Window dimensions and location
        if config.has_section('window'):
            g = self.geometry()
            x = getor('window', 'x', g.x())
            y = getor('window', 'y', g.y())
            w = getor('window', 'w', g.width())
            h = getor('window', 'h', g.height())
            self.setGeometry(x, y, w, h)
        # Splitter sizes
        if config.has_section('splitter'):
            if config.has_option('splitter', 'top') and config.has_option(
                    'splitter', 'bottom'):
                a = int(config.get('splitter', 'top'))
                b = int(config.get('splitter', 'bottom'))
                self._central_splitter.setSizes([a, b])
        # Current and recent files
        if config.has_section('files'):
            if config.has_option('files', 'file'):
                filename = config.get('files', 'file')
                if os.path.isfile(filename):
                    self._file = filename
            self._recent_files = []
            for i in range(0, N_RECENT_FILES):
                opt = 'recent_' + str(i)
                if config.has_option('files', opt):
                    filename = config.get('files', opt)
                    if os.path.isfile(filename):
                        self._recent_files.append(filename)
            self.update_recent_files_menu()
        # Source editors
        self._model_editor.load_config(config, 'model_editor')
        self._protocol_editor.load_config(config, 'protocol_editor')
        self._script_editor.load_config(config, 'script_editor')
        # Model navigator
        nav = getor('model_navigator', 'visible', 'false').strip().lower()
        if nav == 'true':
            self._tool_view_navigator.setChecked(True)
            self.action_show_or_hide_navigator()
    def load_file(self, filename):
        """
        Loads a file into the IDE. Does not provide error handling.
        """
        # Close explorer, if required
        self.close_explorer()
        # Allow user directory and relative paths
        filename = os.path.abspath(os.path.expanduser(filename))
        # Set path to filename's path. Do this before we even know the file is
        # valid: if you click the wrong file by mistake you shouldn't have to
        # browse all the way back again).
        self._path = os.path.dirname(filename)
        # Open file, split into segments
        with open(filename, 'r') as f:
            segments = myokit.split(f)
        # Still here? Then set as file.
        self._file = filename
        # Add to recent files
        self.add_recent_file(filename)
        # Update model editor
        self._model_editor.set_text(segments[0].strip())
        # Update protocol editor
        self._protocol_editor.set_text(segments[1].strip())
        # Update script editor
        self._script_editor.set_text(segments[2].strip())
        # Don't validate model or protocol. Opening an invalid file is not an
        # error in itself.
        # Update console
        self._console.write('Opened ' + self._file)
        # Set working directory to file's path
        os.chdir(self._path)
        # Save settings file
        try:
            self.save_config()
        except Exception:
            pass
        # For some reason, setPlainText('') triggers a change event claiming
        # the text has changed. As a result, files with empty sections will
        # always show up as changed. This is prevented manually below:
        # (Triggers will handle the rest)
        self._model_editor.document().setModified(False)
        self._protocol_editor.document().setModified(False)
        self._script_editor.document().setModified(False)
        # Update interface
        self.update_navigator() # Avoid delay
        self.update_window_title()
    def new_file(self):
        """
        Replaces the editor contents with a new file. Does not do any error
        handling.
        """
        self._file = None
        # Close explorer, if required
        self.close_explorer()
        # Update editors
        self._model_editor.setPlainText('[[model]]\n')
        self._protocol_editor.setPlainText(myokit.default_protocol().code())
        self._script_editor.setPlainText(myokit.default_script())
        # Update interface
        self._tool_save.setEnabled(True)
        self.update_navigator() # Avoid delay
        self.update_window_title()
    def model(self, force=False, console=False, errors_in_console=False):
        """
        Validates and returns the model.
        
        If no model is specified in the model field ``None`` is returned. If
        parse errors occur, the value ``False`` is returned.

        The argument ``force`` can be used to force a reparsing, even if no
        changes were made to the text.

        If ``console`` is set to ``True`` the results of parsing will be
        written to the console. Similarly, the option ``errors_in_console``
        allows errors - but no positive parse results - to be shown in the
        console.
        """
        # Check for cached valid model
        if self._valid_model is not None and not force:
            if console:
                self._console.write('No changes to model since last build.')
            return self._valid_model
        # Parse and validate
        model = None
        # Reset last model error
        self._last_model_error = None
        # Check for empty model field
        lines = self._model_editor.get_text()
        if lines.strip() == '':
            if console:
                self._console.write('No model found.')
            return None
        # Validate and return
        lines = lines.splitlines()
        try:
            # Parse
            model = myokit.parse_model(lines)
            # Show output
            if console:
                self._console.write('No errors found in model definition.')
            if model.has_warnings():
                if console or errors_in_console:
                    self._console.write(model.format_warnings())
            # Cache validated model
            self._valid_model = model
            return model
        except myokit.ParseError as e:
            if console or errors_in_console:
                # Write error to console
                self._console.write(myokit.format_parse_error(e, lines))
                # Store error
                self._last_model_error = e
            return False
        except myokit.IntegrityError as e:
            if console or errors_in_console:
                self.statusBar().showMessage('Model integrity error')
                self._console.write('Model integrity error:')
                self._console.write(e.message)
            return False
    def navigator_item_changed(self, item, previous_item):
        """
        Called whenever the navigator item is changed.
        """
        if self._editor_tabs.currentWidget() != self._model_editor:
            return None
        if item is None:
            return
        # Jump to selected component
        pos = item.data(Qt.UserRole)
        self._model_editor.set_cursor(pos)
    def prompt_save_changes(self, cancel=False):
        """
        Asks the user to save changes and does so if required.

        Returns ``True`` if the action can continue, ``False`` if the action
        should halt. A "Cancel" option will be provided if ``cancel=True``.
        """
        if not self._have_changes:
            return True
        if self._file:
            msg = 'Save changes to ' + str(self._file) + '?'
        else:
            msg = 'Save changes to new file?'
        box = QtWidgets.QMessageBox
        options = box.Yes | box.No
        if cancel:
            options |= box.Cancel
        reply = box.question(self, TITLE, msg, options)
        if reply == box.Yes:
            # Only allow quitting if save succesful
            return self.save_file(save_as=False)
        elif reply == box.No:
            return True
        else:
            return False
    def protocol(self, force=False, console=False, errors_in_console=False):
        """
        Validates the entered pacing protocol and returns it.
        
        If no protocol is specified ``None`` will be returned. If the specified
        protocol has errors, the return value will be ``False``.
        
        If ``force`` is set to ``True`` the protocol will always be reparsed,
        even if no changes were made.
        
        If ``console`` is set to ``True`` the results of parsing will be
        written to the console. Similarly, the option ``errors_in_console``
        allows errors - but no positive parse results - to be shown in the
        console.
        """
        # Check for cached valid protocol
        if self._valid_protocol and not force:
            if console:
                self._console.write('No changes to protocol since last build.')
            return self._valid_protocol
        # Parse and validate
        protocol = None
        # Check for empty protocol field
        lines = self._protocol_editor.get_text()
        if lines.strip() == '':
            if console:
                self._console.write('No protocol found.')
            return None
        # Validate and return
        lines = lines.splitlines()
        try:
            # Parse
            protocol = myokit.parse_protocol(lines)
            # Show output
            if console:
                self._console.write('No errors found in protocol.')
            # Cache valid protocol
            self._valid_protocol = protocol
            return protocol
        except myokit.ParseError as e:
            if console or errors_in_console:
                self._console.write(myokit.format_parse_error(e, lines))
            return False
    def save_config(self):
        """
        Saves the user configuration to an ini file.
        """
        config = ConfigParser.ConfigParser()
        # Window dimensions and location
        config.add_section('window')
        g = self.geometry()
        config.set('window', 'x', str(g.x()))
        config.set('window', 'y', str(g.y()))
        config.set('window', 'w', str(g.width()))
        config.set('window', 'h', str(g.height()))
        # Central splitter
        config.add_section('splitter')
        a, b = self._central_splitter.sizes()
        config.set('splitter', 'top', str(a))
        config.set('splitter', 'bottom', str(b))
        # Current and recent files
        config.add_section('files')
        config.set('files', 'file', self._file)
        for k, filename in enumerate(self._recent_files):
            config.set('files', 'recent_' + str(k), filename)
        # Source editors
        self._model_editor.save_config(config, 'model_editor')
        self._protocol_editor.save_config(config, 'protocol_editor')
        self._script_editor.save_config(config, 'script_editor')
        # Model navigator visibility
        config.add_section('model_navigator')
        config.set('model_navigator', 'visible',
            str(self._tool_view_navigator.isChecked()))
        # Write configuration to ini file
        inifile = os.path.expanduser(SETTINGS_FILE)
        with open(inifile, 'wb') as configfile:
            config.write(configfile)
    def save_file(self, save_as=False):
        """
        Saves the current document. If no file name is known or
        ``save_as=True`` the user is asked for a filename.
        
        Returns ``True`` if the save was succesful
        """
        # Get file name
        if save_as or self._file is None:
            path = self._file
            if path is None:
                path = os.path.join(self._path, 'new-model.mmt')
            filename = QtWidgets.QFileDialog.getSaveFileName(self,
                'Save mmt file', path, filter=FILTER_MMT_SAVE)[0]
            if not filename:
                return
            # Set file
            self._file = str(filename)
            # Add to recent files
            self.add_recent_file(self._file)
            # Set path
            self._path = os.path.dirname(self._file)
            # Set working directory to new path
            os.chdir(self._path)
        # Save
        self.statusBar().showMessage('Saving to ' + str(self._file))
        self._tool_save.setEnabled(False)
        self._tool_save_as.setEnabled(False)
        try:
            # Make _sure_ the text is retrieved _before_ attempting to write it
            # to a file. Otherwise, if anything goes wrong in this step, the
            # file is already emptied.
            m = self._model_editor.get_text()
            p = self._protocol_editor.get_text()
            x = self._script_editor.get_text()
            myokit.save(self._file, m, p, x)
            # Update have_changes state (triggers signals that fix the rest)
            self._model_editor.document().setModified(False)
            self._protocol_editor.document().setModified(False)
            self._script_editor.document().setModified(False)
            # Update window title
            self.update_window_title()
            # Inform user
            self.statusBar().showMessage('File saved as ' + str(self._file))
        except IOError:
            self._tool_save.setEnabled(True)
            self._tool_save_as.setEnabled(True)
            self.statusBar().showMessage('Error saving file.')
            self.show_exception()
            return False
        except Exception:
            self._tool_save.setEnabled(True)
            self._tool_save_as.setEnabled(True)
            self.statusBar().showMessage('Unexpected error saving file.')
            self.show_exception()
            return False
        finally:
            self._tool_save_as.setEnabled(True)
        # Save file history
        try:
            self.save_config()
        except Exception:
            pass
        # Finished
        return True
    def selected_variable(self, model=None):
        """
        Returns the variable currently pointed at by the cursor in the model
        editor. If a selection is made only the left side is used.
        
        If no variable is found ``None`` is returned. If a model error occurs
        ``False`` is returned
        """
        if self._editor_tabs.currentWidget() != self._model_editor:
            return None
        # Get model
        m = self.model(errors_in_console=True)
        if m is None:
            self._console.write('No model found.')
            return False
        elif m == False:
            self._console.write('Errors found in model. Please fix any'
                ' remaining issues before using this function.')
            return False
        # Get variable
        line, char = self._model_editor.cursor_position()
        token = m.variable_at_text_position(line + 1, char)
        if token is None:
            return None
        if isinstance(token[1], myokit.Variable):
            return token[1]
        elif isinstance(token[1], myokit.Name):
            var = token[1].var()
            if isinstance(var, myokit.ModelPart):
                return var
        return None
    def show_exception(self):
        """
        Displays the last exception.
        """
        QtWidgets.QMessageBox.warning(self, TITLE,
            '<h1>An error has occurred.</h1>'
            '<pre>' + traceback.format_exc() + '</pre>')
    def update_navigator(self):
        """
        Updates the model navigator.
        """
        # Find all components and store their positions in a list
        #Sloppy version: query = QtCore.QRegExp(r'^\[') could be faster?
        query = QtCore.QRegExp(r'^\[[a-zA-Z]{1}[a-zA-Z0-9_]*\]')
        pos = 0
        found = self._model_editor.document().find(query, pos)
        new_list = []
        while not found.isNull():
            pos = found.position()
            block = found.block()
            new_list.append((block.text(), pos))
            found = self._model_editor.document().find(query, pos)
        # Update the navigator if required
        if (new_list == self._navigator_list):
            return
        self._navigator_list = new_list
        self._navigator_items.clear()
        self._navigator_items.setSortingEnabled(True)
        for text, pos in self._navigator_list:
            item = QtWidgets.QListWidgetItem(text[1:-1])
            item.setData(Qt.UserRole, pos)
            self._navigator_items.addItem(item)
    def update_recent_files_menu(self):
        """
        Updates the recent files menu.
        """
        for k, filename in enumerate(self._recent_files):
            t = self._recent_file_tools[k]
            t.setText(str(k + 1) + '. ' + os.path.basename(filename))
            t.setStatusTip('Open ' + os.path.abspath(filename))            
            t.setData(filename)
            t.setVisible(True)
        for i in xrange(len(self._recent_files), N_RECENT_FILES):
            self._recent_file_tools[i].setVisible(False)
    def update_window_title(self):
        """
        Sets this window's title based on the current state.
        """
        title = TITLE + ' ' + myokit.VERSION
        if self._file:
            title = os.path.basename(self._file) + ' - ' + title
            if self._have_changes:
                title = '*' + title
        self.setWindowTitle(title)
class Console(QtWidgets.QPlainTextEdit):
    """
    *Extends*: ``QtWidgets.QPlainTextEdit``
    
    Console window used to write plain text output to in the IDE. Shows model
    parsing states and output of running explorations / scripts.
    """
    def __init__(self, parent=None):
        super(Console, self).__init__(parent)
        self.setReadOnly(True)
        font = myokit.gui.qtMonospaceFont()
        font.setPointSize(10)
        self.setFont(font)
        self.setFrameStyle(QtWidgets.QFrame.WinPanel | QtWidgets.QFrame.Sunken)
    def clear(self):
        """
        Clears the console.
        """
        self.setPlainText('')
    def flush(self):
        """
        Ensures output if written to the screen.
        """
        #QtWidgets.QApplication.processEvents()
        # Calling processEvents() here creates issues when multiprocessing.
        # It appears multiple processes try to flush the stdout stream:
        #import os, sys
        #sys.__stdout__.write(str(os.getpid()) + '\n')
        # This could be caught by checking the process id before doing anything
        # but it's better to make sure child processes simply never call this
        # method!
        # Leaving this out now because there is an auto-flush anyway.
    def write(self, text=None):
        """
        Writes text to the console, prefixes a timestamp
        """
        # Note that this will crash if other threads/processes try to write
        # to it.
        # Ignore newlines sent by print statements (bit hacky...)
        if text == '\n':
            return        
        text = text.rstrip()
        #Note: Ignoring empty strings means a script can't do print('') or
        # print('\n') to clear space!
        #if not text:
        #    return
        # Write text
        self.appendPlainText('[' + myokit.time() + '] ' + str(text))
        # Autoscroll
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        # Autoflush
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ExcludeUserInputEvents)
