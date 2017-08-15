#
# Graph Data Extractor (gde)
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
# -----------------------------------------------------------------------------
#
# GDE Document structure.
#
# All data is stored in a Document object which is a hierarchy of DocumentNode
# objects. Each document node has a name unique within its parent. DocumentNode
# objects can contain data in the form of DocumentVariable objects. Each
# DocumentVariable has a name that is unique within its parent, although node
# and variable names may overlap.
#
# The methods to edit and append documents use DocumentActions and can send out
# signals to allow undo/redo and listeners. In the cases where this is not
# desired, the silent_ methods can be used.
#
# Future stuff
from __future__ import print_function
from __future__ import division
# Standard library imports
import sys
import math
import signal
import os.path
import traceback
import numpy as np
import collections
import xml.dom.minidom as minidom
import xml.etree.cElementTree as et
import ConfigParser as configparser
# Myokit
import myokit.gui
# Qt imports
from myokit.gui import Qt, QtCore, QtGui, QtWidgets
# Application title
TITLE = 'Myokit GDE'
# Application icon
def icon():
    icons = [
        'icon-gde.ico',
        'icon-gde-16.xpm',
        'icon-gde-24.xpm',
        'icon-gde-32.xpm',
        'icon-gde-48.xpm',
        'icon-gde-64.xpm',
        'icon-gde-96.xpm',
        'icon-gde-128.xpm',
        'icon-gde-256.xpm',
        ]
    icon = QtGui.QIcon()
    for i in icons:
        icon.addFile(os.path.join(myokit.DIR_DATA, 'gui', i))
    return icon
# Latest supported Gde document version
DOCUMENT_VERSION = 2
# Tags / Node-types (ntypes)
T_IMAGE = 'image'
T_AXES = 'axes'
T_AXIS = 'axis'
T_AXIS_REFERENCE_POINT = 'refpoint'
T_DATA_SETS = 'datasets'
T_DATA_SET = 'dataset'
T_DATA_POINT = 'datapoint'
T_VARIABLE = 'variable'
# Variable types
V_STR   = 'str'     # A string
V_INT   = 'int'     # An integer
V_BOOL  = 'bool'    # A boolean
V_FLOAT = 'float'   # A float
V_NORM  = 'norm'    # A float in the range [0,1]
V_PATH  = 'path'    # A path name
# Gui z-indexes
Z_BACKGROUND = 0
Z_AXIS = 1
Z_DATA_SET = 2 
Z_DATA = 3
Z_SELECTED = 9999
# Settings file
SETTINGS_FILE = '~/.gde.ini'
# Number of recent files to display
N_RECENT_FILES = 5
# About
ABOUT = '<h1>' + TITLE + ' ' + myokit.VERSION + '</h1>' + """
<p>
    Myokit Graph Data Extractor (GDE) is an application for extracting raw
    data points from an image file.
</p>
<p>
    A typical project with the gde has the following steps:
</p>
<ol>
    <li>
        Selecting an image. The image to use can be set by selecting "Set
        image" from the "Edit" menu or by double-clicking on the empty canvas.
    </li>
    <li>
        Setting the axes. Each axis has two reference points you can use to
        position the axes. The "value" parameter of the reference point can
        then be used to give that point a numerical value. These values will be
        used to determine the coordinates when extracting the data. When
        positioning the axes, hold down Ctrl to keep the axis horizontal or
        vertical.
    </li>
    <li>
        Adding data points. Data points can be added by selecting
        "Add data point" from the "Edit" menu or by holding Ctrl and clicking
        anywhere on the empty canvas. Once placed, you can use the mouse to
        drag data points to their correct position.
    </li>
    <li>
        A smoothing, penalised B-spline (or P-spline) can be fitted to the data
        points by enabling the "spline" option in a set of data points. By
        default, the spline will have as many segments as data points, this is
        set automatically when "segments=0" but can be overridden by changing
        the value of "segments" to a non-zero value. The spline has degree
        "degree" and is fit using a penalty criterium of degree "penalty".
        A smoothing factor can be applied by setting "smoothing" to any
        non-zero factor. The spline is sampled (for display and data
        extraction) at linearly spaced points, the number of which is set as
        "samples".
    </li>
    <li>
        Extracting data points. By hitting the "Extract" option on the toolbar
        or selecting "Extract data points" from the "Edit" menu, you can export
        the positions of the data points to a plain text file. If a dataset has
        a spline fitted to its points, the extracted values will be sampled
        from the spline.
    </li>
    <li>
        Finally, it is possible to save your work in the "gde" format (which is
        simply a small xml file).
    </li>
</ol>
<p>
    (Currently running on the BACKEND backend.)
</p>
""".replace('BACKEND', myokit.gui.backend)
#
# Document classes
#
class DocumentNode(QtCore.QObject):
    """
    The ``DocumentNode`` class is used to build a tree structure where each
    node can contain one or more ``DataValue`` objects.

    A ``parent`` must be specified: For document roots this should be the
    QObject owning the document, for all other nodes it should be the parent
    node. 

    Each node has an ``ntype``, which describes the type of node it is. This is
    a string property that can be set freely.

    Data stored in the node can be obtained using ``get_value()``
    """
    # Signals
    # Called when this node was deselected in one of the views
    # Attributes: self
    node_deselected= QtCore.Signal(object)
    # Called when this node was selected in one of the views
    # Attributes: self
    node_selected= QtCore.Signal(object)
    # Called when this node was removed
    # Attributes: self
    node_removed = QtCore.Signal(object)
    # Called when a child was added to this node
    # Attributes: self, child
    child_added = QtCore.Signal(object, object)
    # Called when a child was removed from this node
    # Attributes: self, child
    child_removed = QtCore.Signal(object, object)
    # Called when a variable was added to this node
    # Attributes: self, DocumentVariable
    variable_added = QtCore.Signal(object, object)
    # Called when a variable was removed from this nod
    # Attributes: self, DocumentVariable
    variable_removed = QtCore.Signal(object, object)
    # Called when a variable in this node was changed
    # Attributes: self, DocumentVariable
    variable_changed = QtCore.Signal(object, object)
    def __init__(self, parent, ntype, name):
        # Initialize!
        super(DocumentNode, self).__init__(parent)
        self._parent = parent if isinstance(parent, DocumentNode) else None
        self._ntype = ntype
        self._name = name
        # The document this node belongs to
        self._document = None
        # The document model of the document this node belongs to
        self._model = None
        # This node's data
        self._data = collections.OrderedDict()
        # This node's children
        self._kids = collections.OrderedDict()
        # Selection
        self._selected = False
    def add_child(self, ntype, name, variables=None):
        """
        Appends a child to this node.

        Variables can be added by setting ``variables`` to a list of tuples
        ``(type, name, value)``.

        """
        if variables is None:
            variables = ()
        action = DA_AddNode(self, ntype, name, variables)
        return self.get_document()._perform(action)
    def add_variable(self, vtype, name, value=None):
        """
        Adds a DocumentVariable to this node.
        """
        action = DA_AddVariable(self, vtype, name, value)
        return self.get_document()._perform(action)
    def can_drag(self):
        """
        Returns True if this item is draggable / droppable.
        """
        return self.get_document().can_drag(self)
    def child(self, k):
        """
        Returns the ``k-th`` child node.

        **Required to work with DocumentModel.**
        """
        return self._kids[self._kids.keys()[k]]
    def clear_selection(self):
        """
        Clears the selection (if any) of this node and any children nodes.
        """
        self.deselect()
        for kid in self._kids.itervalues():
            kid.clear_selection()
    def deselect(self):
        """
        Deselect this node in any listening views.
        """
        if self._selected:
            self._selected = False
            self.node_deselected.emit(self)
    def get(self, *names):
        """
        Returns the child node with the given name.
        """
        if len(names) == 0:
            return self
        return self._kids[names[0]].get(*names[1:])
    def get_document(self):
        """
        Retuns the document this node is in.
        """
        node = self
        while node != None:
            if isinstance(node, Document):
                return node
            node = node.get_parent_node()
        raise Exception('No Document set in hierarchy for ' + str(self))
    def get_model(self):
        """
        Returns the document model for the document this node is in.
        """
        if not self._model:
            self._model = self.get_document().get_model()
        return self._model
    def get_model_index(self, column=0):
        """
        Returns a QModelIndex object refering to this node.
        """
        return self.get_model().createIndex(self.index(), column, self)
    def get_model_selection(self):
        """
        Returns a QItemSelection for this node's entire row.
        """
        m = self.get_model()
        a = m.createIndex(self.index(), 0, self)
        b = m.createIndex(self.index(), 1, self)
        return QtCore.QItemSelection(a, b)
    def get_name(self):
        """
        Returns this node's name.
        """
        return self._name
    def get_ntype(self):
        """
        Returns this node's ntype.
        """
        return self._ntype
    def get_parent_node(self):
        """
        Returns this node's parent, or ``None`` if this node is a root node.
        """
        return self._parent
    def get_value(self, name):
        """
        Returns the value of this node's variable with the given name.
        """
        return self._data[str(name)].get_value()
    def get_values(self, *names):
        """
        Returns all the requested variable values (see get_value())
        """
        x = [0] * len(names)
        for k, name in enumerate(names):
            x[k] = self._data[str(name)].get_value()
        return tuple(x)
    def get_variable(self, name):
        """
        Returns this node's variable with the given name.
        """
        name = str(name)
        return self._data[name]
    def get_xml(self):
        """
        Returns an ElementTree xml version of this node.
        """
        e = et.Element(self._ntype)
        e.attrib['name'] = self._name
        for d in self._data.itervalues():
            e.append(d.get_xml())
        for k in self._kids.itervalues():
            e.append(k.get_xml())
        return e
    def has_value(self, name):
        """
        Returns True if this node has a variable with the given name.
        """
        return name in self._data
    def index(self):
        """
        Returns the index of this node in its parent's list of kids.

        **Required to work with DocumentModel**
        """
        if self._parent is None:
            return 0
        return self._parent._kids.values().index(self)
    def is_selected(self):
        """
        Returns True if this node is selected.
        """
        return self._selected
    def __iter__(self):
        """
        Returns an iterator over this node's children.
        """
        return self._kids.itervalues()
    def iterdata(self):
        """
        Returns an iterator over the :class:`DocumentValue` objects stored in
        this node.
        """
        return self._data.itervalues()
    def __len__(self):
        """
        Returns the number of children this node has.
        """
        return len(self._kids)
    def __nonzero__(self):
        """
        Used when writing ``if node:``, without overloading this, the
        value of ``__len__`` would be used in these cases.
        """
        return True
    def parent(self):
        """
        Returns this node's parent.

        **Required to work with DocumentModel.**
        """
        return self._parent
    def remove(self):
        """
        Removes this node.
        """
        action = DA_RemoveNode(self)
        return self.get_document()._perform(action)
    #def remove_variable(self, variable):
    def select(self):
        """
        Tells any views listening to this node that it should be selected.
        """
        if not self._selected:
            self._selected = True
            self.get_document().node_selected.emit(self)
            self.node_selected.emit(self)
    def set_value(self, **values):
        """
        Sets one or more variables in this node using the keyword syntax.
        """
        action = DA_ChangeVariables(self, values)
        return self.get_document()._perform(action)
    def silent_add_child(self, ntype, name):
        """
        Creates and appends a child node without using actions or sending out
        signals.

        Returns the new node.
        """
        node = DocumentNode(self, ntype, str(name))
        return self.silent_add_existing_child(node)
    def silent_add_existing_child(self, child):
        """
        Appends a child node without using actions or sending out signals.

        Returns the added node.
        """
        name = child.get_name()
        if name in self._kids:
            raise AttributeError('Duplicate child node: "' + name + '".')
        # Notify the document model
        m = self.get_model()
        n = len(self._kids)
        m.beginInsertRows(self.get_model_index(), n, n)
        # Add child
        self._kids[name] = child
        m.endInsertRows()
        # Return added child node
        return child
    def silent_add_existing_variable(self, var):
        """
        Appens a variable without using actions or sending out signals.

        Returns the added variable.
        """
        name = var.get_name()
        if name in self._data:
            raise AttributeError('Duplicate variable name: "' + name + '".')
        # Notify the document model
        m = self.get_model()
        n = len(self._data) + 1
        m.beginInsertRows(self.get_model_index(), n, n)
        # Add
        self._data[name] = var
        # Notify document model
        m.endInsertRows()
        # Return added variable
        return var
    def silent_add_variable(self, vtype, name, value=None):
        """
        Creates and appends a variable to this node without using actions or
        sending out signals.

        Returns the new variable.
        """
        var = DocumentVariable(self, vtype, str(name), value)
        return self.silent_add_existing_variable(var)
    def silent_remove_child(self, node):
        """
        Removes a child without using actions or sending out signals.
        """
        if node.get_parent_node() != self:
            raise AttributeError('Node is not a child of this node.')
        node.deselect()
        n = node.index()
        m = self.get_model()
        m.beginRemoveRows(self.get_model_index(), n, n+1)
        del(self._kids[node.get_name()])
        m.endRemoveRows()
    def silent_remove_variable(self, variable):
        """
        Removes a variable without using actions or sending out signals.
        """
        del(self._data[variable.get_name()])
class Document(DocumentNode):
    """
    Represents an xml like document.
    """
    # Signals
    # Emitted when an exception occurs during an action
    # Attributes: Document, DocumentAction, Exception
    action_exception = QtCore.Signal(object, object, Exception)
    # Emitted when the undo/redo history changes
    # Attributes: Document
    undo_redo_change = QtCore.Signal(object)
    # Called when a node was selected in one of the views
    # Attributes: node or none
    node_selected = QtCore.Signal(object)
    # Called when a node is added somewhere in the document
    # Attributes: parent, child
    doc_node_added = QtCore.Signal(object, object)
    # Called when a node is removed from the document
    # Attributes: parent, child
    doc_node_removed = QtCore.Signal(object, object)
    # Called when a document is deleted
    # Attributes: document
    doc_deleted = QtCore.Signal(object)
    def __init__(self, parent, filename=None):
        super(Document, self).__init__(parent, 'document', 'document')
        # Model for use with a treeview
        self._model = DocumentModel(self)
        # List of actions (changes) performed on this model.
        self._undo = []
        # List of undone changes
        self._redo = []
        # Read the given file or create a default structure
        self._read_file(filename)
        # No changes!
        self._changed = False
    def can_drag(self, node):
        """
        Returns True if the given DocumentNode is drag/drop enabled.
        """
        return False
    def can_redo(self):
        """
        Returns True if there are actions that can be redone.
        """
        return len(self._redo) > 0
    def can_undo(self):
        """
        Returns True if there are actions that can be undone.
        """
        return len(self._undo) > 0
    def delete(self):
        """
        Deletes this document.
        """
        self.doc_deleted.emit(self)
    def get_model(self):
        """
        Returns a model of this document for use with TreeView components.
        """
        return self._model
    def has_changes(self):
        """
        Returns True if any changes were made to this document (even if they
        were subsequently undone).
        """
        return self._changed
    def _perform(self, action):
        """
        Performs an action on this model.
        """
        try:
            result = action.perform()
            self._changed = True
        except Exception as e:
            # Show error info in console
            print('Exception performing action: ' + str(type(action)))
            print(traceback.format_exc())
            # Emit signal about exception
            self.action_exception.emit(self, action, e)
            # Return None
            return None
        # Add action to undo list, clear redo list
        self._redo = []
        self._undo.append(action)
        self.undo_redo_change.emit(self)
        # Return action result
        return result
    def _read_file(self, filename=None):
        """
        Loads the data in the given file into this document.
        """
        raise NotImplementedError
    def redo(self):
        """
        Redoes the last undone action.
        """
        action = self._redo[-1]
        try:
            result = action.perform()
            self._changed = True
        except Exception as e:
            # Show error info in console
            print('Exception redoing action: ' + str(type(action)))
            print(traceback.format_exc())
            # Emit signal about exception
            self.action_exception.emit(self, action, e)
            # Return None
            return None
        # Add action to undo list, clear redo list
        self._redo.pop()
        self._undo.append(action)
        self.undo_redo_change.emit(self)
        # Return action result
        return result
    def undo(self):
        """
        Undoes the last action.
        """
        action = self._undo[-1]
        try:
            result = action.undo()
            self._changed = True
        except Exception as e:
            # Show error info in console
            print('Exception undoing action: ' + str(type(action)))
            print(traceback.format_exc())
            # Emit signal about exception
            self.action_exception.emit(self, action, e)
            # Return None
            return None
        # Remove action from undo list, add to redo list
        self._undo.pop()
        self._redo.append(action)
        self.undo_redo_change.emit(self)
        # Return action result
        return result
    def write(self, filename):
        """
        Writes this document to the given path.
        """
        e = self.get_xml()
        xml = et.tostring(e, encoding='utf-8')
        xml = minidom.parseString(xml)
        try:
            f = open(filename, 'w')
            f.write(xml.toprettyxml(encoding='utf-8'))
            self._changed = False
        finally:
            if f:
                f.close()
class DocumentVariable(QtCore.QObject):
    """
    Stores a variable in a document node.
    """
    # Signals
    # Called when this variable was removed
    # Attributes: self
    variable_removed = QtCore.Signal(object)
    # Called when this variable was changed
    # Attributes: self
    variable_changed = QtCore.Signal(object)
    def __init__(self, node, vtype, name, value=None):
        super(DocumentVariable, self).__init__(node)
        self._node = node
        self._vtype = vtype
        self._name = name
        if type(value) in (str, unicode):
            self._value = self._value_from_string(value)
        else:
            self._value = value
    def get_name(self):
        """
        Returns this variable's name.
        """
        return self._name
    def get_node(self):
        """
        Returns the node this variable belongs to.
        """
        return self._node
    def get_str_value(self):
        """
        Returns this variable's value as a string.
        """
        return self._value_to_string(self._value)
    def get_value(self):
        """
        Returns this variable's value in its native type.
        """
        return self._value
    def get_vtype(self):
        """
        Returns this variable's data type.
        """
        return self._vtype
    def set_value(self, value):
        """
        Changes this variable's value.
        """
        if type(value) in (str, unicode):
            value = self._value_from_string(value)
        action = DA_ChangeVariable(self, value)
        return self.get_node().get_document()._perform(action)
    def get_xml(self):
        """
        Returns an ElementTree object representing this variable.
        """
        e = et.Element(T_VARIABLE)
        e.attrib['vtype'] = self._vtype
        e.attrib['name'] = self._name
        e.attrib['value'] = self.get_str_value()
        return e
    def silent_set_value(self, value):
        """
        Changes this variable's data type, without using actions or sending
        signals.
        """
        if type(value) in (str, unicode):
            value = self._value_from_string(value)
        self._value = value
    def _value_from_string(self, value):
        """
        Creates a value from a string.
        """
        if self._vtype == V_INT:
            return int(value) if value else 0
        elif self._vtype == V_FLOAT:
            return float(value) if value else 0
        elif self._vtype == V_NORM:
            v = float(value) if value else 0
            if v < 0: v = 0
            if v > 1: v = 1
            return v
        elif self._vtype in (V_STR, V_PATH):
            return str(value)
        elif self._vtype == V_BOOL:
            return (value.lower() == 'true')
        else:
            raise Exception('Unknown variable type: ' + str(self._vtype))
    def _value_to_string(self, value):
        if self._vtype == V_INT:
            return str(value)
        elif self._vtype == V_BOOL:
            return str(value)
        elif self._vtype in (V_FLOAT, V_NORM):
            return str(value)
        elif self._vtype in (V_STR, V_PATH):
            return '' if value is None else value
        else:
            raise Exception('Unknown variable type: ' + str(self._vtype))
class GdeDocument(Document):
    """
    Represents a document used by the GDE.
    
    The argument ``parent`` should be a QObject owning this document.
    """
    def __init__(self, parent, filename=None):
        # Major version number
        self._version = 0
        # Axis reference points
        self._refs = None
        # X-axis and Y-axis
        self._xaxis = None
        self._yaxis = None
        # Data sets
        self._data_node = None
        self._active_data_set = None
        # Coordinate conversion method
        def trivial(x, y): return x, y
        self.norm2real = trivial
        # Call parent constructor
        super(GdeDocument, self).__init__(parent, filename)
        # Start listening to node selection
        self.node_selected.connect(self.handle_node_selected)
    def add_data_fit(self):
        """
        Adds a data fit to this document.
        """
    def add_data_set(self):
        """
        Adds a data set to this document.
        """
        n = len(self._data_node)
        n = 'Data set ' + str(n + 1)
        a = (
                (V_STR, 'label', n),
                (V_BOOL, 'spline', False),
                (V_INT, 'segments', 0),
                (V_INT, 'degree', 3),
                (V_INT, 'penalty', 3),
                (V_FLOAT, 'smoothing', 0.1),
                (V_INT, 'samples', 100),
            )
        return self._data_node.add_child(T_DATA_SET, n, a)
    def can_drag(self, node):
        """
        Returns True if the given node is draggable.
        """
        return node.get_ntype() == T_DATA_POINT
    def extract_data(self, path):
        """
        Extracts the data from every data set in the scene and writes it to a
        series of csv files.
        """
        # Define function to export a single data set
        fmt = '{:< 1.5g}'
        x = self._xaxis.get_value('label')
        y = self._yaxis.get_value('label')
        header = '"' + x + '","' + y + '"\n'
        def write(path, dset):
            # Check if spline was requested
            spline = dset.get_value('spline')
            if spline:
                x1 = []
                y1 = []            
                for point in dset:
                    x, y = self.norm2real(*point.get_values('x', 'y'))
                    x1.append(x)
                    y1.append(y)
                x1 = np.array(x1, copy=False)
                xmin = np.min(x1)
                xmax = np.max(x1)
                # Get spline parameters
                smo = dset.get_value('smoothing')
                pen = dset.get_value('penalty')
                deg = dset.get_value('degree')
                seg = dset.get_value('segments')
                sam = dset.get_value('samples')        
                if seg < 1:
                    # Segments = 0 means add a segment per data point
                    seg = len(x1)
                # Points to evaluate the spline at
                x2 = np.linspace(xmin, xmax, sam)
                # Fit spline
                y2 = pspline(x1, y1, x2, smo, seg, deg, pen)
                x, y = iter(x2), iter(y2)
                # Write data
                with open(path, 'w') as f:
                    f.write(header)
                    for i in xrange(sam):
                        f.write(  fmt.format(x.next()) + ','
                            + fmt.format(y.next()) + '\n')
            else:
                with open(path, 'w') as f:
                    f.write(header)
                    for point in dset:
                        x, y = self.norm2real(*point.get_values('x', 'y'))
                        f.write(fmt.format(x) + ',' + fmt.format(y) + '\n')
        # Export all data sets
        n = len(self._data_node)
        if n == 0:
            raise Exception('No data sets to export')
        elif n == 1:
            # Write single data set
            write(path, self._data_node.child(0))
        else:
            # Write multiple data sets to multiple files
            base, ext = os.path.splitext(path)
            base += '-'
            for dset in self._data_node:
                path = base + dset.get_value('label') + ext
                write(path, dset)
    def get_active_data_set(self):
        """
        Returns the currently active dataset. If the document doesn't contain
        any datasets a new set is created.
        """
        if self._active_data_set is None:
            # Create data set
            data = self.get('Data')
            try:
                self._active_data_set = data.child(-1)
            except IndexError:
                self.add_data_set()
        return self._active_data_set
    def get_xml(self):
        """
        Returns an ElementTree xml version of this document.
        """
        e = super(GdeDocument, self).get_xml()
        e.attrib['major'] = str(self._version)
        return e
    def handle_axis_changed(self):
        """
        Updates the coordinate conversion scheme when the axes are changed.
        """
        if self._refs is None:
            return
        # Find point where axes cross (may or may not be the origin).
        # Get reference nodes
        node_rx0, node_rx1, node_ry0, node_ry1 = self._refs
        # Get vectors defining axes
        rx = Line2D(
            Point2D(*node_rx0.get_values('x', 'y')),
            Point2D(*node_rx1.get_values('x', 'y')))
        ry = Line2D(
            Point2D(*node_ry0.get_values('x', 'y')),
            Point2D(*node_ry1.get_values('x', 'y')))
        # Get values of the head and tail of the reference vectors
        vx0 = node_rx0.get_value('value')
        vx1 = node_rx1.get_value('value')
        vy0 = node_ry0.get_value('value')
        vy1 = node_ry1.get_value('value')
        # Get unit vectors on the displaced (axis)
        # For rx, the tail will be on the line x=0 but the y value still
        # depends on where the axis was displayed in the original graph.
        rx = Line2D(rx.point(-vx0 / (vx1 - vx0)),
                    rx.point((1.0 - vx0) / (vx1 - vx0)))
        ry = Line2D(ry.point(-vy0 / (vy1 - vy0)),
                    ry.point((1.0 - vy0) / (vy1 - vy0)))
        # Now we swap rx and ry around so rx' tail is where ry's tail was and
        # vice versa.
        sx = Line2D(rx)
        rx.move_to(ry.tail)
        ry.move_to(sx.tail)
        # Now rx is a unit vector on the x-axis pointing in the x direction
        # and ry is a unit vector on the y-axis pointing in the y direction
        # The origin is found by taking the intersection of these points
        origin = rx.intersect(ry)
        # Get determinant of of matrix of unit vectors
        rx.move_to(origin)
        ry.move_to(origin)
        rx = rx.head - rx.tail
        ry = ry.head - ry.tail
        idet = 1.0 / (rx.x * ry.y - ry.x * rx.y)
        # Set transformation matrix
        def transform(x, y):
            # Get position relative to origin
            x -= origin.x
            y -= origin.y
            # Transform to real coordinates
            xx = idet * (x * ry.y - y * ry.x)
            yy = idet * (y * rx.x - x * rx.y)
            return xx, yy
        self.norm2real = transform
    def handle_data_set_added(self, parent, dset):
        """
        Called when a data set is added to the data node.
        """
        self._active_data_set = dset
    def handle_data_set_removed(self, parent, dset):
        """
        Called when a data set is removed from the data node.
        """
        if dset == self._active_data_set:
            self._active_data_set = None
    def handle_node_selected(self, node):
        """
        Called when a node is selected in this document.
        """
        ntype = node.get_ntype()
        if ntype == T_DATA_SET:
            self._active_data_set = node
        elif ntype == T_DATA_POINT:
            self._active_data_set = node.get_parent_node()
    def _read_file(self, filename=None):
        """
        Reads a Gde document or creates a default one.
        """
        def find(parent, ntype, name):
            """
            Finds the first child node of the given type with the given name.
            """
            if parent is None:
                return None
            for kid in parent.iterfind(ntype):
                if kid.attrib['name'] == name:
                    return kid
            return None
        try:
            # Parse file
            self.filename = filename
            doc = None
            if filename != None:
                doc = et.parse(filename)
            # Add version
            if doc:
                doctag = doc.getroot()
                try:
                    major_version = int(doctag.attrib['major'])
                except Exception:
                    major_version = 0
            else:
                major_version = DOCUMENT_VERSION
            self._version = major_version
            # Add image
            image = self.add_child(T_IMAGE, 'Image')
            image.add_variable('path', 'path')
            if doc:
                x = find(doc, T_IMAGE, 'Image')
                x = find(x, T_VARIABLE, 'path')
                if x != None:
                    image.set_value(path=x.attrib['value'])
            # Add axes
            d = [[0.3, 0.8, 0],[0.8, 0.8, 5],[0.2, 0.7, 0],[0.2, 0.2, 5]]
            xlabel = 'x'
            ylabel = 'y'
            if doc:
                x = find(doc, T_AXES, 'Axes')
                def pt(axis, name, idx):
                    x = find(axis, T_AXIS_REFERENCE_POINT, name)
                    y = find(x, T_VARIABLE, 'x')
                    if y != None: d[idx][0] = y.attrib['value']
                    y = find(x, T_VARIABLE, 'y')
                    if y != None: d[idx][1] = y.attrib['value']
                    y = find(x, T_VARIABLE, 'value')
                    if y != None: d[idx][2] = y.attrib['value']
                y = find(x, T_AXIS, 'x')
                if y != None:
                    pt(y, 'ref1', 0)
                    pt(y, 'ref2', 1)
                    z = find(y, T_VARIABLE, 'label')
                    if z != None: xlabel = z.attrib['value']
                y = find(x, T_AXIS, 'y')
                if y != None:
                    pt(y, 'ref1', 2)
                    pt(y, 'ref2', 3)
                    z = find(y, T_VARIABLE, 'label')
                    if z != None: ylabel = z.attrib['value']
            def add_reference_point(parent, name, x, y, v):
                r = parent.add_child(T_AXIS_REFERENCE_POINT, name)
                r.add_variable(V_NORM, 'x', x)
                r.add_variable(V_NORM, 'y', y)
                r.add_variable(V_FLOAT, 'value', float(v))
                return r
            axes = self.add_child(T_AXES, 'Axes')
            xaxis = axes.add_child(T_AXIS, 'x')
            xaxis.add_variable(V_STR, 'label', xlabel)
            x1 = add_reference_point(xaxis, 'ref1', *d[0])
            x2 = add_reference_point(xaxis, 'ref2', *d[1])
            yaxis = axes.add_child(T_AXIS, 'y')
            yaxis.add_variable(V_STR, 'label', ylabel)
            y1 = add_reference_point(yaxis, 'ref1', *d[2])
            y2 = add_reference_point(yaxis, 'ref2', *d[3])
            # Store axes and reference points
            self._xaxis = xaxis
            self._yaxis = yaxis
            self._refs = (x1, x2, y1, y2)
            # Add axis-change listeners to reference points
            x1.variable_changed.connect(self.handle_axis_changed)
            x2.variable_changed.connect(self.handle_axis_changed)
            y1.variable_changed.connect(self.handle_axis_changed)
            y2.variable_changed.connect(self.handle_axis_changed)
            # Create initial norm2real function
            self.handle_axis_changed()
            # Add data set node
            self._data_node = self.add_child(T_DATA_SETS, 'Data')
            # Listen for new data sets
            self._data_node.child_added.connect(self.handle_data_set_added)
            self._data_node.child_removed.connect(self.handle_data_set_removed)
            # Add data sets, data points
            if doc:
                if self._version < 1:
                    root = doc
                else:
                    root = find(doc, T_DATA_SETS, 'Data')
                if root:
                    # Data sets
                    for z in root.iterfind(T_DATA_SET):
                        # Name and label
                        if self._version < 1:
                            name = label = 'Data set 1'
                        else:
                            name = z.attrib['name']
                            label = name
                            x = z.find(T_VARIABLE, 'label')
                            if x != None:
                                label = str(x.attrib['value'])
                        # Spline fitting options
                        def var(root, name, default_value):
                            v = find(root, T_VARIABLE, name)
                            if v is None:
                                return default_value
                            return v.attrib['value'] 
                        spline = var(z, 'spline', False)
                        s_seg = var(z, 'segments', 0)
                        s_deg = var(z, 'degree', 3)
                        s_pen = var(z, 'penalty', 3)
                        s_smo = var(z, 'smoothing', 0.1)
                        s_sam = var(z, 'samples', 100)
                        # Add data set
                        dset = self._data_node.add_child(T_DATA_SET, name, (
                            (V_STR, 'label', label),
                            (V_BOOL, 'spline', spline),
                            (V_INT, 'segments', s_seg),
                            (V_INT, 'degree', s_deg),
                            (V_INT, 'penalty', s_pen),
                            (V_FLOAT, 'smoothing', s_smo),
                            (V_INT, 'samples', s_sam),
                            ))
                        self._active_data_set = dset
                        # Add data points
                        for p in z.iterfind(T_DATA_POINT):
                            x = find(p, T_VARIABLE, 'x')
                            y = find(p, T_VARIABLE, 'y')
                            if x != None and y != None:
                                x = x.attrib['value']
                                y = y.attrib['value']
                                n = str(p.attrib['name'])
                                v = ((V_NORM, 'x', x),(V_NORM, 'y', y))
                                dset.add_child(T_DATA_POINT, n, v)
        finally:
            # Clear undo/redo
            self._undo = []
            self._redo = []
            self._changed = False
            self.undo_redo_change.emit(self)
            # Set version to latest and save as such :)
            self._version = DOCUMENT_VERSION
    def set_active_data_set(self, data_set):
        """
        Sets the currently active data set.
        """
        self._active_data_set = data_set
#
# Tiny 2D Vector and line classes
#
class Point2D(object):
    """
    Represents a point in R2.
    """
    def __init__(self, x, y=None):
        if isinstance(x, Point2D):
            self.x = x.x
            self.y = x.y
        elif y != None:
            self.x = float(x)
            self.y = float(y)
        else:
            raise AttributeError('Point2D can only be created with'
                ' Point2D(float, float) or Point2D(Point2D).')
    def __repr__(self):
        return 'Point2D(' + str(self.x) + ', ' + str(self.y) + ')'
    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'
    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self
    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)
    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self
    def __mul__(self, other):
        other = float(other)
        return Point2D(self.x * other, self.y * other)
    def __rmul__(self, other):
        other = float(other)
        return Point2D(self.x * other, self.y * other)
    def __imul__(self, other):
        other = float(other)
        self.x *= other
        self.y *= other
        return self
    def __neg__(self):
        return Point2D(-self.x, -self.y)
    def __pos__(self):
        return Point2D(self)
    def __len__(self):
        return 2
    def __getitem__(self, key):
        return (self.x, self.y)[key]
    def __setitem__(self, key, value):
        p = [self.x, self.y]
        p[key] = float(value)
        self.x, self.y = p
    def __eq__(self, other):
        if isinstance(other, Point2D):
            return self.x == other.x and self.y == other.y
        return False
    def __ne__(self, other):
        if isinstance(other, Point2D):
            return self.x != other.x or self.y != other.y
        return True
    def transform(self, a, b, c, d):
        """
        Transforms this point by multiplying with the matrix ((a, b),(c, d)).
        """
        x, y = self.x, self.y
        self.x = a * x + b * y
        self.y = c * x + d * y
class Line2D(object):
    """
    Represents a line segment in 2D euclidean space. A line is defined using
    two points, representing the 'head' and 'tail' of a vector connecting the
    points.

    A Line2D can be created in three ways:

        1. Using ``x = Line2D(y)``, where ``y`` is another instance of Line2D.
        2. Using ``x = Line2D(tail, head)``, where both ``tail`` and ``head``
           are instances of ``Point2D``.
        3. Using ``x = Line2D(tail_x, tail_y, head_x, head_y)`` where all
           arguments are of types that can be converted to ``float``.

    """
    def __init__(self, a, b=None, c=None, d=None):
        if isinstance(a, Line2D):
            self.tail = Point2D(a.tail)
            self.head = Point2D(a.head)
        elif isinstance(a, Point2D) and isinstance(b, Point2D):
            self.tail = Point2D(a)
            self.head = Point2D(b)
        elif b != None and c != None and d != None:
            self.tail = Point2D(a, b)
            self.head = Point2D(c, d)
        else:
            raise AttributeError('Invalid signature. Expecting Line2D(Line2D)'
                ' or Line2D(Point2D, Point2D) or Line2D(float, float, float,'
                ' float).')
    def __repr__(self):
        return 'Line2D(' + str(self.tail) + ', ' + str(self.head) + ')'
    def __str__(self):
        return '(' + str(self.tail) + ', ' + str(self.head) + ')'
    def __add__(self, other):
        return Line2D(Point2D(self.tail), Point2D(other.head))
    def __iadd__(self, other):
        self.head = Point2D(other.head)
        return self
    def __neg__(self):
        return Line2D(self.head, self.tail)
    def __pos__(self):
        return Line2D(self)
    def __len__(self):
        return 2
    def __getitem__(self, key):
        return (self.tail, self.head)[key]
    def __setitem__(self, key, value):
        p = [self.tail, self.head]
        p[key] = Point2D(value)
        self.tail, self.head = p
    def __eq__(self, other):
        if isinstance(other, Line2D):
            return self.tail == other.tail and self.head == other.head
        return False
    def __ne__(self, other):
        if isinstance(other, Line2D):
            return self.tail != other.tail or self.head != other.head
        return True
    def intersect(self, other):
        """
        Returns the point where the given lines intersect.
        """
        if self == other:
            raise AttributeError('Identical lines never intersect.')
        x1, y1 = self.tail
        x2, y2 = self.head
        x3, y3 = other.tail
        x4, y4 = other.head
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-14:
            raise AttributeError('Lines are too close to parallel to detect'
                ' intersection.')
        d = 1.0 / d
        c1 = x1 * y2 - y1 * x2
        c2 = x3 * y4 - y3 * x4
        return Point2D(d*(c1*(x3-x4)-c2*(x1-x2)), d*(c1*(y3-y4)-c2*(y1-y2)))
    def move_to(self, point):
        """
        Moves this line definition vector so that its tail is at the given
        point.
        """
        self.translate(point - self.tail)
    def point(self, k):
        """
        Returns a point ``p`` on the line according to::

            p = tail + k * (head - tail)

        where ``k`` is a scalar value and ``p``, ``tail`` and ``head`` are
        2D points.
        """
        return self.tail + k * (self.head - self.tail)
    def translate(self, distance):
        """
        Moves this vector an x and y distance specified by the given Point2D.
        """
        self.tail += distance
        self.head += distance
#
# P-spline calculation
#
def pspline(x, y, x2=None, s=1, nseg=10, deg=3, pdeg=2):
    """
    Computes a penalized B-spline (or P-spline) that fits the data given by
    ``x``, ``y``.
    
    If an argument ``x2`` is given, the spline is evaluated at each point in
    ``x2`` and the resulting values are returned. If no value ``x2`` is given
    the evaluations at ``x`` are returned.
    
    The fitted spline will have ``nseg`` segments of degree ``deg``. Data
    fitting is performed using a penalty of degree ``pdeg`` (e.g. a penalty on
    the ``pdeg``-th derivative). Smoothing is introduced using a smoothing
    factor ``s``.

    Both ``x`` and ``y`` must be 1D arrays of the same length. The length of
    ``x2`` is arbitrary.

    P-splines are described in:
    
        Flexible Smoothing with B-splines and Penalties
        Paul H.C. Eilers and Brian D. Marx
        Statistical Science
        1996, Vol. 11, No. 2, 89-121
        
    This method was adapted from a computer lab script by Paul Eilers, 2007
    http://www.stat.lsu.edu/faculty/marx/
    """
    def bbase(x, xl, xr, nseg=10, deg=3):
        """
        Create a b-spline basis
        
        Arguments::
        
        ``x``
            ?Points at which to evaluate?
        ``xl``
            Left-hand border x-coordinate.
        ``xr``
            Right-hand border x-coordinate.
        ``nseg``
            Number of segments
        ``deg``
            B-spline degree
        """
        # Get knots
        dx = (xr - xl) / float(nseg)
        knots = xl + np.arange(-deg, 1 + nseg + deg) * dx
        # Create ufunc for truncated power function
        def tpower(x, t):#, p):
            """
            Truncated power function (x^p if x > t, else 0)
            """
            return (x > t) * (x - t) ** deg
        tp = np.frompyfunc(tpower, 2, 1)
        # Compute bases
        P = tp.outer(x, knots)
        # For some reason P get dtype object, leads to issues later, so
        P = np.array(P, dtype=x.dtype)
        D = np.diff(np.eye(P.shape[1]), deg + 1, axis=0)
        D /= math.gamma(deg + 1) * dx ** deg
        return np.dot(P, np.transpose(D)) *  (-1) ** (deg + 1)
    # Ensure x and y are numpy arrays
    x = np.array(x, copy=False)
    y = np.array(y, copy=False)
    # Get left bound, right bound
    if x2 is None:
        # Use left and right boundaries of x
        xl = np.min(x)
        xr = np.max(x)
    else:
        # Use left and right boundaries of x2
        x2 = np.array(x2, copy=False)
        xl = min(np.min(x), np.min(x2))
        xr = max(np.max(x), np.max(x2))
    # Compute bases for B-splines
    B = bbase(x, xl, xr, nseg, deg)
    # Compute differences for penalty
    D = np.diff(np.eye(B.shape[1]), pdeg, axis=0)
    P = np.dot(np.transpose(D), D)
    # Compute penalized weighing of B-splines
    a = np.linalg.solve(
            np.dot(np.transpose(B), B) + s * P,
            np.dot(np.transpose(B), y))
    # Evaluate spline and return
    if x2 is None:
        # Evaluate on x
        return np.dot(B, a)
    else:
        # Evaluate on the given data points
        B = bbase(x2, xl, xr, nseg, deg)
        return np.dot(B, a)
class DocumentAction(object):
    """
    Represents an action that can be performed on a document.
    """
    def __init__(self):
        self._performed = False
    def perform(self):
        """
        Performs this action. Return type depends on action.
        """
        if self._performed:
            raise Exception('Action failed: Action cannot be performed'
                ' twice.')
        result = self._perform()
        self._performed = True
        return result
    def undo(self):
        """
        Undoes this action. Return type depends on action.
        """
        if not self._performed:
            raise Exception('Undo failed: Action not yet performed.')
        result = self._undo()
        self._performed = False
        return result
    def _perform(self):
        """
        Used by subclasses to perform an action.
        """
        raise NotImplementedError
    def _undo(self):
        """
        Used by subclasses to undo an action.
        """
        raise NotImplementedError
class DA_AddNode(DocumentAction):
    """
    Adds a child to a parent node.

    Variables can be added using a list of tuples (type, name, value).
    """
    def __init__(self, parent, ntype, name, variables):
        super(DA_AddNode, self).__init__()
        self.parent = parent
        self.name = name
        self.ntype = ntype
        self.child = None
        self.variables = variables
    def _perform(self):
        p = self.parent
        c = self.child
        if c != None:
            p.silent_add_existing_child(c)
        else:
            c = self.child = p.silent_add_child(self.ntype, self.name)
            for t,n,v in self.variables:
                c.silent_add_variable(t, n, v)
        # Signals
        p.get_document().doc_node_added.emit(p, c)
        p.child_added.emit(p, c)
        # Return
        return c
    def _undo(self):
        p = self.parent
        c = self.child
        p.silent_remove_child(c)
        # Signals
        p.get_document().doc_node_removed.emit(p, c)
        p.child_removed.emit(p, c)
        c.node_removed.emit(c)
class DA_RemoveNode(DocumentAction):
    """
    Deletes a child node.
    """
    def __init__(self, node):
        super(DA_RemoveNode, self).__init__()
        self.child = node
        self.parent = node.get_parent_node()
    def _perform(self):
        p = self.parent
        c = self.child
        p.silent_remove_child(c)
        # Signals
        p.get_document().doc_node_removed.emit(p, c)
        p.child_removed.emit(p, c)
        c.node_removed.emit(c)
    def _undo(self):
        p = self.parent
        c = self.child
        p.silent_add_existing_child(c)
        # Signals
        p.get_document().doc_node_added.emit(p, c)
        p.child_added.emit(p, c)
class DA_AddVariable(DocumentAction):
    """
    Adds a DocumentVariable to a DocumentNode
    """
    def __init__(self, parent, vtype, name, value=None):
        super(DA_AddVariable, self).__init__()
        self.parent = parent
        self.vtype = vtype
        self.name = name
        self.value = value
        self.variable = None
    def _perform(self):
        if self.variable != None:
            self.parent.silent_add_existing_variable(self.variable)
        else:
            self.variable = self.parent.silent_add_variable(self.vtype,
                self.name, self.value)
        self.parent.variable_added.emit(self.parent, self.variable)
        return self.variable
    def _undo(self):
        self.parent.silent_remove_variable(self.variable)
        self.parent.variable_removed.emit(self.parent, self.variable)
        self.variable.variable_removed.emit(self.variable)
class DA_RemoveVariable(DocumentAction):
    """
    Removes a DocumentVariable form a DocumentNode
    """
    def __init__(self, parent, variable):
        super(DA_RemoveVariable, self).__init__()
        self.parent = parent
        self.variable = variable
    def _perform(self):
        self.parent.silent_remove_variable(self.variable)
        self.parent.variable_removed.emit(self.parent, self.variable)
        self.variable.variable_removed.emit(self.variable)
    def _undo(self):
        self.parent.silent_add_existing_variable(self.variable)
        self.parent.variable_added.emit(self.parent, self.variable)
class DA_ChangeVariable(DocumentAction):
    """
    Changes a value within a DocumentNode
    """
    def __init__(self, variable, value):
        super(DA_ChangeVariable, self).__init__()
        self.variable = variable
        self.parent = variable.get_node()
        self.new_value = value
        self.old_value = None
    def _perform(self):
        if self.old_value is None:
            self.old_value = self.variable.get_value()
        self.variable.silent_set_value(self.new_value)
        self.parent.variable_changed.emit(self.parent, self.variable)
        self.variable.variable_changed.emit(self.variable)
    def _undo(self):
        self.variable.silent_set_value(self.old_value)
        self.parent.variable_changed.emit(self.parent, self.variable)
        self.variable.variable_changed.emit(self.variable)
class DA_ChangeVariables(DocumentAction):
    """
    Changes one or more variables in a node. ``variables`` must be a dict
    mapping variable names to values.
    """
    def __init__(self, node, variables):
        super(DA_ChangeVariables, self).__init__()
        self.node = node
        self.variables = variables
        self.old_variables = None
    def _perform(self):
        if self.old_variables is None:
            self.old_variables = {}
            for name, value in self.variables.iteritems():
                self.old_variables[name] = \
                    self.node.get_variable(name).get_value()
        for name, value in self.variables.iteritems():
            var = self.node.get_variable(name)
            var.silent_set_value(value)
            self.node.variable_changed.emit(self.node, var)
            var.variable_changed.emit(var)
    def _undo(self):
        for name, value in self.old_variables.iteritems():
            var = self.node.get_variable(name)
            var.silent_set_value(value)
            self.node.variable_changed.emit(self.node, var)
            var.variable_changed.emit(var)
#
# DocumentModel
#
# The DocumentModel allows (read only) access to the model structure for Qt
# widgets like the TreeView.
#
class DocumentModel(QtCore.QAbstractItemModel):
    """
    Implements the Qt abstract item model for a :class:`Document`.
    """
    def __init__(self, document):
        super(DocumentModel, self).__init__(document)
        self.document = document
    def columnCount(self, node):
        """
        Return number of columns in this node
        """
        return 1
    def data(self, index, role):
        """
        Returns the data stored at the given index (where index is given as a
        qt ``QModelIndex``).
        """
        # Only allow valid indices
        if not index.isValid():
            return None
        # Only return nodes for display
        if role != Qt.DisplayRole:
            return None
        # Get the node this index points at
        item = index.internalPointer()
        if index.column() == 1:
            return item.get_ntype()
        else:
            return item.get_name()
    def flags(self, index):
        """
        Returns the relevant flags for the item at this index.
        """
        # All items are enabled
        flags = Qt.ItemIsEnabled
        # You can drop an item anywhere, even outside the range of valid
        # indices.
        #flags |= Qt.ItemIsDropEnabled
        if not index.isValid():
            return flags
        # Allow selecting items
        flags |= Qt.ItemIsSelectable
        # Allow dragging data points
        node = index.internalPointer()
        #if node.can_drag():
        #    flags |= Qt.ItemIsDragEnabled
        # Allow editing items
        #flags |= Qt.ItemIsEditable
        # Return flags
        return flags
    def headerData(self, section, orientation, role):
        """
        Return the header for the row/column requested as ``section``.
        """
        # Only provide header for display, not for editing(?)
        if role != Qt.DisplayRole:
            return None
        # Only provide horizontal header
        if orientation == Qt.Horizontal:
            return ('Name', 'Type')[section]
        else:
            return None
    def index(self, row, column, parent):
        """
        Create an index for the selected row and column, counted from the given
        parent index (as a ``QModelIndex`` object).
        """
        # Test if index is available
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()
        # Get parent DocumentNode
        if parent.isValid():
            parent = parent.internalPointer()
        else:
            parent = self.document
        # Find child, return index
        try:
            child = parent.child(row)
            return self.createIndex(row, column, child)
        except KeyError:
            return QtCore.QModelIndex()
    def parent(self, index):
        """
        Return an index to the parent of the item at the given index.
        """
        if not index.isValid():
            return QtCore.QModelIndex()
        # Get parent
        parent = index.internalPointer().parent()
        # Return index to parent
        if parent is None or parent == self.document:
            return QtCore.QModelIndex()
        else:
            return self.createIndex(parent.index(), 0, parent)
    def rowCount(self, parent):
        """
        Returns the number of children in the :class:`DocumentNode` at the
        given index.
        """
        if parent.column() > 0:
            return 0
        if parent.isValid():
            parent = parent.internalPointer()
        else:
            parent = self.document
        return len(parent)
    def supportedDropActions(self):
        """
        Returns a flag indicating the supported drop actions for QItems.
        """
        return QtCore.Qt.MoveAction
        # | QtCore.Qt.CopyAction
#
# Gui classes
#
class GraphDataExtractor(myokit.gui.MyokitApplication):
    """
    Main window for the graph data extractor.
    """
    def __init__(self, filename=None):
        super(GraphDataExtractor, self).__init__()
        # Set application icon
        self.setWindowIcon(icon())
        # Create gde document
        self._document = GdeDocument(self)
        # Set size, center
        self.resize(800,600)
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        # Add image widget
        self._scene = GdeScene(self)
        self._view  = GdeView(self._scene)
        self.setCentralWidget(self._view)
        self._scene.mouse_moved.connect(self.handle_mouse_moved)
        # Add property docks
        self._document_dock = DocumentEditDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._document_dock)
        self._variable_dock = DocumentVariableEditDock(self,
            self._document_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._variable_dock)
        # Cursor position label on status bar
        self._label_cursor = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self._label_cursor)
        # Menu bar
        self.create_menu()
        # Tool bar
        self.create_toolbar()
        # Status bar
        self.statusBar().showMessage('Ready')
        # Current path
        self._path = QtCore.QDir.currentPath()
        self._file = None
        self._recent_files = []
        # Load settings from ini file
        self.load_config()
        # Load document
        doc = None
        if filename is not None:
            if os.path.isfile(filename):
                doc = filename
        self.load_document(doc)
        self.update_window_title()
    def action_about(self):
        """
        Displays the about dialog.
        """
        self.clear_focus()
        QtWidgets.QMessageBox.about(self, TITLE, ABOUT)
    def action_add_data_point(self, x=0.5, y=0.5):
        """
        Adds a data point to the document.
        """
        self.clear_focus()
        self._document.clear_selection()
        data = self._document.get_active_data_set()
        n = len(data)
        if n == 0:
            name = 'point_1'
        else:
            last = data.child(len(data) - 1)
            last = str(last.get_name())
            name = 'point_' + str(1 + int(last[1+last.find('_'):]))
        p = data.add_child(T_DATA_POINT, name, (
            (V_NORM, 'x', x),
            (V_NORM, 'y', y)))
        p.select()
        return p
    def action_add_data_set(self):
        """
        Adds a new data set to the document.
        """
        self.clear_focus()
        self._document.clear_selection()
        dset = self._document.add_data_set()
        dset.select()
        return dset
    def action_extract(self):
        """
        Extracts the data points and saves them in a csv file.
        """
        data = self._document.get('Data')
        if len(data) == 0:
            QtWidgets.QMessageBox.warning(self, TITLE,
                '<h1>No data to export.</h1>'
                '<pre>Please add some data points first.</pre>')
            return
        self.clear_focus()
        fname = QtWidgets.QFileDialog.getSaveFileName(self,
            'Extract data to plain text csv file',
            self._path, filter='Comma-separated file (*.csv *.txt)')[0]
        if fname:
            fname = str(fname)
            if fname:
                self._document.extract_data(fname)
    def action_license(self):
        """
        Displays this program's licensing information.
        """
        self.clear_focus()
        QtWidgets.QMessageBox.about(self, TITLE, myokit.LICENSE_HTML)
    def action_new(self):
        """
        Lets the user start a new project.
        """
        self.clear_focus()
        self.prompt_save_changes()
        self.load_document()
        self.update_window_title()
    def action_open(self):
        """
        Let the user select an gde file to load.
        """
        self.clear_focus()
        self.prompt_save_changes()
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open gde file',
            self._path, filter='Gde files (*.gde)')[0]
        if fname:
            fname = str(fname)
            if fname:
                self.load_document(fname)
            self.update_window_title()
    def action_open_recent(self):
        """
        Opens a recent file.
        """
        if not self.prompt_save_changes():
            return
        action = self.sender()
        if action:
            filename = str(action.data())
            self.load_document(filename)
    def action_redo(self):
        """
        Redo the last action.
        """
        self.clear_focus()
        self._document.redo()
    def action_revert(self):
        """
        Reloads the current file (if any).
        """
        self.clear_focus()
        if self._document.has_changes() and self._file != None:
            box = QtWidgets.QMessageBox
            reply = box.question(self, TITLE,
                    "Undo all changes since last save?", box.Yes | box.No)
            if reply == box.Yes:
                self.load_document(self._file)
    def action_save(self, save_as=False):
        """
        Saves the current document to a file.
        """
        self.clear_focus()
        if save_as or self._file is None:
            fname = QtWidgets.QFileDialog.getSaveFileName(self,
                'Save gde file', self._path, filter='Gde files (*.gde)')[0]
            if fname:
                fname = str(fname)
                if fname:
                    if os.path.splitext(fname)[1] == '':
                        fname += '.gde'
                    self.set_filename(fname)
                    self._document.write(self._file)
        else:
            self._document.write(self._file)
        self.update_window_title()
    def action_save_as(self):
        """
        Saves the current document to a file.
        """
        self.clear_focus()
        self.action_save(save_as=True)
    def action_set_image(self):
        """
        Sets the image file
        """
        self.clear_focus()
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image file',
            self._path)[0]
        if fname:
            fname = str(fname)
            # Chop off path shared with current file
            if self._file:
                fpath = os.path.abspath(os.path.dirname(fname))
                cpath = os.path.abspath(os.path.dirname(self._file))
                n = len(cpath)
                if fpath[:n] == cpath:
                    fname = os.path.join(fpath[1+n:], os.path.basename(fname))
            if fname != '':
                node = self._document.get('Image')
                node.set_value(path=fname)
                node.select()
    def action_undo(self):
        """
        Undo the last action.
        """
        self.clear_focus()
        self._document.undo()
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
    def clear_focus(self):
        """
        Used to clear focus (potentially finishing edit operations!) before
        any action is performed.
        """
        w = QtWidgets.QApplication.focusWidget()
        if w != None and isinstance(w, DocumentVariableField):
            w.clearFocus()
    def closeEvent(self, event=None):
        """
        Called when window is closed. To force a close (and trigger this
        function, call self.close())
        """
        self.clear_focus()
        if self.prompt_save_changes():
            # Save configuration
            self.save_config()
            if event:
                # Accept the event, close the window
                event.accept()
        else:
            if event:
                # Ignore the event, window stays open
                event.ignore()
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
        self._tool_new.setStatusTip('Start a new project')
        self._tool_new.setIcon(QtGui.QIcon.fromTheme('document-new'))
        self._tool_new.triggered.connect(self.action_new)
        self._menu_file.addAction(self._tool_new)
        # File > Open
        self._tool_open = QtWidgets.QAction('&Open', self)
        self._tool_open.setShortcut('Ctrl+O')
        self._tool_open.setStatusTip('Open a saved project')
        self._tool_open.setIcon(QtGui.QIcon.fromTheme('document-open'))
        self._tool_open.triggered.connect(self.action_open)
        self._menu_file.addAction(self._tool_open)
        # File > ----
        self._menu_file.addSeparator()
        # File > Save
        self._tool_save = QtWidgets.QAction('&Save', self)
        self._tool_save.setShortcut('Ctrl+S')
        self._tool_save.setStatusTip('Save the current project')
        self._tool_save.setIcon(QtGui.QIcon.fromTheme('document-save'))
        self._tool_save.triggered.connect(self.action_save)
        self._menu_file.addAction(self._tool_save)
        # File > Save As...
        self._tool_saveas = QtWidgets.QAction('&Save As...', self)
        self._tool_saveas.setShortcut('Ctrl+Shift+S')
        self._tool_saveas.setStatusTip('Save the current project as a new'
            ' file')
        self._tool_saveas.setIcon(QtGui.QIcon.fromTheme('document-save-as'))
        self._tool_saveas.triggered.connect(self.action_save_as)
        self._menu_file.addAction(self._tool_saveas)
        # File > Revert
        self._tool_revert = QtWidgets.QAction('&Revert', self)
        self._tool_revert.setStatusTip('Revert to the last saved state')
        self._tool_revert.setIcon(QtGui.QIcon.fromTheme('document-revert'))
        self._tool_revert.triggered.connect(self.action_revert)
        self._menu_file.addAction(self._tool_revert)
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
        self._tool_exit.setIcon(QtGui.QIcon.fromTheme('application-exit'))
        self._tool_exit.triggered.connect(self.close)
        self._menu_file.addAction(self._tool_exit)
        # Edit menu
        self._menu_edit = self._menu.addMenu('&Edit')
        # Edit > Undo
        self._tool_undo = QtWidgets.QAction('&Undo', self)
        self._tool_undo.setStatusTip('Undo the last action')
        self._tool_undo.setShortcut('Ctrl+Z')
        self._tool_undo.setIcon(QtGui.QIcon.fromTheme('edit-undo'))
        self._tool_undo.setEnabled(False)
        self._tool_undo.triggered.connect(self.action_undo)
        self._menu_edit.addAction(self._tool_undo)
        # Edit > Redo
        self._tool_redo = QtWidgets.QAction('&Redo', self)
        self._tool_redo.setStatusTip('Redo the last undone action')
        self._tool_redo.setShortcut('Ctrl+Shift+Z')
        self._tool_redo.setIcon(QtGui.QIcon.fromTheme('edit-redo'))
        self._tool_redo.setEnabled(False)
        self._tool_redo.triggered.connect(self.action_redo)
        self._menu_edit.addAction(self._tool_redo)
        # Edit > ----
        self._menu_edit.addSeparator()
        # Edit > Set image
        self._tool_image = QtWidgets.QAction('Set image', self)
        self._tool_image.setStatusTip('Select an image file.')
        self._tool_image.setIcon(QtGui.QIcon.fromTheme('insert-image'))
        self._tool_image.triggered.connect(self.action_set_image)
        self._menu_edit.addAction(self._tool_image)
        # Edit > Add data set
        self._tool_data_set = QtWidgets.QAction('Add data set', self)
        self._tool_data_set.setStatusTip('Add a new data set.')
        self._tool_data_set.setIcon(QtGui.QIcon.fromTheme('insert-object'))
        self._tool_data_set.triggered.connect(self.action_add_data_set)
        self._menu_edit.addAction(self._tool_data_set)
        # Edit > Add data point
        #self._tool_data_point = QtWidgets.QAction('Add data point', self)
        #self._tool_data_point.setStatusTip('Add a data point.')
        #self._tool_data_point.setIcon(QtGui.QIcon.fromTheme('list-add'))
        #self._tool_data_point.triggered.connect(self.action_add_data_point)
        #self._menu_edit.addAction(self._tool_data_point)     
        # Edit > -----
        self._menu_edit.addSeparator()
        # Edit > Extract data
        self._tool_extract = QtWidgets.QAction('Extract data', self)
        self._tool_extract.setStatusTip('Extract the data points.')
        self._tool_extract.setShortcut('Ctrl+E')
        self._tool_extract.setIcon(QtGui.QIcon.fromTheme('document-send'))
        self._tool_extract.triggered.connect(self.action_extract)
        self._menu_edit.addAction(self._tool_extract)
        # Help menu
        self._menu_help = self._menu.addMenu('&Help')
        # Help > About
        self._tool_about = QtWidgets.QAction('&About', self)
        self._tool_about.setStatusTip('View information about this program.')
        self._tool_about.triggered.connect(self.action_about)
        self._menu_help.addAction(self._tool_about)
        self._tool_license = QtWidgets.QAction('&License', self)
        self._tool_license.setStatusTip('View this program\'s license info.')
        self._tool_license.triggered.connect(self.action_license)
        self._menu_help.addAction(self._tool_license)
    def create_toolbar(self):
        """
        Creates this widget's toolbar
        """
        self._toolbar = self.addToolBar('tools')
        self._toolbar.setFloatable(False)
        self._toolbar.setMovable(False)
        #self._toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toolbar.addAction(self._tool_new)
        self._toolbar.addAction(self._tool_open)
        self._toolbar.addAction(self._tool_save)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._tool_undo)
        self._toolbar.addAction(self._tool_redo)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._tool_image)
        self._toolbar.addAction(self._tool_data_set)
        #self._toolbar.addAction(self._tool_data_point)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._tool_extract)
    def handle_action_exception(self, document, exception):
        """
        Handles exceptions during an action exception perform or undo method.
        """
        QtWidgets.QMessageBox.warning(self, TITLE,
            '<h1>An error has occurred.</h1>'
            '<pre>' + traceback.format_exc() + '</pre>')
    def handle_mouse_moved(self, x, y):
        """
        Displays the current cursor position.
        """
        xx, yy = self._document.norm2real(x, y)
        F = '{:< 1.3g}'
        self._label_cursor.setText('(' + F.format(x) + ', ' + F.format(y) +')'
            ' :: (' + F.format(xx) + ', ' + F.format(yy) +')')
    def handle_node_selected(self, node):
        """
        Handles changes in document selection.
        """
        if node.get_ntype() == T_AXIS_REFERENCE_POINT:
            msg = 'Drag to move.'
            msg += ' Hold Ctrl to create orthogonal axis.'
            self.statusBar().showMessage(msg)
        else:
            self.statusBar().clearMessage()
    def handle_undo_redo_change(self, document):
        """
        Handles changes to undo/redo history.
        """
        self._tool_undo.setEnabled(document.can_undo())
        self._tool_redo.setEnabled(document.can_redo())
    def load_config(self):
        """
        Loads the user configuration from an ini file.
        """
        # Read ini file
        inifile = os.path.expanduser(SETTINGS_FILE)
        if not os.path.isfile(inifile):
            return
        config = configparser.ConfigParser()
        config.read(inifile)
        # Window dimensions and location
        if config.has_section('window'):
            g = self.geometry()
            def getor(name, alt):
                if config.has_option('window', name):
                    return int(config.get('window', name))
                return alt
            x = getor('x', g.x())
            y = getor('y', g.y())
            w = getor('w', g.width())
            h = getor('h', g.height())
            self.setGeometry(x, y, w, h)
        # Current files, directory, etc
        if config.has_section('files'):
            if config.has_option('files', 'path'):
                path = config.get('files', 'path')
                if os.path.isdir(path):
                    self._path = path
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
    def load_document(self, filename=None):
        """
        Loads a document into the editor.
        """
        if self._document:
            self._document.delete()
        self._document = GdeDocument(self, filename)
        self.set_filename(filename)
        # Add to recent files
        if filename is not None:
            self.add_recent_file(filename)
        # React to changes in undo/redo status
        self._document.action_exception.connect(self.handle_action_exception)
        self._document.undo_redo_change.connect(self.handle_undo_redo_change)
        # React to node selection
        self._document.node_selected.connect(self.handle_node_selected)
        # Tell docks about new document
        self._document_dock.set_document(self._document)
        self._variable_dock.set_document(self._document)
        # Tell scene about new document
        self._scene.set_document(self._document)
        # Tell undo/redo buttons about change
        self.handle_undo_redo_change(self._document)
    def prompt_save_changes(self):
        """
        Checks if there were any changes made to this document, if so, it
        prompts the user to save the current project.

        Returns True if the user selected "yes" or "no" or there were no
        changes to save. Returns False if the user selected "cancel".
        """
        if self._document.has_changes():
            box = QtWidgets.QMessageBox
            reply = box.question(self, TITLE,
                    "Save changes to document?", box.Yes | box.No | box.Cancel)
            if reply == box.Yes:
                self.action_save()
                return True
            elif reply == box.No:
                return True
            else:
                return False
        return True
    def save_config(self):
        """
        Saves the user configuration to an ini file.
        """
        config = configparser.ConfigParser()
        # Window dimensions and location
        config.add_section('window')
        #if (self.IsFullScreen() or self.IsMaximized()):
        #    config.set('window', 'maximized', 'True')
        #else:
        #    config.set('window', 'maximized', 'False')
        g = self.geometry()
        config.set('window', 'x', str(g.x()))
        config.set('window', 'y', str(g.y()))
        config.set('window', 'w', str(g.width()))
        config.set('window', 'h', str(g.height()))
        # Current and recent files
        config.add_section('files')
        config.set('files', 'path', self._path)
        for k, filename in enumerate(self._recent_files):
            config.set('files', 'recent_' + str(k), filename)
        # Write configuration to ini file
        inifile = os.path.expanduser(SETTINGS_FILE)
        with open(inifile, 'wb') as configfile:
            config.write(configfile)
    def set_filename(self, filename=None):
        """
        Changes the current filename.
        """
        if filename:
            self._file = filename
            self._path = os.path.abspath(os.path.dirname(filename))
            os.chdir(self._path)
            # Add to recent files
            self.add_recent_file(self._file)
        else:
            self._file = None
    def update_recent_files_menu(self):
        """
        Updates the recent files menu.
        """
        for k, filename in enumerate(self._recent_files):
            t = self._recent_file_tools[k]
            t.setText(str(k + 1) + '. ' + os.path.basename(filename))
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
        self.setWindowTitle(title)
#
# Text based document editing.
#
# The text-based, tree view of the document is implemented using the widgets
# below.
#
class DocumentEditDock(QtWidgets.QDockWidget):
    """
    A dock for selecting properties to edit.
    """
    def __init__(self, parent):
        super(DocumentEditDock, self).__init__(parent=parent)
        self.setWindowTitle('Browse properties')
        # Add tree view
        self._tree = DocumentTreeView(self)
        self.setWidget(self._tree)
    def set_document(self, document):
        """
        Sets or replaces this dock's document.
        """
        self._document = document
        self._tree.set_document(document)
class DocumentTreeView(QtWidgets.QTreeView):
    """
    Treeview for documents.
    """
    def __init__(self, parent):
        super(DocumentTreeView, self).__init__(parent=parent)
        self._document = None        
        #self.setDragEnabled(True)
        #self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.show()
    def node_selected_local(self, event):
        """
        Selection changed in this view.
        """
        selection = self._sm.selectedRows()
        if len(selection) > 0:
            node = selection[0].internalPointer()
            if not node.is_selected():
                self._document.clear_selection()
                node.select()
    def node_selected(self, node):
        """
        Selection changed in any view.
        """
        if node != None:
            new = node.get_model_selection()
            if new != self._sm.selection():
                self._sm.clearSelection()
                self._sm.select(new,QtCore.QItemSelectionModel.Select)
                index = node.get_model_index()
                while index.isValid():
                    self.expand(index)
                    index = index.parent()
    def set_document(self, document):
        """
        Sets or replaces this view's document.
        """
        self._document = document
        # Set document model
        self.setModel(document.get_model())
        # Set single/multiple selection mode
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # Expand tree a little
        self.expandToDepth(2)
        # React to changes in selected node
        document.node_selected.connect(self.node_selected)
        # React to local changes in selection
        self._sm = self.selectionModel()
        self._sm.selectionChanged.connect(self.node_selected_local)
        self.clearSelection()
class DocumentVariableEditDock(QtWidgets.QDockWidget):
    """
    A dock for editing a node's variables.
    """
    def __init__(self, parent, select_dock):
        super(DocumentVariableEditDock, self).__init__(parent)
        self.setWindowTitle('Edit properties')
        self._select = select_dock
        self._document = None
        # Add scroll area with editor
        self._editor = DocumentVariableList(self)
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._editor)
        self.setWidget(self._scroll)
    def set_document(self, document):
        """
        Sets or replaces this dock's document.
        """
        self._editor.set_document(document)
class DocumentVariableList(QtWidgets.QWidget):
    """
    Displays a list of properties and allows them to be edited.
    """
    def __init__(self, parent):
        super(DocumentVariableList, self).__init__(parent)
        # Add layout
        self._vbox = QtWidgets.QVBoxLayout(self)
        self._grid = QtWidgets.QGridLayout()
        self._vbox.addLayout(self._grid)
        self._vbox.addStretch(1)
        self.setLayout(self._vbox)
        # List of kids
        self._kids = []
    def set_document(self, document):
        """
        Tells this variable list to work with the given document.
        """
        self._node = None
        # Listen for deleted nodes
        document.doc_node_removed.connect(self.handle_node_deleted)
        # React to changes in selected node
        document.node_selected.connect(self.handle_node_selected)
    def handle_node_deleted(self, document, node):
        """
        Called when a node is deleted.
        """
        if node == self._node:
            self.set_node(None)
    def handle_node_selected(self, node):
        """
        Called when a node is selected in one of the views.
        """
        self.set_node(node)
    def set_node(self, node):
        """
        Sets or replaces this variable list's node.
        """
        # Clear existing kids
        while self._kids:
            k = self._kids.pop()
            self._grid.removeWidget(k)
            k.deleteLater()
            del k
        # Check if node was given
        if node is None:
            return
        # Load new kids
        for k, var in enumerate(node.iterdata()):
            label = QtWidgets.QLabel(var.get_name(), parent=self)
            field = DocumentVariableField.create(self, var)
            self._grid.addWidget(label, k, 0)
            self._grid.addWidget(field, k, 1)
            self._kids.append(label)
            self._kids.append(field)
        # Save this node!
        self._node = node
class DocumentVariableField(QtWidgets.QWidget):
    """
    Abstract class to be extended by variable edit fields of different types
    """
    def __init__(self, parent, variable):
        super(DocumentVariableField, self).__init__(parent)
        self._variable = variable
        # Add document change listener
        variable.variable_changed.connect(self.handleVariableChanged)
    @staticmethod
    def create(parent, variable):
        """
        Creates a new DocumentVariableField of the appropriate class.
        """
        vtype = variable.get_vtype()
        if vtype == V_BOOL:
            return BoolVariableField(parent, variable)
        else:
            return TextVariableField(parent, variable)        
    def handleFieldChanged(self):
        """
        Called when the field initiates a change.
        """
        pass
    def handleVariableChanged(self, variable):
        """
        Called when the underlying variable is changed (including changes made
        by this field).
        """
        pass
class TextVariableField(QtWidgets.QLineEdit, DocumentVariableField):
    """
    Editor field for text based values.
    """
    def __init__(self, parent, variable):
        super(TextVariableField, self).__init__(parent, variable)
        vtype = variable.get_vtype()
        if vtype == V_STR:
            pass
        elif vtype == V_INT:
            self.setValidator(QtGui.QIntValidator())
        elif vtype == V_FLOAT:
            self.setValidator(QtGui.QDoubleValidator())
        elif vtype == V_NORM:
            self.setValidator(QtGui.QDoubleValidator(0, 1, 10))
        elif vtype == V_PATH:
            pass
        else:
            raise ValueError('Unsupported variable type <' + str(vtype) + '>.')
        self.setText(variable.get_str_value())
        self.editingFinished.connect(self.handleFieldChanged)
    def focusOutEvent(self, event):
        """
        Called when this field loses focus.
        """
        self.setModified(True)
        self.handleFieldChanged()
        return QtWidgets.QLineEdit.focusOutEvent(self, event)
    def handleFieldChanged(self):
        """
        Called when the value is changed by the user.
        """
        if self.isModified():
            self._variable.set_value(self.text())
            self.setModified(False)
    def handleVariableChanged(self, variable):
        """
        Called when the underlying variable is changed (including changes made
        by this field).
        """
        self.setText(self._variable.get_str_value())
class BoolVariableField(QtWidgets.QCheckBox, DocumentVariableField):
    """
    Editor field for text based values.
    """
    def __init__(self, parent, variable):
        super(BoolVariableField, self).__init__(parent, variable)
        if variable.get_vtype() != V_BOOL:
            raise ValueError('Unsupported variable type <' + str(vtype) + '>.')
        #self.setText(variable.get_str_value())
        self.setChecked(variable.get_value())
        self.stateChanged.connect(self.handleFieldChanged)
    def handleFieldChanged(self):
        """
        Called when this field initiates a change.
        """
        checked = self.isChecked()
        if self._variable.get_value() != checked:
            self._variable.set_value(checked)
    def handleVariableChanged(self, variable):
        """
        Called when the underlying variable is changed (including changes made
        by this field).
        """
        checked = self._variable.get_value()
        if self.isChecked() != checked:
            self.setChecked(checked)
#
# Graphical widgets,
#
# The graphical editor is implemented using the custom widgets defined below.
#
class GdeView(QtWidgets.QGraphicsView):
    """
    Views the main GDE scene.
    """
    def __init__(self, scene):
        super(GdeView, self).__init__(scene)
        # Always track mouse position
        self.setMouseTracking(True)
        # Set crosshair cursor
        self.setCursor(Qt.CrossCursor)
        # Set rendering hints
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        #self.setViewportUpdateMode(
        #    QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        # Fit scene rect in view
        self.fitInView(self.sceneRect())
        self.setAlignment(Qt.AlignLeft)
    def resizeEvent(self, event):
        """
        Called when the view is resized.
        """
        self.fitInView(self.sceneRect())
        #return event
class GdeScene(QtWidgets.QGraphicsScene):
    """
    Editable scene displaying the current Gde document
    """
    # Signals
    # Somebody moved the mouse
    # Attributes: (norm x, norm y)
    mouse_moved = QtCore.Signal(float, float)
    # Width and scale
    W = 10000
    S = 1.0 / W
    def __init__(self, gde):
        self._gde = gde
        self._document = None
        # Image
        self._image = None
        # Axes
        self._xaxis = None
        self._yaxis = None
        # Data points (regardless of the series they belong to)
        self._data = {}
        # Data set items (splines)
        self._sets = {}        
        # Init
        super(GdeScene, self).__init__(gde)
        self.setBackgroundBrush(QtGui.QColor(255, 255, 255))
        self.setSceneRect(0, 0, self.W, self.W)
    def clear(self):
        """
        Clears this scene, disconnects from document.
        """
        # Disconnect from document
        doc = self._document
        if doc:
            doc.doc_node_added.disconnect(self.handle_node_added)
            doc.doc_node_removed.disconnect(self.handle_node_removed)
            doc.doc_deleted.disconnect(self.clear)
            # Disconnect any listeners from child items
            self._image.disconnect()
            self._xaxis.disconnect()
            self._yaxis.disconnect()
            #for item in self._data.itervalues():
            #    item.disconnect()
            for item in self._sets.itervalues():
                item.disconnect()
        # Child items will disconnect when deleted by refcount decrease.
        self._image = None
        self._xaxis = None
        self._yaxis = None
        #self._data = {}
        self._sets = {}
        self._document = None
        super(GdeScene, self).clear()
    def get_item_for_node(self, node):
        """
        Returns the first scene item found for the given node.
        """
        def deep(parent):
            for child in parent.children():
                if child._node == node:
                    return child
                item = deep(child)
                if item != None:
                    return item
            return None
        return deep(self)
    def handle_node_added(self, parent, child):
        """
        A node is added to the document.
        """
        ntype = child.get_ntype()
        if ntype == T_DATA_SET:
            # Add data set item
            item = DataSetItem(child)
            self._data[child] = item
            self.addItem(item)
    def handle_node_removed(self, parent, child):
        """
        A node is removed from the document.
        """
        ntype = child.get_ntype()
        if ntype == T_DATA_SET:
            try:
                item = self._sets[child]
                del(self._sets[child])
                self.removeItem(item)
            except KeyError:
                pass
    def mouseMoveEvent(self, event):
        """
        Show mouse position in status bar
        """
        p = event.scenePos()
        x, y = p.x() * self.S, p.y() * self.S
        x = 0 if x < 0 else 1 if x > 1 else x
        y = 0 if y < 0 else 1 if y > 1 else y
        self.mouse_moved.emit(x, y)
        return QtWidgets.QGraphicsScene.mouseMoveEvent(self, event)
    def mouseDoubleClickEvent(self, event):
        """
        Double-click: set background image.
        """
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() == Qt.NoModifier:
                self._gde.action_set_image()
                return
        return QtWidgets.QGraphicsScene.mouseDoubleClickEvent(self, event)
    def mousePressEvent(self, event):
        """
        Ctrl-Click: add data point
        """
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() == Qt.ControlModifier:
                p = event.scenePos()
                x, y = self.scene2norm(p.x(), p.y())
                self._gde.action_add_data_point(x, y)
        return QtWidgets.QGraphicsScene.mousePressEvent(self, event)
    def norm2scene(self, x, y):
        """
        Converts normalised coordinates to scene coordinates.
        """
        return x * self.W, y * self.W
    def scene2norm(self, x,  y):
        """
        Converts scene coordinates to normalised coordinates.
        """
        return x * self.S, y * self.S
    def set_document(self, document):
        """
        Constructs a scene based on a document
        """
        self.clear()
        self._document = document
        # Add image
        self._image = ImageItem(document.get('Image'))
        self.addItem(self._image)
        # Add axes
        axes = document.get('Axes')
        xax = axes.get('x')
        yax = axes.get('y')
        self._xaxis = AxisItem(xax)
        self._yaxis = AxisItem(yax)
        self.addItem(self._xaxis)
        self.addItem(self._yaxis)
        # Add data sets
        for series in document.get('Data'):
            item = DataSetItem(series)
            self._sets[series] = item
            self.addItem(item)
        # Listen for new data sets
        document.doc_node_added.connect(self.handle_node_added)
        document.doc_node_removed.connect(self.handle_node_removed)
        document.doc_deleted.connect(self.clear)
class SceneItem(QtWidgets.QGraphicsItem):
    """
    Item on the scene that is connected to a part of the model.
    
    Items are hierarchical: most items have ``parent=None``, but items can be
    created as child items by setting the parent argument to another 
    :class:`SceneItem`.
    """
    def __init__(self, node, parent=None):
        self._node = node
        self._original_z_index = 0
        super(SceneItem, self).__init__(parent=parent)
        # React to node selection changes
        node.node_selected.connect(self.select)
        node.node_deselected.connect(self.deselect)
        # React to node variable changes
        node.variable_changed.connect(self.handle_variable_changed)
        # React to document deletion
        node.get_document().doc_deleted.connect(self.handle_document_deleted)
    def deselect(self):
        """
        Deselects this node, if possible.
        """
        if self.flags() and QtWidgets.QGraphicsItem.ItemIsSelectable:
            if self.isSelected():
                self.setSelected(False)
                self.setZValue(self._original_z_index)
    def disconnect(self):
        """
        Disconnect all listeners connected to by this item. Implementing
        subclasses that override this method should still make a call to
        the parent method.
        """
        node = self._node
        if node:
            node.node_selected.disconnect(self.select)
            node.node_deselected.disconnect(self.deselect)
            node.variable_changed.disconnect(self.handle_variable_changed)
            node.get_document().doc_deleted.disconnect(
                self.handle_document_deleted)
    def handle_document_deleted(self, doc):
        """
        Called when the document this item's node is in is deleted.
        """
        #self.set_node(None)
        pass
    def handle_variable_changed(self, node, variable):
        """
        Called when a variable inside this item's node is changed.
        """
        pass
    def itemChange(self, change, value):
        """
        Handles event when this item has changed
        """
        if change == QtWidgets.QGraphicsItem.ItemSceneHasChanged:
            self.set_scene(value)
        return QtWidgets.QGraphicsItem.itemChange(self, change, value)
    def init(self, scene, node):
        """
        Called when this item needs to be (re-)set to the given scene and node.
        Both scene and node may be None.
        """
        pass
    def get_node(self):
        """
        Returns this item's :class:`DocumentNode`.
        """
        return self._node
    def get_npos(self):
        """
        Returns this item's normalised coordinates.
        """
        p = self.scenePos()
        s = self.scene()
        return s.scene2norm(p.x(), p.y())
    def select(self):
        """
        Selects this node, if possible.
        """
        if self.flags() and QtWidgets.QGraphicsItem.ItemIsSelectable:
            if not self.isSelected():
                self.setSelected(True)
                self.setFocus()
                self._original_z_index = self.zValue()
                self.setZValue(Z_SELECTED)
    def set_node(self, node=None):
        """
        Sets or replaces this item's node.
        """
        self._node = node
        scene = self.scene()
        if scene != None:
            self.init(scene, node)
    def set_npos(self, x, y):
        """
        Sets this item's position based on the normalized coordinates (x, y)
        """
        scene = self.scene()
        if scene != None:
            self.setPos(*scene.norm2scene(x, y))
    def set_scene(self, scene):
        """
        Sets or replaces this item's scene
        """
        if self._node != None:
            self.init(scene, self._node)
class DraggableItem(SceneItem):
    """
    Item on the scene that can be picked up and moved around.
    """
    # Signals
    # Item was moved by the user.
    item_moved = QtCore.Signal(float, float)
    def __init__(self, node, parent=None):
        super(DraggableItem, self).__init__(node, parent)
        # Allow moving and selecting
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
        # Allow keyboard focus & keyboard events
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable)
        # Enable move and geometry change events
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        # Set cursor, only allow left mouse button
        self.setCursor(Qt.OpenHandCursor)
        self.setAcceptedMouseButtons(Qt.LeftButton);
        # Used to gather dragging into a single event
        self._original_location = False        
    def mousePressEvent(self, event):
        """
        Mouse pressed? Then show drag icon and allow moving
        """
        self.setCursor(Qt.OpenHandCursor)
        return QtWidgets.QGraphicsItem.mousePressEvent(self, event)
    def mouseMoveEvent(self, event):
        """
        Item being moved using the mouse.
        """
        self.setCursor(Qt.BlankCursor)
        if not self._original_location:
            self._original_location = self.pos()
        return QtWidgets.QGraphicsItem.mouseMoveEvent(self, event)
    def mouseReleaseEvent(self, event):
        """
        Mouse button released again.
        """
        self.setCursor(Qt.OpenHandCursor)
        if self._original_location:
            p = self.pos()
            if p != self._original_location:
                self.handle_drag(p.x(), p.y())
            self._original_location = False
        return QtWidgets.QGraphicsItem.mouseReleaseEvent(self, event)
    def itemChange(self, change, value):
        """
        Stay on the scene. (Like a sex machine).
        """
        scene = self.scene()
        node = self.get_node()
        if scene:
            if change == QtWidgets.QGraphicsItem.ItemPositionChange:
                # Item has been moved, check if it stays within the scene
                rect = scene.sceneRect()
                if not rect.contains(value):
                    value.setX(min(rect.right(), max(value.x(), rect.left())))
                    value.setY(min(rect.bottom(), max(value.y(), rect.top())))
                x, y = self.restrict_movement(value.x(), value.y())
                value.setX(x)
                value.setY(y)
                self.handle_drag_live(x, y)
            elif change == QtWidgets.QGraphicsItem.ItemSelectedHasChanged:
                # Item has been selected or deselected
                if value:
                    node.select()
                else:
                    node.deselect()
        # Call parent's method
        super(DraggableItem, self).itemChange(change, value)
        # Pass on event
        return QtWidgets.QGraphicsItem.itemChange(self, change, value)
    def handle_drag(self, x, y):
        """
        Called if the user has dragged to node to scene coordinates (x, y) and
        released it.
        """
        pass
    def handle_drag_live(self, x, y):
        """
        Called while the user is dragging the node to scene coordinates (x, y).
        """
        pass
    def restrict_movement(self, x, y):
        """
        Allows restricted movement of points. Called with the scene
        coordinates. Should return a tuple (x, y).
        """
        return x, y
class ImageItem(SceneItem):
    """
    Draws a full-size background image onto the scene.
    """
    def __init__(self, node):
        super(ImageItem, self).__init__(node)
        self.setZValue(Z_BACKGROUND)
        self._image = None
        # Allow selecting
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
    def boundingRect(self):
        """
        Returns this item's bounding rectangle.
        """
        s = self.scene()
        if s:
            return s.sceneRect()
    def handle_variable_changed(self, node, variable):
        """
        Variable in item's node has changed.
        """
        self.init(self.scene(), node)
    def init(self, scene, node):
        """
        Initialize to the given scene and node.
        """
        self._image = None
        if node != None:
            path = node.get_value('path')
            if path:
                try:
                    self._image = QtGui.QImage(node.get_value('path'))
                except Exception:
                    pass
        if scene != None:
            self.update()
    def paint(self, painter, option, widget):
        """
        Paints this item.
        """
        is_none = self._image is None
        is_null = (not is_none) and self._image.isNull()
        if not (is_none or is_null):
            painter.setOpacity(0.5)
            painter.drawImage(self.scene().sceneRect(), self._image)
        else:
            text = 'Double-click to select image file...\n' \
                   'Ctrl-click to add data points'
            if is_null:
                text = '<< Invalid image file selected >>\n\n' + text
            painter.setOpacity(0.7)
            painter.setFont(QtGui.QFont('Decorative', 200))
            painter.drawText(self.scene().sceneRect(), QtCore.Qt.AlignCenter,
                text)
class AxisItem(SceneItem):
    """
    Data axis
    """
    # Signals
    def __init__(self, node):
        super(AxisItem, self).__init__(node)
        # Which axis am I?
        self._isx = node.get_name() == 'x'
        # Reference points and axis coordinates
        self._r1 = None
        self._r2 = None
        self._coords = None
        # Set z-value
        self.setZValue(Z_AXIS)
        # Create pen
        self._pen = QtGui.QPen()
        self._pen.setWidth(0)
        self._pen.setColor(QtGui.QColor(0, 0, 0))
    def boundingRect(self):
        """
        Returns this item's bounding rectangle.
        """
        if self._coords:
            # Returned value should _not_ include children.
            x1, y1, x2, y2 = self._coords
            return QtCore.QRectF(x1, y1, x2-x1, y2-y1)
        else:
            return QtCore.QRectF(0, 0, 1, 1)
    def init(self, scene, node):
        """
        Initialize to the given node and scene
        """
        # Create reference points
        if node != None:
            ref1_node = node.get('ref1')
            ref2_node = node.get('ref2')
        else:
            ref1_node = None
            ref2_node = None
        # Set nodes
        if self._r1 is None:
            self._r1 = AxisPointItem(ref1_node, self)
        else:
            self._r1.set_node(ref1_node)
        if self._r2 is None:
            self._r2 = AxisPointItem(ref2_node, self)
        else:
            self._r2.set_node(ref2_node)
        # Initialize kids
        if scene != None:
            self._r1.set_scene(scene)
            self._r2.set_scene(scene)
        # Update coordinates
        if self._r1 != None and self._r2 != None:
            self.update_coords()
    def is_x(self):
        """
        Returns true if this is an x-axis.
        """
        return self._isx
    def paint(self, painter, option, widget):
        """
        Paints this reference point
        """
        if self._coords:
            painter.drawLine(*self._coords)
    def get_sibling_ref(self, ref):
        """
        Given reference point 1, this method returns point 2 and vice versa.
        """
        return self._r1 if ref == self._r2 else self._r2
    def update_coords(self):
        """
        Updates the coordinates of this axis.
        """
        scene = self.scene()
        if scene != None and self._r1 != None and self._r2 != None:
            # Get scene coordinates
            r = scene.sceneRect()
            sx1 = r.x()
            sy1 = r.y()
            sx2 = sx1 + r.width()
            sy2 = sy1 + r.height()
            # Add padding to scene coordinates
            p = 10
            sx1 -= p
            sy1 -= p
            sx2 += p
            sy2 += p
            # Find slope
            r1 = self._r1.pos()
            r2 = self._r2.pos()
            r1 = r1.x(), r1.y()
            r2 = r2.x(), r2.y()
            slope = r2[0] - r1[0]
            if slope < 0:
                r1, r2 = r2, r1
                slope *= -1
            # Set coordinates
            x0, y0 = r1
            if abs(slope) < 1e-9:
                x1, y1 = x0, sy1
                x2, y2 = x0, sy2
            else:
                slope = (r2[1] - r1[1]) / float(slope)
                if abs(slope) > 0.5:
                    # Steep angle, draw from top to bottom
                    slope = 1.0 / slope
                    x1, y1 = x0 + slope * (sy1 - y0), sy1
                    x2, y2 = x0 + slope * (sy2 - y0), sy2
                else:
                    # Flat angle, draw from left to right
                    x1, y1 = sx1, y0 + slope * (sx1 - x0)
                    x2, y2 = sx2, y0 + slope * (sx2 - x0)
            self._coords = (x1, y1, x2, y2)
            # Update scene
            scene.update()
class AxisPointItem(DraggableItem):
    """
    Reference point for an axis.
    """
    def __init__(self, node, axis):
        super(AxisPointItem, self).__init__(node, parent=axis)
        # Save axis
        self._axis = axis
        # Dimensions
        self._r = 300
        self._d = 2 * self._r
        # Create pen
        self._pen = QtGui.QPen()
        self._pen.setWidth(0)
        self._pen.setColor(QtGui.QColor(0, 128, 0))
        # Restriction on movement
        self._restrict = False
        self._dragging = False
        self._sibling = None
        self._isx = axis.is_x()
        # Set z-value
        self.setZValue(Z_AXIS)
    def boundingRect(self):
        """
        Returns this item's bounding rectangle.
        """
        return QtCore.QRectF(-self._r, -self._r, self._d, self._d)
    def get_npos(self):
        """
        Returns this item's normalised coordinates.
        """
        p = self.scenePos()
        s = self.scene()
        return s.scene2norm(p.x(), p.y())
    def handle_drag(self, x, y):
        """
        Item has been dragged and released.
        """
        x, y = self.scene().scene2norm(x, y)
        self.get_node().set_value(x=x, y=y)
        self._dragging = False
        self._restrict = False
    def handle_drag_live(self, x, y):
        """
        Item is being dragged.
        """
        self._dragging = True
        self._axis.update_coords()
    def handle_variable_changed(self, node, variable):
        """
        Variable in item's node has changed.
        """
        self.set_npos(node.get_value('x'), node.get_value('y'))
    def init(self, scene, node):
        """
        Initialize to the given scene and node.
        """
        if node != None and scene != None:
            self.set_npos(node.get_value('x'), node.get_value('y'))
            self.update()
        self._sibling = self._axis.get_sibling_ref(self)
    def keyPressEvent(self, event):
        """
        A key is pressed while this item has focus.
        """
        if self._dragging and event.key() == Qt.Key_Control:
            self._restrict = True
    def keyReleaseEvent(self, event):
        """
        A key is released while this item has focus.
        """
        if self._dragging and event.key() == Qt.Key_Control:
            self._restrict = False
    def paint(self, painter, option, widget):
        """
        Paints this reference point
        """
        painter.setPen(self._pen)
        r1 = self._r
        d1 = self._d
        r2 = 0.75 * r1
        r3 = 0.75 * r2
        d3 = 2.0 * r3
        if self.isSelected() or self.hasFocus():
            painter.drawEllipse(-r1, -r1, d1, d1)
        painter.drawEllipse(-r3, -r3, d3, d3)
        painter.drawLine(-r2, 0, r2, 0)
        painter.drawLine(0, -r2, 0, r2)
    def restrict_movement(self, x, y):
        """
        Used to restrict the drag movement of this item if ctrl is held.
        """
        if self._restrict:
            rp = self._sibling.pos()
            if self._isx:
                return x, rp.y()
            else:
                return rp.x(), y
        else:
            return x, y
    def set_npos(self, x, y):
        """
        Repositions this item using normalised coordinates (range 0..1)
        """
        super(AxisPointItem, self).set_npos(x, y)
        self._axis.update_coords()
class DataSetItem(SceneItem):
    """
    Data Set item. Usually empty, but can draw an interpolating p-spline.
    """
    # Signals
    def __init__(self, node):
        super(DataSetItem, self).__init__(node)
        # Set z-value
        self.setZValue(Z_DATA_SET)
        # Data point items
        self._data = {}
        # Path
        self._path = None        
        # Create pen
        self._pen = QtGui.QPen()
        self._pen.setWidth(50)
        self._pen.setColor(QtGui.QColor(0, 0, 255, 128))
        # React to child addition / removal
        node.child_added.connect(self.handle_child_added)
        node.child_removed.connect(self.handle_child_removed)
    def boundingRect(self):
        """
        Returns this item's bounding rectangle.
        """
        if self._path:
            scene = self.scene()       
            if scene:
                w, h = scene.norm2scene(1, 1)
                return QtCore.QRectF(0, 0, w, h)
        return QtCore.QRectF(0, 0, 1, 1)
    def disconnect(self):
        """
        Disconnects any listeners attached to this item.
        """
        node = self._node
        if node:
            node.child_added.disconnect(self.handle_child_added)
            node.child_removed.disconnect(self.handle_child_removed)
        super(DataSetItem, self).disconnect()
    def handle_child_added(self, parent, child):
        """
        Handle addition of a data point.
        """
        item = DataPointItem(child, self)
        self._data[child] = item
        scene = self.scene()
        if scene:
            item.set_scene(scene)
        self.update_spline()
    def handle_child_removed(self, parent, child):
        """
        Handle removal of a data point.
        """
        try:
            item = self._data[child]
        except KeyError:
            return
        del(self._data[child])
        # Remove child item (it's a bit weird...)
        item.setParentItem(None)
        item.deselect()
        scene = self.scene()
        if scene:
            scene.removeItem(item)
        # Disconnect any listeners
        item.disconnect()
        self.update_spline()
    def handle_variable_changed(self):
        """
        Handles the event where a child variable is changed.
        """
        self.update_spline()
    def init(self, scene, node):
        """
        Initialize to the given node and scene
        """
        # Add data point items
        if node != None:
            for child in node:
                item = DataPointItem(child, self)
                self._data[child] = item
                if scene:
                    item.set_scene(scene)
            self.update_spline()
    def paint(self, painter, option, widget):
        """
        Paints this spline
        """
        if self._path is None:
            return
        # Create painter
        #painter = QtGui.QPainter()
        #painter.begin(self)
        # Set pen
        painter.setPen(self._pen)
        # Draw path
        painter.drawPath(self._path)
    def update_spline(self):
        """
        Creates a spline based on the current data set.
        """
        # Get node & scene, don't draw spline if nothing found
        self._path = None
        node = self.get_node()
        scene = self.scene()
        if node is None or scene is None:
            return
        # Spline enabled?
        enabled = self._node.get_value('spline')
        if not enabled:
            scene.update() # Removes spline if previously drawn
            return
        # Gather data points
        x1 = []
        y1 = []
        for point in self._node:
            x1.append(point.get_value('x'))
            y1.append(point.get_value('y'))
        x1 = np.array(x1, copy=False)
        xmin = np.min(x1)
        xmax = np.max(x1)            
        # Fewer than two points? Then don't draw anything
        if len(x1) < 2:
            return
        # Get spline parameters
        smo = self._node.get_value('smoothing')
        pen = self._node.get_value('penalty')
        deg = self._node.get_value('degree')
        seg = self._node.get_value('segments')
        sam = self._node.get_value('samples')        
        if seg < 1:
            # Segments = 0 means add a segment per data point
            seg = len(x1)
        # Points to evaluate the spline at
        x2 = np.linspace(xmin, xmax, sam)
        # Reticulate spline
        y2 = pspline(x1, y1, x2, smo, seg, deg, pen)
        # Create path
        path = QtGui.QPainterPath()
        x, y = iter(x2), iter(y2)
        path.moveTo(*scene.norm2scene(x.next(), y.next()))
        for i in xrange(1, sam):
            path.lineTo(*scene.norm2scene(x.next(), y.next()))
        self._path = path
        # Redraw
        self.update()
class DataPointItem(DraggableItem):
    """
    Data point view on the image.
    """
    def __init__(self, node, parent):
        # Parent data set item
        self._parent = parent
        # Dimensions
        self._r = 240
        self._d = 2 * self._r
        # Create pen
        self._pen = QtGui.QPen()
        self._pen.setWidth(0)
        self._pen.setColor(QtGui.QColor(0, 0, 255))
        # Create item
        super(DataPointItem, self).__init__(node, parent=parent)
        # Set z-value
        self.setZValue(Z_DATA)
    def boundingRect(self):
        """
        Returns this item's bounding rectangle.
        """
        return QtCore.QRectF(-self._r, -self._r, self._d, self._d)
    def handle_drag(self, x, y):
        """
        Item has been dragged and released.
        """
        x, y = self.scene().scene2norm(x, y)
        self.get_node().set_value(x=x, y=y)
    def handle_variable_added(self, node, variable):
        """
        Variable added to item's node
        """
        if node.has_value('x') and node.has_value('y'):
            self.set_npos(node.get_value('x'), node.get_value('y'))
            self._parent.update_spline()
    def handle_variable_changed(self, node, variable):
        """
        Variable in item's node has changed.
        """
        node = self.get_node()
        if node.has_value('x'):
            self.set_npos(node.get_value('x'), node.get_value('y'))
            self._parent.update_spline()
    def keyPressEvent(self, event):
        """
        A key is pressed while this item has focus.
        """
        if event.key() == Qt.Key_Delete:
            self.get_node().remove()
        else:
            return QtWidgets.QGraphicsItem.keyPressEvent(self, event)
    def paint(self, painter, option, widget):
        """
        Paints this reference point
        """
        painter.setPen(self._pen) #TODO Get Pen from parent
        r1 = self._r
        d1 = self._d
        r2 = 0.75 * r1
        r3 = 0.75 * r2
        d3 = 2.0 * r3
        if self.isSelected() or self.hasFocus():
            painter.drawEllipse(-r1, -r1, d1, d1)
        painter.drawEllipse(-r3, -r3, d3, d3)
        painter.drawLine(-r2, 0, r2, 0)
        painter.drawLine(0, -r2, 0, r2)
    def set_scene(self, scene):
        """
        Scene is set or replaced.
        """
        node = self.get_node()
        if node != None:
            if node.has_value('x'):
                self.set_npos(node.get_value('x'), node.get_value('y'))
