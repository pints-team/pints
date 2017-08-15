# 
# Defines the python classes that represent a myocyte model.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
from collections import OrderedDict
import math
import re
import myokit
try:
    from cStringIO import StringIO
except ImportError: #Python3
    from io import StringIO
TAB = ' '*4
NAME = re.compile(r'^[a-zA-Z]\w*$')
META = re.compile(r'^[a-zA-Z]\w*(:[a-zA-Z]\w*)*$')
def check_name(name):
    """
    Tests if the given name is a valid myokit name and raises a
    :class:`myokit.InvalidNameError` if it isn't. Names are returned converted
    to ``str``.
    """
    # Check str compatibility
    name = str(name)
    # Check name syntax
    if NAME.match(name) is None:
        raise myokit.InvalidNameError('The name <' + str(name) + '> is '
            ' invalid. The first character of a name should be a letter from'
            ' the range [a-zA-Z]. Any subsequent characters can be taken from'
            ' the set [a-zA-Z0-9_].')
    # Check for keywords
    if name in myokit.KEYWORDS:
        raise myokit.InvalidNameError('The name <' + str(name) + '> is a'
            ' reserved keyword.')
    return name
class MetaDataContainer(dict):
    """
    Dictionary that stores meta-data.
    """
    def __getitem__(self, key):
        # Check key
        if META.match(key) is None:
            raise myokit.InvalidMetaDataNameError('The key <' + str(key) + '>'
                ' is not a valid meta-data property identifier.')
        return super(MetaDataContainer, self).__getitem__(key)
    def __setitem__(self, key, item):
        # Check item
        item = str(item)
        # Check key
        if META.match(key) is None:
            raise myokit.InvalidMetaDataNameError('The key <' + str(key) + '>'
                ' is not a valid meta-data property identifier.')
        super(MetaDataContainer, self).__setitem__(key, item)
class ObjectWithMeta(object):
    """
    Base class for objects with meta data.
    
    Meta-data properties are all stored in a dict and should be string:string
    mappings.
    """
    def __init__(self):
        super(ObjectWithMeta, self).__init__()
        self.meta = MetaDataContainer()
    def _clone_metadata(self, clone):
        """
        Clones this object's metadata into ``clone``.
        """
        for k, v in self.meta.iteritems():
            clone.meta[str(k)] = str(v)
    def _code_meta(self, b, tabs=0, ignore=None):
        """
        Internal method to format meta data as part of the code() operation.
        Can be used by all subclasses to uniformly format their meta data.
        A list of meta property keys to omit (ignore) can be added.
        """
        if not ignore:
            ignore = []
        for k,v in sorted(self.meta.iteritems()):
            if k in ignore:
                continue
            v = str(v)
            key = TAB * tabs + k.strip() + ': '
            eol = '\n'
            if ('\n' in v) or ('\r' in v) or (v.strip() == ''):
                v.replace('\r\n', '\n')
                v.replace('\r', '\n')
                b.write(key + '"""\n')
                pre = TAB * (1 + tabs)
                for line in v.split(eol):
                    b.write(pre + line + eol)
                b.write(pre + '"""\n')
            else:
                b.write(key + v + eol)
class ModelPart(ObjectWithMeta):
    """
    Base class for model parts.
    """
    def __init__(self, parent, name):
        """
        Creates a new ModelPart

        The given parent should be a ModelPart or None. The name should be
        unique within the set of children for the given parent.
        """
        super(ModelPart, self).__init__()
        self._parent = parent   # This object's parent
        self._model = None      # The model this object belongs to
        self._name = str(name)  # Local name
        self._uname = None      # Globally unique name
        self._token = None      # Parser token for errors, not always set
    def _clone_modelpart_data(self, clone):
        """
        Clones this ModelPart's data into ``clone``.
        """
        clone._uname = str(self._uname)
        self._clone_metadata(clone)
    def code(self):
        """
        Returns this object in ``mmt`` syntax.
        """
        b = StringIO()
        self._code(b, 0)
        return b.getvalue()
    def _code(self, b, t):
        """
        Internal version of _code(), to be implemented by all subclasses.

        The argument ``t`` specifies the number of tabs to indent the code
        with. The argument ``b`` is a cStringIO buffer.
        """
        raise NotImplementedError
    def _delete(self):
        """
        Tells this object it's being deleted. Removes links to parent and/or
        model.
        """
        self._parent = None
        self._model = None
    def has_ancestor(self, obj):
        """
        Returns ``True`` if the given object ``obj`` is an ancestor of this
        :class:`ModelPart`.
        """
        par = self._parent
        try:
            while par is not None:
                if par == obj:
                    return True
                par = par._parent
        except AttributeError:
            return False
    def is_ancestor(self, obj):
        """
        Returns ``True`` if this object is an ancestor of the given
        :class:`ModelPart`` ``object``.
        """
        return obj.has_ancestor(self)
    def model(self):
        """
        Returns the model this object belongs to (if set).
        """
        if self._model is None:
            self._model = self.parent(Model)
        return self._model
    def name(self):
        """
        Returns this object's name.
        """
        return self._name
    def parent(self, kind=None):
        """
        Returns this object's parent.

        If the optional variable ``kind`` is set, the method will scan until an
        instance of the requested type is found.
        """
        if kind is None:
            return self._parent
        parent = self._parent
        while parent is not None and not isinstance(parent, kind):
            parent = parent._parent
        return parent
    def qname(self, hide = None):
        """
        Returns this object's fully qualified name. That is, an object named
        ``y`` with a parent named ``x`` will return the string ``x.y``. If
        ``y`` has a child ``z`` its qname will be ``x.y.z``.

        If the optional argument ``hide`` is set to this object's parent the
        parent qualifier will be omitted.
        """
        if self._parent is None or self._parent == hide:
            return self._name
        return self._parent.qname(hide) + '.' + self._name
    def __repr__(self):
        return '<' + str(type(self)) + '(' + self.qname() + ')>'
    def __str__(self):
        return self.qname()
    def uname(self):
        """
        Returns a globally unique name for this object.
        """
        return self._uname
class VarProvider(object):
    """
    *Abstract class*

    This class provides an iterator over variables and equations for any object
    that can provide access to an iterator over its variables.
    """
    def _create_variable_stream(self, deep, sort):
        """
        Returns a stream over this object's variables.

        When the argument ``deep`` is set to ``True``, the stream should
        include any nested variables as well. (Only in contexts where this
        makes sense. Otherwise, ``deep`` can be ignored.)

        When the argument ``sort`` is set to ``True``, the stream should be
        returned in a consistent order.
        """
        raise NotImplementedError
    def count_equations(self, const=None, inter=None, state=None, bound=None,
            deep=False):
        """
        Returns the number of equations matching the given criteria. See
        :meth:`equations` for an explanation of the arguments.
        """
        return len(list(self.equations(const, inter, state, bound, deep)))
    def count_variables(self, const=None, inter=None, state=None, bound=None,
            deep=False):
        """
        Returns the number of variables matching the given criteria. See
        :meth:`variables` for an explanation of the arguments.
        """
        return len(list(self.variables(const, inter, state, bound, deep)))
    def has_equations(self, const=None, inter=None, state=None, bound=None,
            deep=False):
        """
        Returns True if there are any equations that can be returned by calling
        :meth:``equations`` with the same arguments.
        """
        for x in self.equations(const, inter, state, bound, deep):
            return True
        return False
    def has_variable(self, name):
        """
        Returns True if the given name corresponds to a variable in this
        object. Accepts both single names ``x`` and qualified names ``x.y`` as
        input.

        This function performs the same search as ``variable``, so in most
        cases it will be more efficient to call ``variable()`` in a
        try-catch block rather than checking for existence explicitly.
        """
        try:
            self.var(name)
        except KeyError as e:
            return False
        return True
    def has_variables(self, const=None, inter=None, state=None, bound=None,
            deep=False):
        """
        Returns True if there are any variables that can be returned by calling
        :meth:``variables`` with the same arguments.
        """
        for x in self.variables(const, inter, state, bound, deep):
            return True
        return False
    def equations(self, const=None, inter=None, state=None, bound=None,
            deep=False):
        """
        Creates and returns a filtered iterator over the equations of this
        object's variables.

        The returned values can be filtered using the following arguments:

        ``const=True|False|None``
            Set to ``True`` to return only constants' equations. ``False`` to
            exclude all constants and any other value to ignore this check.

            For a definition of "constant variable" see
            :meth:`variables`.

        ``inter=True|False|None``
            Set to ``True`` to return only intermediary variables' equations,
            ``False`` to exclude all intermediary variables and any other value
            to ignore this check.

            For a definition of "intermediary variable" see
            :meth:`variables`.

        ``state=True|False|None``
            Set to ``True`` to return only state variables' equations,
            ``False`` to exclude all state variables and any other value to
            ignore this check.
            
        ``bound=True|False|None``
            Set to ``True`` to return only bound variables' equations,
            ``False`` to exclude all bound variables and any other value to
            ignore this check.

        ``deep=True|False`` (by default it's ``False``)
             Set to ``True`` to include the equations of nested variables
             meeting all other criteria.
        """
        def viter(stream):
            for x in stream:
                yield x.eq()
            raise StopIteration
        return viter(self.variables(const, inter, state, bound, deep))
    def variables(self, const=None, inter=None, state=None, bound=None,
            deep=False, sort=False):
        """
        Creates and returns a filtered iterator over the contained variables.

        The returned values can be filtered using the following arguments:

        ``const=True|False|None``
            Constants are defined as variables that do not depend on state
            variables or derivatives. In other words, any variable whose value
            can be determined before starting an ODE solving routine.

            Set to ``True`` to return only constants, ``False`` to exclude all
            constants and any other value to ignore this check.

        ``inter=True|False|None``
            Intermediary variables are those variables that are not constant
            but are not part of the state. In other words, intermediary
            variables are the variables that need to be calculated at every
            step of an ODE solving routine before calculating the ODE
            derivatives and updating the state.

            Set to ``True`` to return only intermediary variables, ``False`` to
            exclude all intermediary variables and any other value to ignore
            this check.

        ``state=True|False|None``
            Set to True to return only state variables, False to exclude all
            state variables and any other value to ignore this check.

        ``bound=True|False|None``
            Set to True to return only variables bound to an external value,
            False to exclude all bound variables and any other value to forgo
            this check.

        ``deep=True|False`` (by default it's ``False``)
             Set to True to return nested variables meeting all other criteria.

        ``sort=True|False`` (by default it's ``False``)
            Set to True to return the variables in a consistent order.
        """
        def viter(stream, const, inter, state, bound, deep):
            for var in stream:
                # Filter constants
                if const == True or const == False:
                    if const != var._is_constant: continue
                # Filter intermediary variables
                if inter == True or inter == False:
                    if inter != var._is_intermediary: continue
                # Filter states
                if state == True or state == False:
                    if state != var._is_state: continue
                # Filter bound variables
                if bound == True or bound == False:
                    if bound != var._is_bound: continue
                # Yield this variable
                yield var
            raise StopIteration
        stream = self._create_variable_stream(deep, sort)
        return viter(stream, const, inter, state, bound, deep)
    def var(self, name):
        """
        Searches for the given variable and returns it if found. Accepts both
        single names ``x`` and qualified names ``x.y`` as input.
        """
        names = name.split('.')
        var = self
        for x in names:
            var = var[x]
        return var
class VarOwner(ModelPart, VarProvider):
    """
    Represents a holder of Variables
   
    Minimal dictionary support is provided: A variable named "x" can be
    obtained from an owner ``m`` using ``m["x"]``. The number of variables in
    ``m`` is given by ``len(m)`` and the presence of "x" in ``m`` can be tested
    using ``if "x" in m:``.
    """
    def __init__(self, parent, name):
        super(VarOwner, self).__init__(parent, name)
        self._variables = {}
        # Set component
        self._component = self
        while type(self._component) != Component:
            self._component = self._component.parent()
    def add_variable(self, name):
        """
        Adds a child variable with the given `name` to this :class:`VarOwner`.
        """
        if not self.can_add_variable(name):
            raise myokit.DuplicateName('The name <' + str(name) + '> is'
                ' already in use within this scope.')
        try:
            var = None
            self._variables[name] = var = Variable(self, name)
        finally:
            self.model()._reset_validation()
            if var: var._reset_cache()
        return var
    def add_variable_allow_renaming(self, name):
        """
        Attempts to add a child variable with the given `name` to this
        :class:`VarOwner`, but uses a different name if this causes any
        conflicts.
        
        The new variable's name will be modified by appending `_1`, `_2`, etc.
        until the conflict is resolved.
        
        This method can be used when symbolically manipulating a model in
        situations where the exact names are unimportant.
        
        Returns the newly created variable.
        """
        try:
            return self.add_variable(name)
        except myokit.DuplicateName:
            n_tries = 256
            for i in xrange(n_tries):
                try:
                    return self.add_variable(name + '_' + str(1 + i))
                except DuplicateName:
                    pass
            raise Exception('Unable to add variable after maximum number of'
                ' attempts (' + str(n_tries) + ').')
    def can_add_variable(self, name):
        """
        Returns ``True`` if a variable can be added to this :class:`VarOwner`
        under the given ``name``.

        This method is automatically called by :meth:`add_variable()` and
        `move_variable()`, there is no need to call it before using these
        methods.
        """
        name = check_name(name)
        # Determine if variable name would clash with an existing name:
        # Scenario 1: There is an alias accessible to this varowner that
        # clashes with this name
        if self._component.has_alias(name):
            return False
        # Scenario 2: One of this VarOwner's ancestors already contains a
        # variable of that name.
        par = self.parent()
        while type(par) != Model:
            if name in par:
                return False
            par = par.parent()
        # Scenario 3: One of this VarOwner's descendants already has that name.
        for var in self.variables(deep=True):
            if var.name() == name:
                return False
        # It's free!
        return True
    def __contains__(self, key):
        return key in self._variables
    def _create_variable_stream(self, deep, sort):
        if deep:
            if sort:
                def viter(owner):
                    for n,v in sorted(owner._variables.iteritems()):
                        yield v
                        for w in v._create_variable_stream(True, True):
                            yield w
                return viter(self)
            else:
                def viter(owner):
                    for v in owner._variables.itervalues():
                        yield v
                        for w in v._create_variable_stream(True, False):
                            yield w
                return viter(self)
        else:
            if sort:
                def viter(owner):
                    for n, v in sorted(owner._variables.iteritems()):
                        yield v
                return viter(self)
            else:
                return self._variables.itervalues()
    def get(self, name, class_filter=None):
        """
        Searches for a variable with the given ``qname`` and returns it.
        
        To return only objects of a certain class, pass it in as
        ``class_filter``.
        
        The qnames are specified relative to the :class:`VarOwner`. For
        example, in a model with a component ``ina`` and a variable ``ina.h``
        we expect the following results
        
            >>> import myokit
            ... m = myokit.load_model('example')
            ... c = m.get('ina')        # Retrieves the component <ina>
            ... h = c.get('h')          # Retrieves the variable <ina.h>
            ... x = c.get('ina.h')      # Searches for <ina.ina.h>: KeyError!
            
        """
        if isinstance(name, ModelPart):
            return name
        names = name.split('.')
        x = self
        try:
            for name in names:
                x = x[name]
        except KeyError:
            raise KeyError(str(name))
        if class_filter:
            if not isinstance(x, class_filter):
                raise KeyError(str(name) + ' of class ' + str(class_filter))
            pass
        return x                
    def __getitem__(self, key):
        return self._variables[key]
    def __len__(self):
        return len(self._variables)
    def move_variable(self, variable, new_parent, new_name=None):
        """
        Moves the given variable to another :class:`VarOwner` ``new_parent``.
        In addition, this method can be used to rename variables (either with
        or without moving them).
        """
        if variable.parent() != self:
            raise ValueError('move_variable failed: variable <'
                + variable.qname() + '> does not have parent <'
                + self.qname() + '>.')
        old_name = variable.name()
        if new_name:
            new_name = str(new_name)
        else:
            new_name = old_name
        if new_parent == self and new_name == old_name:
            return
        if not new_parent.can_add_variable(new_name):
            raise myokit.DuplicateName('The name <' + new_name + '> is'
                ' already in use as a variable name within this scope.')
        try:
            # Change listing in VarOwner objects
            del(self._variables[old_name])
            new_parent._variables[new_name] = variable
            # Change variable's _parent and _name attribute
            variable._parent = new_parent
            variable._name   = new_name
        finally:
            variable._reset_cache()
            self.model()._reset_validation()
    def remove_variable(self, variable, recursive=False):
        """
        Removes the given variable from this :class:`VarOwner` and from the
        model.
        
        If ``recursive`` is ``True``, any child variables will be deleted as
        well.
        """
        if variable.parent() != self:
            raise ValueError('remove_variable failed: variable <'
                + variable.qname() + '> does not have parent <'
                + self.qname() + '>.')
        # Handle internal variable deletion steps
        variable._delete(recursive=recursive)
        try:
            # Remove from this VarOwner
            del(self._variables[variable.name()])
        finally:
            self.model()._reset_validation()
    def _remove_variable_internal(self, variable):
        """
        Removes the given variable from this :class:`VarOwner` but doesn't do
        any of the bookkeeping steps.
        """
        del(self._variables[variable.name()])
    def _resolve(self, name):
        """
        Resolves a local variable name to a variable. Raises an
        :class:`UnresolvedReferenceError` if the name doesn't correspond to any
        variable accessible from this :class:`VarOwner's <VarOwner>` scope.
        """
        def sa(name):
            # Suggest alternative
            m = self.model()
            (var, sug, msg) = m.suggest_variable(name)
            return msg
        # Try resolving as an alias
        try:
            return self._component._alias_map[name]
        except KeyError:
            pass
        # Resolve as a local variable name
        dot = name.find('.')
        if dot < 0:
            # Local name given: child of this VarOwner or child of ancestor
            par = self
            while type(par) != Model:
                try:
                    return par[name]
                except KeyError:
                    par = par.parent()
            raise myokit.UnresolvedReferenceError(name, sa(name))
        else:
            # Component given. Resolve as direct child of component
            model = self.model()
            cname = name[0:dot]
            vname = name[1+dot:]
            try:
                comp = model[cname]
            except KeyError:
                raise myokit.UnresolvedReferenceError(name, sa(name))
            try:
                return comp[vname]
            except KeyError:
                raise myokit.UnresolvedReferenceError(name, sa(name))
class Model(ObjectWithMeta, VarProvider):
    """
    Represents an electrophysiological cell model, structured in components.
    
    Components can be added to the model using :meth:`add_component()`. Access
    to a model's component is provided through the :meth:`get()` method and
    :meth:`components()`.
    
    Minimal dictionary support is provided: A component named "x" can be
    obtained from a model ``m`` using ``m["x"]``. The number of components in
    ``m`` is given by ``len(m)`` and the presence of "x" in ``m`` can be tested
    using ``if "x" in m:``.
    
    Variables stored inside components can be accessed using :meth:`get()` or
    :meth:`itervalues()`. Values defined through their derivative make up the
    model state and can be accessed using :meth:`states()`. States have
    initial values accessible through :meth:`inits()`.
    
    A model's validity can be checked using :meth:`is_valid()`, which returns
    the latest validation status and :meth:`validate()`, which (re)validates
    the model. Warnings can be obtained using :meth:`warnings()`
    
    The optional constructor argument ``name`` can be used to set a meta
    property "name".
    
    Meta-data properties can be accessed via the property ``meta``, for example
    ``model.meta['key']= 'value'``.
    """
    def __init__(self, name=None):
        super(Model, self).__init__()
        # A dictionary of components
        self._components = {}
        # The model's state variables
        self._state = []
        # The model's current state (list of floats)
        self._current_state = []
        # A dict mapping binding names to variables
        self._bindings = {}
        # A dict mapping label names to variables
        self._labels = {}
        # A set of user functions
        self._user_functions = {}
        # A list of warnings about the model's integrity
        self._warnings = []
        # A list of unique names (for easier export)
        # Some names may be taken up by system functions etc
        self._reserved_unames = set()
        self.reserve_unique_names(*myokit.KEYWORDS)
        # A dictionary token_start : (token, object) relating some (not all!)
        #  tokens to a model. Will be filled by parser when reading a model.
        self._tokens = {}
        # Validation status: True, False or None (not tested)
        self._valid = None
        # Name meta property
        if name:
            self.meta['name'] = str(name)
    def add_component(self, name):
        """
        Adds a component with the given `name` to this model.

        This method resets the model's validation status.
        """
        name = check_name(name)
        # Can't be called engine
        if name == 'external':
            raise myokit.InvalidNameError('"engine" is a reserved component'
                ' name.')
        # Check for duplicate names
        if name in self:
            raise myokit.DuplicateName('There is already a component'
                ' named <' + str(name) + '> in this model.')
        try:
            self._components[name] = comp = Component(self, name)
        finally:
            self._valid = None
        return comp
    def add_component_allow_renaming(self, name):
        """
        Attempts to add a component with the given `name` to this model, but
        uses a different name if this causes any conflicts.
        
        The new component's name will be modified by appending `_1`, `_2`, etc.
        until the conflict is resolved.
        
        This method can be used when symbolically manipulating a model in
        situations where the exact names are unimportant.
        
        Returns the newly created component.
        """
        try:
            return self.add_component(name)
        except myokit.DuplicateName:
            n_tries = 256
            for i in xrange(n_tries):
                try:
                    return self.add_component(name + '_' + str(1 + i))
                except DuplicateName:
                    pass
            raise Exception('Unable to add component after maximum number of'
                ' attempts (' + str(n_tries) + ').')
    def add_function(self, name, arguments, template):
        """
        Adds a user function to this model.
        """
        name = check_name(name)
        # Check argument names and uniqueness
        arguments = [myokit.Name(check_name(x)) for x in arguments]
        n = len(arguments)
        if len(set(arguments)) != n:
            for k, arg in enumerate(arguments):
                if arg in arguments[k+1:]:
                    raise myokit.DuplicateFunctionArgument('The argument name'
                        ' <' + str(arg) + '> is already in use in this'
                        ' function.')
        # Check function uniqueness. Add number of arguments to name to allow
        # overloading
        uname = name + '(' + str(n) + ')'
        if uname in self._user_functions:
            raise myokit.DuplicateFunctionName('A function called "' + name 
                + '" with ' + str(n) + ' arguments is already defined in this'
                'model.')
        # Check template
        refs = template.references()
        for ref in refs:
            if isinstance(ref, myokit.Derivative):
                raise myokit.InvalidFunction('The dot() operator cannot be'
                    ' used in user functions.')
        # Check for unused arguments, undeclared arguments
        ref_names = set([x._value for x in refs])
        for arg in arguments:
            arg = arg._value
            if not arg in ref_names:
                raise myokit.InvalidFunction('The function argument <'
                    + arg + '> is declared but never used.')
            ref_names.remove(arg)
        if ref_names:
            raise myokit.InvalidFunction('The variable <' + ref_names.pop()
                + '> is used in the function but never declared.')
        # Create function and return
        func = UserFunction(name, arguments, template)
        self._user_functions[uname] = func
        return func
    def binding(self, binding):
        """
        Returns the variable with the binding label ``binding``. If no such
        variable is found, ``None`` is returned.
        """
        try:
            return self._bindings[binding]
        except KeyError:
            return None
    def bindings(self):
        """
        Returns an iterator over all (binding label : variable) mappings in
        this model.
        """
        # New dict allows removing labels using this iterator
        return dict(self._bindings).iteritems()
    def check_units(self, mode=myokit.UNIT_TOLERANT):
        """
        Checks the units used in this model. Models can specify units in two 
        ways:
        
        1. By setting a Variable unit. This is done using the ``in`` keyword in
           ``mmt`` syntax or through the method
           :meth:`myokit.Variable.set_unit()`. This specifies the unit the
           variable's value should be in.
        2. By adding units to the literals in variables' right hand
           expressions. This is done using square brackets in ``mmt`` syntax
           (for example ``5 [m] / 10 [s]``) or by adding a unit when creating
           a Number object, for example ``Number(2, myokit.parse_unit('mV')``.
           
        Per variable, the unit check proceeds in two steps:
        
        1. The unit resulting from the variable's rhs is evaluated. This may
           trigger an :class:`myokit.IncompatibleUnitExpression` if any
           inompatibilities are found in the expression.
        2. The calculated unit is compared with the variable unit. An
           ``IncompatibleUnitError`` will be triggered if the two units don't
           match.
        
        The way unspecified units (i.e. ``unit=None``) will be handled depends
        on the used `mode`:
        
        In strict mode (``mode=myokit.UNIT_STRICT``), all unspecified units in
        expressions are treated as implying "dimensionless". For example, the
        expression ``None * [mV]`` will read as ``[1] * [mV]`` which will
        result in the unit ``[mV]``. This mode highlights problems with
        expressions such as ``5 + V``, where ``V`` is in ``[mV]``. The correct
        way to write this for the strict check to pass is by specifying a unit
        for the literal five: ``5 [mV] + V``. Functions requiring dimensionless
        input, such as ``exp()`` will trigger an error if the input is not
        dimensionless. For example::
        
            exp(V + 5)
            
        should be written as
        
            exp((V + 5[mV]) / 1 [mV])
            
        to pass the check in strict mode.
        
        In tolerant mode, expressions involving a ``None`` will assume it
        represents either ``[1]`` or a compatible unit. For example
        ``None * [mV]`` will be read as ``[1] * [mV]``, while ``None + [A]``
        will be treated as ``[A] + [A]``. Functions requiring dimensionless
        input will simply discard the unit if incompatible.
        
        In both modes, the second step is performed in a tolerant manner for
        unspecified units: If either the variable unit or the calculated unit
        is unspecified, the check proceeds without raises exceptions. Without
        this rule, specifications of constants would become needlessly complex.
        For example::
        
            gNa = 10 [mS]
            
        would have to be written as
        
            gNa = 10 [mS] in [mS]
        
        """
        # Check variable units against calculated units (and check caluclated
        # units in the process).
        t = self.time_unit()
        for var in self.variables(deep=True):
            v = var.unit()
            e = var.rhs().eval_unit(mode=mode)
            if e is not None and var.is_state():
                e *= t
            if v != e and v is not None and e is not None:
                raise myokit.IncompatibleUnitError('Incompatible units in <'
                    + var.qname() + '> Variable unit ' + str(v)
                    + ' differs from calculated unit ' + str(e) + '.')
    def clone(self):
        """
        Returns a deep clone of this model.
        """
        clone = Model()
        # Copy meta data
        self._clone_metadata(clone)
        # Clone component/variable structure
        for c in self._components.itervalues():
            c._clone1(clone)
        # Clone state
        for k, v in enumerate(self._state):
            clone.get(v.qname()).promote(self._current_state[k])
        # Create mapping of old var references to new references
        lhsmap = {}
        for v in self.variables(deep=True):
            lhsmap[myokit.Name(v)] = myokit.Name(clone.get(v.qname()))
        for v in self.states():
            lhsmap[myokit.Derivative(myokit.Name(v))] = myokit.Derivative(
                myokit.Name(clone.get(v.qname())))
        # Clone component/variable contents (equations, references)
        for k, c in self._components.iteritems():
            c._clone2(clone[k], lhsmap)
        # Copy unique names
        clone.reserve_unique_names(*iter(self._reserved_unames))
        return clone
    def code(self, line_numbers=False):
        """
        Returns this model in ``mmt`` syntax.

        Line numbers can be added by setting ``line_numbers=True``.
        """
        b = StringIO()
        b.write('[[model]]\n')
        self._code(b, 0)
        if line_numbers:
            lines = b.get_value().split('\n')
            out = []
            n = 1 + int(math.ceil(math.log10(len(lines))))
            for k, line in enumerate(lines):
                out.append('%*d ' % (n, 1 + k) + line)
            return '\n'.join(out)
        else:
            return b.getvalue()
    def _code(self, b, t):
        """
        Internal version of Model.code()
        """
        # b = buffer, t = number of tabs
        # Meta properties
        self._code_meta(b, 0)
        # Initial state
        if self._state:
            pre = t*TAB
            b.write(pre + '# Initial values\n')
            names = [eq.lhs.code() for eq in self.inits()]
            n = max([len(name) for name in names])
            names = iter(names)
            for eq in self.inits():
                name = names.next()
                b.write(pre + name + ' ' * (n - len(name)) + ' = '
                    + eq.rhs.code() + '\n')
            b.write(pre + '\n')
        # Components
        for c in self.components(sort=True):
            c._code(b, t)
    def components(self, sort=False):
        """
        Returns an iterator over this model's component objects.
        """
        if sort:
            def i(s):
                for k,v in s:
                    yield v
            return i(sorted(self._components.iteritems()))
        else:
            return self._components.itervalues()
    def component_cycles(self):
        """
        Checks this model for cyclical references between components. For
        example if ``a.p`` depends on ``b.q`` while ``b.x`` depends on ``a.y``.
        
        The returned value is a list of "cycles", for example, if the only
        cycle found was `a > b > a`, the method would return::
        
            [
                [Component('a'), Component('b'), Component('a')]
            ]
            
        If two components cause a cycle by multiple routes, this is counted as
        a single cycle. The list of cycles returned is sorted from most
        occurences to fewest. The cycle chosen to represent an often occurring
        route is chosen by searching for the smallest cycle. Because there is
        some randomness in the way the model is traversed, this may not be the
        same route every time the method is called.
        
        To get a quick overview of the cycles in a model, use::
        
            for c in m.component_cycles():
                print(' > '.join([x.name() for x in c]))
        
        For big models, this method can take a considerable amount of time to
        execute. To just check if there are any interdependent components, use
        the faster method :meth:`has_interdependent_components()`.
        """
        # Collect dependencies between components in a structure:
        # compdeps = {
        #       c1 : set([c2, c3, c4, ...]),
        #       c2 : set([c3, c5, ...]),
        #        :
        #        }
        compdeps = dict([(c, set()) for c in self.components()])
        for var in self.variables(deep=True):
            c1 = var.parent(Component)
            d1 = compdeps[c1]
            for ref in var.rhs().references():
                if not ref.is_state_value():
                    c2 = ref.var().parent(Component)
                    if c1 != c2:
                        d1.add(c2)
        # Follow each component, find cycles
        cycles = []
        followed = set()
        def follow(comp, trail):
            if comp in trail:
                cycles.append(trail[trail.index(comp):] + [comp])
                return
            trail.append(comp)
            for comp2 in compdeps[comp]:
                if comp2 not in followed:
                    follow(comp2, trail)
            trail.pop()
            followed.add(comp)
        for comp in self.components():
            if comp not in followed:
                follow(comp, [])
        # Score cycles by occurence of first and last component
        scores = {}
        for cycle in cycles:
            cid = (cycle[0], cycle[-1])
            try:
                scores[cid][0] += 1
            except KeyError:
                scores[cid] = [1, cycle]
            if len(cycle) < len(scores[cid][1]):
                scores[cid][1] = cycle
        cycles = scores.values()
        cycles.sort(key=lambda x: -x[0])
        return [x[1] for x in cycles]
    def __contains__(self, key):
        return key in self._components
    def count_components(self):
        """
        Returns the number of components in this model.
        """
        return len(self._components)
    def count_states(self):
        """
        Returns the number of state variables in this model.
        """
        return len(self._state)
    def create_unique_names(self):
        """
        Create a globally unique name for each Component and Variable.

        Ideally, the global name equals the variable or component's basic name.
        If a name is disputed the following strategy will be used:

        1. For variables, the parent name will be added as a prefix to **all**
           variables claiming the disputed name.
        2. If problems persist, a suffix ``_i`` will be added, where ``i`` is
           the first integer which doesn't result in clashing names.
        """
        # Gather disputed names
        allnames = self._reserved_unames.copy()
        disputed = set()
        for comp in self.components():
            name = comp._name
            if name not in disputed:
                if name in allnames:
                    disputed.add(name)
                else:
                    allnames.add(name)
            for var in comp.variables(deep=True):
                name = var._name
                if name not in disputed:
                    if name in allnames:
                        disputed.add(name)
                    else:
                        allnames.add(name)
        # Set unique names
        for comp in self.components():
            # Set names for component
            name = comp._name
            if name in disputed:
                i = 1
                root = name + '_'
                name = root + str(i)
                while name in allnames:
                    i += 1
                    name = root + str(i)
                allnames.add(name)
            comp._uname = name
            # Set names for variables
            for var in comp.variables(deep=True):
                name = var._name
                if name in disputed:
                    name = var.qname().replace('.', '_')
                    if name in allnames:
                        i = 1
                        root = name + '_'
                        name = root + str(i)
                        while name in allnames:
                            i += 1
                            name = root + str(i)
                    allnames.add(name)
                var._uname = name
    def _create_variable_stream(self, deep, sort):
        def stream(model):
            for c in model.components(sort=sort):
                for v in c.variables(deep=deep, sort=sort):
                    yield v
        return stream(self)
    def eval_state_derivatives(self, state=None, inputs=None,
            precision=myokit.DOUBLE_PRECISION, ignore_errors=False):
        """
        Evaluates and returns the values of all state variable derivatives.
        The values are returned in a list sorted in the same order as the
        state variables.

        If given, the state values given by ``state`` will be used as starting
        point. Here ``state`` can be any object accepted as input by
        :meth:``map_to_state()``.
        
        To set the values of external inputs, a dictionary mapping binding
        labels to values can be passed in as ``inputs``.
        
        To assist in finding the origins of numerical errors, the equations
        can be evaluated using 32 bit floating point. To do this, set
        ``precision=myokit.SINGLE_PRECISION``.
        
        By default, the evaluation routine checks for numerical errors
        (divide-by-zeros, invalid operations and overflows). To run an
        evaluation without error checking, set ``ignore_errors=True``.
        """
        # Apply new state if required
        if state is not None:
            org_state = self.state()
            self.set_state(self.map_to_state(state))
        # Apply values of external inputs if required
        if inputs is not None:
            # Check if values in ``inputs`` are all numbers
            temp = inputs
            inputs = {}
            for label, number in temp.iteritems():
                if label in self._bindings:
                    inputs[label] = float(number)
            del(temp)
            # Store original rhs values, set temporary new ones
            org_inputs = {}
            for label, number in inputs:
                var = self._bindings[label]
                org_inputs[var] = var.rhs()
                var.set_rhs(myokit.Number(number))
        # Get solvable order
        order = self.solvable_order()
        # Evaluate all variables in solvable order
        values = {}
        for group in order.itervalues():
            for eq in group:
                values[eq.lhs] = eq.rhs.eval(values, precision=precision,
                    ignore_errors=ignore_errors)
        # Extract state from evaluated values
        out = [0]*self.count_states()
        for i, v in enumerate(self.states()):
            out[i] = values[v.lhs()]
        # Reset original state
        if state is not None:
            self.set_state(org_state)
        # Reset original values of variables set to external inputs
        if inputs is not None:
            for var, rhs in org_inputs.iteritems():
                var.set_rhs(rhs)
        # Return calculated state
        return out
    def expressions_between(self, start, finish):
        """
        Takes two variables and returns all equations through which ``start``
        influences ``finish``.
        
        """
        # Check input
        if isinstance(start, Variable):
            start = start.lhs()
        if isinstance(finish, Variable):
            finish = finish.lhs()
        if start.var().model() != self:
            raise ValueError('LhsExpression "start" must point to a variable'
                ' in this model.')
        if finish.var().model() != self:
            raise ValueError('LhsExpression "finish" must point to a variable'
                ' in this model.')
        # Find all routes from start to finish
        # Along the way, add each found route to `order` in such a manner that
        # the elements maintain their dependency order.
        order = [start]
        visited = set()
        def merge(trail):
            """ Merges the new found trail with the existing order """
            p = 0
            for t in trail:
                try:
                    # Item in list: Advance cursor
                    p = 1 + order.index(t, p)
                except ValueError:
                    # Item not in list: Add at current position
                    order.insert(p, t)
                    p += 1
        # Start searching backwards from the finish
        def follow(lhs, trail):
            """ Searches backwards """
            if lhs in order:
                return merge(trail)
            if lhs in visited:
                return
            trail.append(lhs)
            for d in lhs.rhs().references():
                if d.is_state_value():
                    continue
                follow(d, trail)
            trail.pop()
            visited.add(lhs)
        # Create ordered list of lhs expressions
        follow(finish, [])
        # Convert `order` to a list of equations
        eq_list = []
        for lhs in reversed(order):
            eq_list.append(Equation(lhs, lhs.rhs()))
        return eq_list
    def expressions_for(self, variable):
        """
        Returns a tuple ``(eqs, args)`` where ``eqs`` is a list of Equation
        objects in solvable order containing the minimal set of equations
        needed to evaluate the given ``variable`` and ``args`` is a list of
        the state variables and bound variables these expressions require as
        input.
        """
        # Get variable, expression
        if not isinstance(variable, ModelPart):
            variable = self.get(variable)
        expression = variable.rhs()
        # Get shallow dependencies of all required equations
        shallow = {}
        arguments = []
        equations = {}
        def add_dep(lhs):
            if lhs in shallow or lhs in arguments:
                return
            var = lhs.var()
            if var.is_state() or var.is_bound():
                arguments.append(lhs)
                return
            rhs = var.rhs()
            dps = rhs.references()
            shallow[lhs] = dps
            equations[lhs] = rhs
            for dep in dps:
                add_dep(dep)
        for lhs in expression.references():
            add_dep(lhs)
        # Filter out dependencies on arguments
        for dps in shallow.itervalues():
            for arg in arguments:
                if arg in dps:
                    dps.remove(arg)
        # Order expressions list of expressions
        eq_list = []
        while len(shallow):
            done = []
            for lhs,dps in shallow.iteritems():
                if len(dps) == 0:
                    eq_list.append(Equation(lhs, equations[lhs]))
                    done.append(lhs)
            if len(done) == 0:
                raise Exception('Failed to solve system of equations.')
            for lhs in done:
                del(shallow[lhs])
                for dps in shallow.itervalues():
                    if lhs in dps:
                        dps.remove(lhs)
        # Add final equation and return
        eq_list.append(Equation(variable.lhs(), variable.rhs()))
        return (eq_list, arguments)
    def format_state(self, state=None, state2=None):
        """
        Converts the given list of floating point numbers to a string where
        each line has the format ``<full_qualified_name> = <float_value>``. If
        no state is given the one returned by :meth:`state` is used.

        An optional second state can be added for display as ``state2``.
        """
        n = len(self._state)
        if state is not None:
            if len(state) != n:
                raise ValueError('Argument ``state`` must be a list of ('
                    + str(n) + ') floating point numbers.')
        else:
            state = self.state()
        if state2 is not None:
            if len(state2) != n:
                raise ValueError('Argument ``state2`` must be a list of ('
                    + str(n) + ') floating point numbers.')
        out = []
        n = max([len(x.qname()) for x in self.states()])
        for k, var in enumerate(self.states()):
            out.append(var.qname() + ' '*(n-len(var.qname()))
                + ' = ' + myokit.strfloat(state[k]))
        if state2 is not None:
            n = max([len(x) for x in out])
            for k, var in enumerate(self.states()):
                out[k] += ' '*(4+n-len(out[k])) + myokit.strfloat(state2[k])
        return '\n'.join(out)
    def format_state_derivs(self, state, deriv):
        """
        Like :meth:`format_state` but displays the derivatives along with
        each state's value.
        """
        n = len(self._state)
        if len(state) != n:
            raise ValueError('Argument ``state`` must be a list of ('
                + str(n) + ') floating point numbers.')
        if len(deriv) != n:
            raise ValueError('Argument ``deriv`` must be a list of ('
                + str(n) + ') floating point numbers.')
        out = []
        n = max([len(x.qname()) for x in self.states()])
        for i, var in enumerate(self.states()):
            s = myokit.strfloat(state[i])
            d = myokit.strfloat(deriv[i])
            out.append(var.qname() + ' '*(n - len(var.qname()))
                 + ' = ' + s + ' '*(24 - len(s)) + '   dot = ' + d)
        return '\n'.join(out)
    def format_warnings(self):
        """
        Formats all warnings generated during the last call to :meth:`validate`
        and returns a string containing the result.
        """
        pre = str(len(self._warnings)) + ' validation warning(s) given'
        if len(self._warnings) == 0:
            return pre
        out = []
        i = 0
        f = ' ({:' + str(1 + int(math.log10(len(self._warnings)))) + '}) '
        for w in self._warnings:
            i += 1
            m = w.message if isinstance(w, myokit.MyokitError) else str(w)
            out.append(f.format(i) + m)
        return pre + ':\n' + '\n'.join(out)
    def get(self, name, class_filter=None):
        """
        Searches for a component or variable with the given ``qname`` and
        returns it.
        
        To return only objects of a certain class, pass it in as
        ``class_filter``.
        """
        if isinstance(name, ModelPart):
            return name
        names = name.split('.')
        x = self
        try:
            for name in names:
                x = x[name]
        except KeyError:
            raise KeyError(str(name))
        if class_filter:
            if not isinstance(x, class_filter):
                raise KeyError(str(name) + ' of class ' + str(class_filter))
            pass
        return x
    def get_function(self, name, nargs):
        """
        Returns the user function with name ``name`` and ``nargs`` arguments.
        """
        return self._user_functions[str(name) + '(' + str(nargs) + ')']
    def __getitem__(self, key):
        return self._components[key]
    def has_component(self, name):  # has_variable() is an inherited method
        """
        Returns ``True`` if this model has a component with the given ``name``.
        """
        return name in self._components
    def has_interdependent_components(self):
        """
        Returns ``True`` if this model contains mutually dependent components.
        That is, if there exists any component A whose variables depend on
        variables from component B, while variables in component B depend on
        variables in component A.
        
        To see the variables causing the interdependence, use
        :meth:`component_cycles()`.
        """
        equations = self.solvable_order()
        remaining = equations['*remaining*']
        return len(remaining) > 0
    def has_warnings(self):
        """
        Returns True if this model has any warnings.
        """
        return len(self._warnings)
    def inits(self):
        """
        Returns an iterator over the ``Equation`` objects defining this model's
        current state.
        """
        def StateDefIterator(model):
            for k, var in enumerate(model._state):
                yield Equation(myokit.Name(var),
                    myokit.Number(self._current_state[k]))
            raise StopIteration
        return StateDefIterator(self)
    def is_valid(self):
        """
        Returns ``True`` if this model is valid, ``False`` if it is invalid and
        ``None`` if the validation status has not been determined with
        :meth:`validate()`.

        Valid models may still have one or more warnings.
        """
        return self._valid
    def label(self, label):
        """
        Returns the variable with the given label. If no variable is labelled
        as ``label`` it returns ``None``.
        """
        try:
            return self._labels[label]
        except KeyError:
            return None
    def labels(self):
        """
        Returns an iterator over all (label : variable) mappings in this model.
        """
        # New dict allows removing labels using this iterator
        return dict(self._labels).iteritems()
    def __len__(self):
        return len(self._components)
    def load_state(self, filename):
        """
        Sets the model state using data from a file formatted in any style
        accepted by :func:`myokit.parse_state`.
        """
        self.set_state(myokit.load_state(filename, self))
    def map_component_dependencies(self, omit_states=True,
            omit_constants=False):
        """
        Scans all equations and creates a map of inter-component dependencies.

        The result is an ordered dictionary with the following structure::

            {
                comp1 : [dep1, dep2, dep3, ...],
                comp2 : [dep1, dep2, dep3, ...],
                ...
            }

        where ``comp1, comp2, ...`` are Components and ``dep1, dep2, ...`` are
        the components they depend upon.

        Only direct dependencies are listed: If ``A`` depends on ``B`` and
        ``B`` depends on ``C`` then the returned value for ``A`` is ``[B]``,
        not ``[B,C]``.

        By default, dependencies on state variables' current values are
        omitted. This behaviour can be changed by setting ``omit_states`` to
        ``False``.

        To omit all dependencies on constants, set ``omit_constants`` to
        ``True``.
        """
        # Map shallow dependencies
        shallow = self.map_shallow_dependencies(omit_states=omit_states,
            omit_constants=omit_constants)
        # Create output structure
        deps = {}
        for comp in self.components():
            deps[comp] = set()
        # Gather dependencies per component
        for lhs, dps in shallow.iteritems():
            c1 = lhs.var().parent(Component)
            for dep in dps:
                c2 = dep.var().parent(Component)
                if c2 != c1:
                    deps[c1].add(c2)
        # Convert sets to lists
        for comp, dps in deps.iteritems():
            deps[comp] = list(dps)
        # Return
        return deps
    def map_component_io(self, omit_states=False, omit_derivatives=False,
            omit_constants=False):
        """
        Scans all equations and creates a list of input and output variables
        for each component.

        *Input variables* are taken to be any foreign variables the component
        needs to perform its calculations, plus the current values of its own
        state variables if it needs them.

        *Output variables* are taken to be any values calculated by this
        component but used outside it. This includes the derivatives of the
        state variables it calculates.

        The output can be customized using the following arguments:
        
        ``include_states``
            Determines whether or not state variables (including the
            component's own ones) should be present in the input lists.
        ``include_derivatives``
            Determines if state variable derivatives should appear in the lists
            of inputs and outputs.
        ``include_constants``
            Determines if constants should appear in the lists of inputs.

        The result is a tuple containing two ``OrderedDict`` objects, each of
        the following structure::

            {
                comp1: [lhs1, lhs2, ...],
                comp2: [lhs1, lhs2, ...],
                ...
            }

        The first dict contains the inputs to every component, the second dict
        contains every components output values. The values in the lists are
        :class:``LhsExpression`` objects referring to either a variable or its
        derivative.
        """
        # Map shallow dependencies
        shallow = self.map_shallow_dependencies(omit_states=omit_states,
            omit_constants=omit_constants)
        # Create output structure
        di = {}  # Inputs to each component (dependencies)
        do = {}  # Outputs from each component
        for comp in self.components():
            di[comp] = set()
            do[comp] = set()
        # Add own derivatives, even if not explicitely used
        if not omit_derivatives:
            for var in self.states():
                do[var.parent(Component)].add(var.lhs())
        # Add inputs and outputs
        for user, deps in shallow.iteritems():
            c_user = user.var().parent(Component)
            for usee in deps:
                c_usee = usee.var().parent(Component)
                is_deriv = usee.is_derivative()
                is_state = (not is_deriv) and usee.var().is_state()
                # States should be inputs, regardless of their component
                # All others? Check parents
                if is_state:
                    if not omit_states:
                        di[c_user].add(usee) # States are never outputs
                elif c_user != c_usee and not (is_deriv and omit_derivatives):
                    di[c_user].add(usee)
                    do[c_usee].add(usee)
        # Convert sets to lists
        for comp in self.components():
            di[comp] = list(di[comp])
            do[comp] = list(do[comp])
        # Return
        return (di, do)
    def map_deep_dependencies(self, collapse=False, omit_states=True,
        filter_encompassed=False):
        """
        Scans the list of equations stored in this model and creates a map
        relating each equation's left hand side to a list of other
        :class:`LhsExpressions <LhsExpression>` it depends on, either directly
        or through an intermediate variable.

        The method :func:`map_shallow_dependencies` performs a similar check,
        but only returns *direct* dependencies. This method expands the full
        dependency tree.

        The result is an ``OrderedDict`` with the following structure::

          {
            lhs1 : [dep1, dep2, dep3, ...],
            ...
          }

        where ``lhs1`` is a :class:`LhsExpression` and ``dep1``, ``dep2`` and
        ``dep3`` are the :class:`LhsExpression` objects it depends upon.

        If the system is not solvable: that is, if it contains cyclical
        references, a :class:`myokit.IntegrityError` is raised.

        If the optional parameter ``collapse`` is set to ``True``
        nested variables will not be listed separatly. Instead, their
        dependencies will be added to the dependency lists of their parents.

        By default, dependencies on state variables' current values are
        omitted. This behaviour can be changed by setting ``omit_states`` to
        ``False``.

        In case of a dependency such as::

            a = f(b)
            b = g(c)

        The set returned for ``a`` will include ``b`` and ``c``. So while ``a``
        here depends on both ``b`` and ``c``, ``b``'s sole dependency on ``c``
        means ``a`` can be calculated if just ``c`` is known. To filter out the
        dependency on ``b``, set ``filter_encompassed`` to ``True``. Note that
        this will also filter out all dependencies on constants, since a
        constant's dependencies (an empty set) can be said to be included in
        all other sets.
        """
        # Get map of shallow dependencies
        shallow = self.map_shallow_dependencies(omit_states=omit_states)
        # Create map of deep dependencies
        deep = {}
        # Store set of lhs's found by following equations
        equations = dict(zip(shallow.keys(),
            [x.rhs() for x in shallow.keys()]))
        # Add state variables
        if not omit_states:
            for var in self.variables():
                if var.is_state():
                    equations[myokit.Name(var)] = myokit.Number(0)
        unused = set(equations.keys())
        # Tree searching function
        def follow(lhs, deps=None, trail=None):
            if deps is None:
                deps = set()
            if trail is None:
                trail = []
            # Followed already? Return stored result
            if lhs in deep:
                for dep in deep[lhs]:
                    deps.add(dep)
                return deps
            # Cycle check
            if lhs in trail:
                trail = trail[trail.index(lhs):]
                trail.append(lhs)
                raise myokit.CyclicalDependencyError(trail)
            # Not in shallow? Error in dict
            if lhs not in shallow:
                raise KeyError('Variable ' + str(lhs._value) + ' not found in'
                    ' dict of shallow dependencies')
            # Add this var as a dependency
            deps.add(lhs)
            # Follow child dependencies
            trail2 = trail + [lhs]
            for kid in shallow[lhs]:
                follow(kid, deps, trail2)
                if kid in unused:
                    unused.remove(kid)
            return deps
        # Follow every lhs (checks for cycles)
        for lhs in equations:
            deep[lhs] = follow(lhs)
        for lhs in equations:
            deep[lhs].remove(lhs)
        # Collapse nesting
        if collapse:
            nested = []
            for x, y in deep.iteritems():
                var = x.var()
                if var.is_nested():
                    nested.append(x)
                    while isinstance(var.parent(), Variable):
                        var = var.parent()
                    var = deep[var.lhs()]
                    for rhs in y:
                        var.add(rhs)
            for x in nested:
                for y in deep.itervalues():
                    if x in y:
                        y.remove(x)
            for x in nested:
                del deep[x]
        # Filter encompassed variables
        if filter_encompassed:
            for x, xdeps in deep.iteritems():
                tofilter = []
                for y in xdeps:
                    if y.is_state_value():
                        # Don't filter out state values
                        continue
                    ydeps = deep[y]
                    if ydeps <= xdeps:  # <= means is_subset or equal
                        tofilter.append(y)
                for y in tofilter:
                    xdeps.remove(y)
        # Return
        return deep
    def map_shallow_dependencies(self, collapse=False, omit_states=True,
            omit_constants=False):
        """
        Scans the list of equations stored in this model and creates a map
        relating each equation's left hand side to a list of other
        :class:`LhsExpressions <LhsExpression>` it depends on directly.

        In contrast to the method :func:`map_deep_dependencies`, that performs
        a deep search for all nested dependencies, this method only returns
        *direct* dependencies. I.E. LhsExpressions named specifically in the
        equation's right hand side.

        The result is an ``OrderedDict`` with the following structure::

          {
            lhs1 : [dep1, dep2, dep3, ...],
            ...
          }

        where ``lhs1`` is a :class:`LhsExpression` and ``dep1``, ``dep2`` and
        ``dep3`` are the :class:`LhsExpression` objects it depends upon.

        In special cases, the obtained result can differ from the mathematical
        concept of a dependency: if ``x = 0 * y`` the function will return a
        dependency of ``x`` on ``y``. Similarly, for ``x = if(cond, a, b)`` the
        function will report a dependency of ``x`` on ``cond``, ``a`` and
        ``b``.

        If the optional parameter ``collapse`` is set to ``True``
        nested variables will not be listed separatly. Instead, their
        dependencies will be added to the dependency lists of their parents.

        By default, dependencies on state variables' current values are
        omitted. This behaviour can be changed by setting ``omit_states`` to
        ``False``.
        """
        # Find dependencies for every stored equation
        out = {}
        inc = not omit_states
        const = False if omit_constants else None
        for eq in self.equations(deep=True, const=const):
            out[eq.lhs] = deps = set()
            # Bound variable has zero dependencies
            if eq.lhs.var().is_bound():
                continue
            # Find references
            for dep in eq.rhs.references():
                if omit_states and dep.is_state_value():
                    continue
                elif omit_constants and dep.var().is_constant():
                    continue
                else:
                    deps.add(dep)
        # Collapse nested variables
        if collapse:
            nested = []
            for x, y in out.iteritems():
                var = x.var()
                if var.is_nested():
                    nested.append(x)
                    while isinstance(var.parent(), Variable):
                        var = var.parent()
                    var = out[var.lhs()]
                    for rhs in y:
                        var.add(rhs)
            for x in nested:
                for y in out.itervalues():
                    if x in y:
                        y.remove(x)
            for x in nested:
                del out[x]
        # Add empty mappings for state variables
        if inc:
            # State vars are never nested, deep search not needed
            for var in self.states():
                out[myokit.Name(var)] = set()
        # Return
        return out
    def map_to_state(self, state):
        """
        This utility function accepts a number of different input types and
        returns a list of floats in the same order as the model's state
        variables.

        The following types of input are meaningfully handled:

         - A dictionary mapping state variable names or objects to floats
         - A dictionary mapping state variable names or objects to lists. The
           last value in every list will be used. This allows the dictionary
           returned by :meth:`myokit.Simulation.run()` to be used directly.
         - A formatted string declaring each variable's value using the syntax
           ``component.var_name = 1.234`` or simply containing a comma or space
           delimited list of floats.
         - A list of scalars. In this case, the list length will be
           checked and all values converted to float. No re-ordering takes
           place.
           
        """
        n = self.count_states()
        if type(state) in [str, unicode]:
            # String given. Parse into name:float map or list
            state = myokit.parse_state(state)
        if isinstance(state, dict):
            # Dictionary (sub)types. Parse into list of floats
            ini = [0] * n
            for k, v in enumerate(self.states()):
                if v in state:
                    value = state[v]
                elif v.qname() in state:
                    value = state[v.qname()]
                else:
                    raise ValueError('Missing state variable: '
                        + v.qname())
                # List or similar? Then use last value
                try:
                    ini[k] = float(value[-1])
                except TypeError:
                    ini[k] = float(value)
            state = ini
        # From this point on, assume list
        if len(state) != n:
            raise ValueError('Wrong number of list entries, expecting ('
                + str(n) + ') values, got (' + str(len(state)) + ').')
        # Convert all to float, create new list, return
        return [float(x) for x in state]
    def merge_interdependent_components(self):
        """
        Checks if the model contains components that each depend on the other.
        If so, these components will be merged together into a new component
        called "remaining".
        """
        equations = self.solvable_order()
        remaining = equations['*remaining*']
        if len(remaining) < 1:
            # Model is already okay
            return
        # Get name for new component
        comp = 'remaining'
        if comp in self._components:
            comp_root = comp + '_'
            i = 1
            while comp in self._components:
                i += 1
                comp = comp_root + str(i)
        # Create new component (resets validation status)
        comp = self.add_component(comp)
        # Move variables to new component
        for eq in remaining:
            var = eq.lhs.var()
            var.parent().move_variable(var, comp, var.uname())
        # Done!
    def name(self):
        """
        Returns a name for this model: If the meta data property ``name`` is
        set this is used, if not the string "Unnamed model" is returned.
        """
        try:
            name = self.meta['name']
        except KeyError:
            return 'Unnamed model'
        return str(name)
    def prepare_bindings(self, labels):
        """
        Takes a mapping of binding labels to internal references as input and
        returns a mapping of variables to internal references. All variables
        appearing in the map will have their right hand side set to zero. All
        bindings not mapped to any internal reference will be deleted.

        The argument ``mapping`` should take the form::

            labels = {
                'binding_label_1' : internal_name_1,
                'binding_label_2' : internal_name_2,
                ...
                }

        The returned dictionary will have the form::

            variables = {
                variable_x : internal_name_1,
                variable_y : internal_name_2,
                ...
                }

        Unsupported bindings (i.e. bindings not appearing in ``labels``) will
        be ignored.
        """
        unused = []
        variables = {}
        for label, var in self._bindings.iteritems():
            try:
                variables[var] = labels[label]
            except KeyError:
                unused.append(var)
                continue
            var.set_rhs(0)
        for var in unused:
            var.set_binding(None)
        return variables
    def _register_binding(self, label, variable=None):
        """
        Used by variables to inform the model of the addition or removal of a
        binding label to/from a variable.
        """
        if variable is None:
            # Remove binding
            del(self._bindings[label])
        else:
            # Check for existing binding
            if label in self._bindings:
                raise myokit.InvalidBindingError('Duplicate binding: <'
                    + str(self._bindings[label].qname()) + '> is already bound'
                    ' to "' + str(label) + '".')
            # Check for existing label
            if label in self._labels:
                raise myokit.InvalidBindingError('Duplicate binding: Binding "'
                    + str(label) + '" is already in use as a label for <'
                    + str(self._labels[label].qname()) + '>.')
            # Add binding
            self._bindings[label] = variable
    def _register_label(self, label, variable=None):
        """
        Used by variables to inform the model of the addition or removal of a
        label to/from a variable.
        """
        if variable is None:
            # Remove label
            del(self._labels[label])
        else:
            # Check for existing label
            if label in self._labels:
                raise myokit.InvalidLabelError('Duplicate label: "'
                    + str(label) + '" is already in use as a label for <'
                    + str(self._labels[label].qname()) + '>.')
            # Check for existing binding
            if label in self._bindings:
                raise myokit.InvalidLabelError('Duplicate label: Label "'
                    + str(label) + '" is already in use as a binding for <'
                    + str(self._bindings[label].qname()) + '>.')
            # Add label
            self._labels[label] = variable
    def reorder_state(self, order):
        """
        Changes the order of this model's state variables. The argument
        ``order`` must be a list of state variables or their qnames.

        This method does not affect the model's validation status.
        """
        n = len(self._state)
        if len(order) != n:
            raise ValueError('The given list must contain the same number of'
                ' entries as there are state variables in this model.')
        state = []
        current = []
        for v in order:
            v = self.get(v)
            if not v.is_state():
                raise ValueError('The entries of ``order`` must all be state'
                    ' variables or state variable qnames.')
            if v in state:
                raise ValueError('Duplicate entry in order specification: "'
                    + str(v.qname()) + '".')
            state.append(v)
            current.append(self._current_state[v._indice])
        self._state = state
        self._current_state = current
        for k, v in enumerate(state):
            v._indice = k
    def remove_component(self, component):
        """
        Removes a component from the model.

        This will reset the model's validation status.
        """
        try:
            # Get component object
            if not isinstance(component, Component):
                component = self._components[component]
            # Tell component it's being deleted
            component._delete()
            # Delete component from list
            del(self._components[component.qname()])
        finally:
            self._valid = None
    def __repr__(self):
        """
        Returns a representation of this model in the form
        ``<Model(model_name)>``.
        """
        return '<Model(' + self.name() + ')>'
    def reserve_unique_names(self, *unames):
        """
        Reserves one or more names that won't be used in unique names (unames).

        This function can be used to add keywords to a model before exporting.
        After reserving one or more keywords, use :meth:`set_unique_names` to
        reset the model's unames.

        Adding new names does _not_ clear the previously reserved names.
        """
        for name in unames:
            self._reserved_unames.add(name)
    def _reset_indices(self):
        """
        Resets the indices of this model's state variables.
        """
        for k, v in enumerate(self._state):
            v._indice = k
    def _reset_validation(self):
        """
        Will reset the model's validation status to not validated.
        """
        self._valid = None
    def save_state(self, filename):
        """
        Saves the model state to a file.
        """
        return myokit.save_state(filename, self.state(), self)
    def set_name(self, name=None):
        """
        Changes the value of the meta-property "name".
        """
        if name is None:
            try:
                del(self._meta['name'])
            except KeyError:
                pass
        else:
            self.meta['name'] = str(name)
    def set_state(self, state):
        """
        Changes this model's state. Accepts any type of input handled by
        :meth:`map_to_state`.
        """
        self._current_state = self.map_to_state(state)
    def set_value(self, qname, value):
        """
        Changes a variable's defining expression to the given number or
        :class:`myokit.Expression`.

        For state variables, this updates the expression for their derivative.
        For all other variables, the expression for their value is updated.

        This will reset the model's validation status.
        """
        self.get(qname).set_rhs(value)
    def show_evaluation_of(self, var):
        """
        Returns a string representing the evaluation of a single variable.

        The variable's equation and value are displayed, along with the value
        and formula of any nested variables and the values of all dependencies.
        """
        def format_float(number):
            s = str(number)
            if len(s) < 10:
                return s
            return '%0.17e' % number
        # Add basic info
        spacer = '-'*60
        var, out = self._var_info(var, spacer)
        # Add initial value
        rhs = var.rhs()
        if var.is_state():
            out.append('Initial value = ' + str(var.state_value()))
            out.append(spacer)
        varname = var.lhs().code()
        # Add references
        deps = rhs.references()
        if deps:
            n = max([len(x.code()) for x in deps])
            for dep in deps:
                out.append(dep.code() + ' '*(n - len(dep.code()))
                    + ' = ' + format_float(dep.eval()))
            out.append(spacer)
        # Add nested variables
        if len(var) > 0:
            n = max(len(x.name()) for x in var.variables(deep=True))
            for tmp in var.variables(deep=True):
                out.append(tmp.name() + ' '*(n - len(tmp.name()))
                    + ' = ' + tmp.rhs().code())
                out.append(' ' * n +' = '+ format_float(tmp.rhs().eval()))
            out.append(spacer)
        # Add final value
        out.append(varname + ' = ' + rhs.code())
        out.append(' '*len(varname) + ' = ' + format_float(rhs.eval()))
        return '\n'.join(out)
    def show_expressions_for(self, var):
        """
        Returns a string containing the expressions needed to evaluate the
        given variable from the state variables.
        """
        spacer = '-'*60
        var, out = self._var_info(var, spacer)
        eqs, args = self.expressions_for(var)
        if args:
            out.append(str(var) + ' is a function of:')
            for arg in args:
                out.append('  ' + str(arg))
            out.append(spacer)
        out.append('Expressions for ' + str(var) + ':')
        for eq in eqs:
            out.append('  ' + str(eq))
        return '\n'.join(out)
    def show_line(self, var):
        """
        Returns a string containing the type of variable this is and the line
        it was defined on.
        """
        var, out = self._var_info(var)
        if var._token:
            out.append('Defined on line ' + str(var._token[2]))
        return '\n'.join(out)
    def solvable_order(self):
        """
        Returns all equations in a solvable order. The resulting output has the
        following structure::

          OrderedDict {
            'comp1' : EquationList([eq1, eq2, eq3, ...]),
            'comp2' : EquationList([eq1, eq2, eq3, ...]),
            ...,
            '*remaining*' : EquationList([eq1, eq2, eq3])
          }

        The ``OrderedDict`` contains each component's name as a key and maps it
        to a list of equations that can be solved for this component, stored in
        an :class:``EquationList`` object which extends the :class:`list` type
        with a number of special iterators.

        The order of the components is such that the components with the fewest
        dependencies are listed first, in an effort to ensure as many equations
        as possible can be solved on a per-component basis.

        The final entry in the ``OrderedDict`` is a list of all remaining
        equations that could not be solved on a per-component basis. For models
        that contain fully separable components (that is, if the model contains
        only components that depend on each other non-cyclically) this list
        will be empty.
        """
        #TODO: Cache this!
        # Get components in solvable order
        solvable_comps = []
        cdeps = self.map_component_dependencies()
        while True:
            new = []
            for comp, deps in cdeps.iteritems():
                if comp not in solvable_comps and len(deps) == 0:
                    solvable_comps.append(comp)
                    new.append(comp)
            if len(new) == 0:
                break
            for comp in new:
                for deps in cdeps.itervalues():
                    if comp in deps:
                        deps.remove(comp)
        # Create component output dict and todo list
        # Important: both should be created simultaneously to ensure they're
        #  in the same order!
        out = OrderedDict()     # Final output dict (comp.qname:EquationList)
        todo = OrderedDict()    # Todo dict (comp:dict)
        for comp in solvable_comps:
            out[comp.name()] = EquationList()
            todo[comp] = {}
        for comp in self.components():
            out[comp.name()] = EquationList()
            todo[comp] = {}
        # Populate component todo lists
        for lhs, rhs in self.equations(deep=True):
            comp = lhs.var().parent(Component)
            todo[comp][lhs] = Equation(lhs, rhs)
        # Get map of shallow dependencies, use as to-do list
        deps = self.map_shallow_dependencies()
        # Nested variable grouping: nested variables aren't added to the list
        # directly but stored until their parent variable is added. This does
        # not interfere with the ordering since only parent and sibling
        # variables are allowed to depend on them.
        nested = OrderedDict() # Nested variable equations (var:eq)
        def expand(par):
            expd = []
            for var,eq in nested.iteritems():
                if var._parent == par:
                    expand(var)
                    eq_list.append(eq)
                    expd.append(var)
            for var in expd:
                del(nested[var])
        def addeq(lhs, eq, done):
            """
            Adds eq to eq_list, removes it from done
            """
            var = lhs.var()
            if var.is_nested():
                nested[var] = eq
                done.append(lhs)
            else:
                expand(var)
                eq_list.append(eq)
                done.append(lhs)
        # Now handle each component, as far as we get
        for comp, eqs in todo.iteritems():
            eq_list = out[comp.name()]
            while True:
                done = []
                for lhs, eq in eqs.iteritems():
                    if len(deps[lhs]) == 0:
                        addeq(lhs, eq, done)
                if len(done) == 0:
                    break
                for lhs in done:
                    # Remove from todo list, remove all deps on done lhs's
                    del eqs[lhs]
                    del deps[lhs]
                    for dps in deps.itervalues():
                        if lhs in dps:
                            dps.remove(lhs)
        # Get remaining equations
        todd = todo
        todo = {}
        for comp, eqs in todd.iteritems():
            for lhs, eq in eqs.iteritems():
                todo[lhs] = eq
        del(todd) # Bye todd!
        # Add remaining equations in one big chunk
        out['*remaining*'] = eq_list = EquationList()
        while True:
            done = []
            for lhs, eq in todo.iteritems():
                if len(deps[lhs]) == 0:
                    addeq(lhs, eq, done)
            if len(done) == 0:
                break
            for lhs in done:
                del todo[lhs]
                del deps[lhs]
                for dps in deps.itervalues():
                    if lhs in dps:
                        dps.remove(lhs)
        if len(todo):
            raise Exception('Equation ordering failed')
        # Return
        return out
    def solvable_subset(self, *args):
        """
        Returns all equations dependent on one or more :class:`LhsExpression`
        objects (the "roots") in a solvable order. The resulting equations are
        stored in an :class:`EquationList`.
        
        The returned equations can be used to recalculate the model
        expressions, given fixed values for the roots and leaving all other
        variable unaffected.
        
        The input arguments can be given as :class:`LhsExpression` objects or
        string names of variables, in this case, the Lhs returned by
        :meth:`Variable.lhs()` will be used.        
        """
        # 1. Get set of root lhs objects
        msg = 'All input arguments to solvable_subset must be' \
              ' LhsExpression objects or string names of variables'
        roots = set()
        for lhs in args:
            if type(lhs) in (str, unicode):
                lhs = self.get(lhs)
                if not isinstance(lhs, myokit.Variable):
                    raise ValueError(msg)
                lhs = lhs.lhs()
            if not isinstance(lhs, myokit.LhsExpression):
                raise ValueError(msg)
            roots.add(lhs)
        # 2. Get subtree starting at those roots
        def add_to_subtree(var, tree):
            lhs = var.lhs()
            if lhs in subtree:
                return
            subtree.add(lhs)
            for kid in var.refs_by(): # Can never be state value
                add_to_subtree(kid, tree)
        subtree = set()
        for lhs in roots:
            subtree.add(lhs)
            if lhs.is_state_value(): # Roots may contain state values
                for var in lhs.var().refs_by(state_refs=True):
                    add_to_subtree(var, subtree)
                # Iterator doesn't return self rep when dot(x) = f(x)
                # So add dot(x) manually if it references x.
                var = lhs.var()
                if lhs in var.rhs().references():
                    add_to_subtree(var, subtree)
            else:
                for var in lhs.var().refs_by():
                    add_to_subtree(var, subtree)
        # 3. Get map of references within subtree
        deps = {}
        for lhs in subtree:
            if lhs in roots:
                continue
            dps = set()
            for ref in lhs.rhs().references():
                if ref in subtree and not ref in roots:
                    dps.add(ref)
            deps[lhs] = dps
        # 4. Get solvable list of equations
        eqs = EquationList()
        while deps:
            todo = set()
            for lhs, dps in deps.iteritems():
                if not dps:
                    todo.add(lhs)
            if not todo:
                raise Exception('Equation ordering failed.')
            for lhs in todo:
                del(deps[lhs])
                eqs.append(Equation(lhs, lhs.rhs()))
            for lhs, dps in deps.iteritems():
                deps[lhs] -= todo
        return eqs
    def state(self):
        """
        Returns the current state of the model as a list of floating point
        numbers.
        """
        return list(self._current_state)
    def states(self):
        """
        Returns an iterator over this model's state :class:`variable
        <myokit.Variable>` objects.
        """
        return iter(self._state)
    def suggest_variable(self, name):
        """
        Returns a tuple ``(found, suggested, msg)``.

        If the requested variable is found, only the ``found`` part of the
        tuple is set. If not, the second and third argument are set. Here
        ``suggested`` will take the form of a suggested variable with a similar
        name, while ``msg`` will be a suggested error message.
        """
        # Return variable or set error message
        name = str(name)
        if '.' not in name:
            # Attempt to Suggest component
            for c in self.components():
                for v in c.variables():
                    if v.name() == name:
                        return (None, v, 'No component specified for: <'
                            + name + '>. Did you mean <'+v.qname()+'> ?')
            # No alternative component found
            msg = 'Unknown variable: <' + name + '>.'
        else:
            par, var = name.split('.', 1)
            if par not in self._components:
                msg = 'Unknown component: <' + par + '>.'
            else:
                try:
                    names = var.split('.')
                    var = self._components[par]
                    for nm in names:
                        var = var[nm]
                    return (var, None, None)
                except KeyError:
                    msg = 'Unknown variable: <' + name + '>.'
        # Suggest closest match
        qname = name
        d = 1 + name.rfind('.')
        if d > 0: name = name[d:]
        name_low = name.lower()
        qname_low = qname.lower()
        mn = 99999
        sg = None
        for v in self.variables(deep=True):
            n1 = v.name()
            n2 = v.qname()
            d = min(myokit.lvsd(name, n1),
                    myokit.lvsd(qname, n2),
                    myokit.lvsd(name_low, n1.lower()),
                    myokit.lvsd(qname_low, n2.lower()))
            if d < mn:
                mn = d
                sg = v
        if sg is not None:
            msg += ' Did you mean "' + sg.qname() + '"?'
            if sg.qname().lower() == name.lower():
                msg += ' (Case mismatch)'
            return (None, sg, msg)
        return (None, None, msg)
    def time(self):
        """
        Returns this model's time variable.

        The time variable is identified by it's binding to the external source
        "time". For a valid model, this method always returns a unique
        variable. If no time variable has been declared ``None`` is returned.
        """
        try:
            return self._bindings['time']
        except KeyError:
            return None
    def time_unit(self):
        """
        Returns the units used by this model's time variable.
        
        If no variable has been declared or the variable doesn't specify a unit
        ``myokit.units.dimensionless`` is returned.
        """
        try:
            time = self._bindings['time']
        except KeyError:
            return myokit.units.dimensionless
        unit = time.unit()
        return unit if unit is not None else myokit.units.dimensionless
    def user_functions(self):
        """
        Returns a dictionary mapping this model's user function names to their
        :class:`Expression` objects.
        """
        # Names and expressions are immutable, so ok to return
        return dict(self._user_functions)
    def validate(self, fix_crudely=False):
        """
        Attempts to check model validity, raises errors if it isn't. Any
        warnings previously set will be erased.

        An attempt to crudely "fix" the model can be made by setting
        ``fix_crudely`` to True.
        """
        # Reset warnings
        self._warnings = []
        # Check time variable
        time = self.time()
        if time is None:
            self._valid = False
            raise myokit.MissingTimeVariableError()
        if time.binding() != 'time':
            self._valid = False
            raise myokit.IntegrityError('Invalid time variable set. Time'
                ' variable must be bound to external value "time".')
        # Validation of components, variables
        for c in self.components():
            if c._parent != self:
                msg = 'Component parent doesn\'t match with enclosing model <'
                msg += c.name() + '>.'
                if not fix_crudely:
                    self._valid = False
                    raise myokit.IntegrityError(msg)
                else:
                    self._warn(msg + ' Attempting fix.')
                    c._parent = self
            # Deep validation
            c.validate(fix_crudely)
        # Test component mapping
        for n, c in self._components.iteritems():
            if n != c.qname():
                self._valid = False
                raise myokit.IntegrityError('Component called <'
                    + c.qname() + '> found at index <' + n + '>.')
        # Test current state values
        n = len(self._state)
        if n != len(self._current_state):
            self._valid = False
            raise myokit.IntegrityError('Current state values list must have'
                ' same size as state variables list.')
        # Find cycles, warn of unused variables
        self._validate_solvability(fix_crudely)
        # Create globally unique names
        self.create_unique_names()
        # Return
        self._valid = True
    def _validate_solvability(self, remove_unused=False):
        """
        Tests if all values are used and checks for cycles.

        The method used is straightforward:

          1. All variables are marked as "unused".
          2. For every unused derivative, the dependencies are followed
             recursively, keeping track of the current trail.
          3. Visited variables are marked as used.
          4. If a visited variable occurs in the current trail, a cyclical
             dependency is found and an exception is raised.
          5. When all state variables have been visited, any variables not
             marked as used are unused. A warning is generated for each unused
             variable. Bound and labelled variables are always counted as
             "used".

        If the optional argument ``remove_unused`` is set to true, unused
        variables are removed from the model.
        """
        # Mark all variables as unused
        for var in self.variables(deep=True):
            var._used = False
        # Define "following" function
        def follow(lhs, var, trail, mark_used=True):
            if lhs in trail:
                # Catch cycles
                i = trail.index(lhs)
                c = trail[i:]
                c.append(lhs)
                raise myokit.CyclicalDependencyError(c)
            if not var._used:
                trail.append(lhs)
                for d in var.rhs().references():
                    v = d.var()
                    if not v.is_state() and d.is_derivative:
                        follow(d, v, trail, mark_used)
                trail.pop()
                if mark_used:
                    var._used = True
        # Follow all state variables (unless already visited), all bound
        # variables and all used variables.
        used = [x for x in self._state]
        used += [x for x in self._bindings.itervalues()]
        used += [x for x in self._labels.itervalues()]
        # Check for cycles
        trail = []
        # Despite its "slow" is-in check, using a list for this purpose is
        # faster than using an OrderedDict. Tested this. It's probably because
        # the trail is usually quite short and hashing is slow.
        for var in used:
            lhs = var.lhs()
            trail = []
            follow(lhs, var, trail)
        # Check for unused variables
        to_remove = []
        for var in self.variables(deep=True):
            if not var._used:
                # Everything else counts as unused
                if remove_unused:
                    # Option 1, delete unused variables.
                    to_remove.append(var)
                else:
                    # Option 2, warn about it.
                    self._warn(myokit.UnusedVariableError(var))
                    # In this case, the unused variables should also be checked
                    # for cyclical references!
                    lhs = var.lhs()
                    trail = []
                    follow(lhs, var, trail, False)
        # Remove unused variables (if enabled)
        for var in to_remove:
            # Unused variables may still depend on each other
            var.set_rhs(0)
        for var in to_remove:
            self._warn('Removing unused variable: <' + var.qname() + '>.')
            # If a nested set of variables is in to_remove, the parent may be
            # deleted before the kids, which will trigger a delete of the kids.
            # To avoid trying to delete things twice, check if the parent still
            # exists.
            if var._parent:
                var._parent.remove_variable(var, recursive=True)
        # Remove _used attributes
        for var in self.variables(deep=True):
            del (var._used)
    def value(self, qname):
        """
        Returns the value of a variable.
        """
        return self.get(qname).rhs().eval()
    def variable_at_text_position(self, line, char):
        """
        Searches for a (token, object) tuple matching the given text position.
        Not all tokens are stored, so this function may return None.

        Both ``line`` and ``char`` should be given as integers, with the first
        line and first char having index ``0``.
        """
        if not line in self._tokens:
            return None
            # This will cause problems for multi-line strings...
            # (But at the moment, these don't register tokens anyway)
        tokens = self._tokens[line]
        t = None
        for c in sorted(tokens, reverse=True):
            if c <= char:
                t = tokens[c]
                break
        if t is not None:
            # Test if position is in token
            if char >= t[0][3] + len(t[0][1]):
                t = None
        return t
    def _var_info(self, var, spacer=None):
        """
        Gathers basic information about a variable. Returns a tuple (var, out)
        where var is the variable (this function will also accept strings and
        uses suggest_variable to find a match) and out is a list of strings
        where each entry is a line of the description.
        """
        out = []
        if not isinstance(var, ModelPart):
            var = self.suggest_variable(var)
            if var[0] is None:
                if var[1] is None:
                    raise Exception(var[2])
                var = var[1]
                out.append('Variable not found, assuming <'
                    + var.qname() + '>.')
            else:
                var = var[0]
        if var.is_state():
            kind = 'State variable'
        elif var.is_constant():
            if var.rhs().is_literal():
                kind = 'Literal constant'
            else:
                kind = 'Calculated constant'
        else:
            kind = 'Intermediary variable'
        out.append('Showing: ' + var.qname() + '  ('+kind+')')
        if spacer is not None:
            out.append(spacer)
        desc = var.meta.get('desc')
        if desc is not None:
            if desc is not None:
                out.append('desc: ' + str(desc))
            if spacer is not None:
                out.append(spacer)
        return (var, out)
    def _warn(self, msg):
        """
        Appends a warning to this model's list.
        """
        self._warnings.append(msg)
    def warnings(self):
        """
        Returns a list containing the warnings generated in this model.
        """
        return self._warnings
class Component(VarOwner):
    """
    A Component acts as a container of :class:`variables <Variable>`.

    Variables can be accessed using the ``comp['var_name']`` syntax or through
    the iterator methods.
    
    Meta-data properties can be accessed via the property ``meta``, for example
    ``model.meta['key']= 'value'``.
    """
    def __init__(self, model, name):
        super(Component, self).__init__(model, name)
        self._alias_map = {} # Maps variable names to other variables names
    def _clone1(self, model):
        """
        Performs the first part of component cloning: Clones this component's
        variables into the given :class:`Model` given as ``model``, but
        doesn't set any references (such as in aliases or expressions.)
        """
        component = model.add_component(self.name())
        self._clone_modelpart_data(component)
        for v in self.variables():
            v._clone1(component)
    def _clone2(self, component, lhsmap):
        """
        Performs the second part of component cloning: Iterates over the
        variables in the new, incomplete :class:`Component` given as
        ``component`` and adds their expressions. Sets aliases etc.
        
        The argument ``lhsmap`` should be a dictionary mapping old
        :class:`LhsExpression` objects their equivalents in the new model.
        """
        model = component.model()
        # Clone alias map
        for k, v in self._alias_map.iteritems():
            component.add_alias(k, model.get(v.qname()))
        # Clone equations
        for v in self.variables():
            v._clone2(component[v.name()], lhsmap)
    def add_alias(self, name, variable):
        """
        Adds an alias to this component. The alias ``name`` will be refer to
        the :class:`Variable` object given as ``variable``.
        
        Aliases can only be created for variables of other components.
        """
        name = check_name(name)
        if not self.can_add_variable(name):
            raise myokit.DuplicateName('The name <' + str(name) + '> is'
                ' already in use within this scope.')
        par = variable.parent()
        if type(par) != Component:
            raise myokit.IllegalAliasError('Aliases can only be created for'
                ' variables whose parent is a Component.')
        if par == self:
            raise myokit.IllegalAliasError('Cannot create an alias for'
                ' variables in the same component.')
        self._alias_map[name] = variable
    def _delete(self):
        """
        Tells this component it's being deleted.
        """
        # Find all components that reference this one
        reffers = set()
        for var in self.variables(): # No need for deep!
            refs = var._refs_by.union(var._srefs_by)
            for ref in refs:
                c = ref.parent(Component)
                if c != self:
                    reffers.add(c)
        if reffers:
            raise myokit.IntegrityError('Can not delete component <'
                + self.qname() + '> it is used by components '
                + ' and '.join(['<' + c.qname() + '>' for c in reffers]))
        # No problem? Then delete all variables from component
        for var in self.variables():
            var._delete(recursive=True, whole_component=True)
        # Delete links to parent
        super(Component, self)._delete()
    def alias(self, name):
        """
        Returns the :class:`Variable` referred to using the alias ``name``.
        """
        return self._alias_map[name]
    def alias_for(self, variable):
        """
        Returns an alias for the :class:`Variable` variable. Raises a
        ``KeyError`` if no such alias is found.
        """
        for alias, var in self._alias_map.iteritems():
            if var == variable:
                return alias
        raise KeyError('No alias found for <' + variable.qname() + '>.')
    def _code(self, b, t):
        """
        Internal version of Component.code()
        """
        pre = t*TAB
        b.write(pre + '[' + self.name() + ']\n')
        # Append meta properties
        self._code_meta(b, t)
        # Append aliases
        for alias, var in self._alias_map.iteritems():
            b.write(pre + 'use ' + var.qname() + ' as ' + alias + '\n')
        # Append values
        for v in self.variables(sort=True):
            v._code(b, t)
        b.write(pre + '\n')
    def qname(self, hide=None):
        """
        A component's ``qname`` is simply its name. No model name is prefixed.
        """
        return self._name
    def has_alias(self, name):
        """
        Returns ``True`` if this :class:`Component` contains an alias with the
        given name.
        """
        return name in self._alias_map
    def has_alias_for(self, variable):
        """
        Returns ``True`` if this :class:`Component` has an alias for the given
        :class:`Variable`.
        """
        return variable in self._alias_map.itervalues()     
    def remove_alias(self, name):
        """
        Removes an alias from this :class:`Component`.
        """
        del(self._alias_map[name])
    def remove_aliases_for(self, var):
        """
        Removes any alias for the given variable from this :class:`Component`.
        """
        todo = []
        for name, avar in self._alias_map.iteritems():
            if avar == var:
                todo.append(name)
        for name in todo:
            del(self._alias_map[name])
    def __repr__(self):
        """
        Returns a representation of this component.
        """
        return '<Component(' + self.qname() + ')>'
    def validate(self, fix_crudely=False):
        """
        Attempts to check component validity, raises errors if it isn't.

        An attempt to crudely "fix" the component can be made by setting
        ``fix_crudely`` to True.
        """
        m = self.model()
        if m is None:
            raise myokit.IntegrityError('No model found in hierarchy for <'
                + self.qname() + '>.')
        # Validate child variables
        for v in self.variables():
            if v._parent != self:
                msg = 'Child variable\'s parent does not match with actual' \
                      ' parent: parent of <' + v.qname() + '> is set to <'\
                      + str(v._parent.qname()) + '>, but the variable is' \
                      ' stored in <' + str(self.qname()) + '>.'
                if not fix_crudely:
                    raise myokit.IntegrityError(msg, v._token)
                else:
                    self.model()._warn(msg + ' Attempting fix.')
                    v._parent = self
            # Deep validation
            v.validate(fix_crudely)
class Variable(VarOwner):
    """
    Represents a variable.

    Each variable has a single defining equation. For state variables, this
    equation has a derivative on the left-hand side (lhs), for all other
    variables the lhs of the defining equation is simply the variable's name.

    Variables can be made into state variables using
    :meth:`Variable.promote()`.

    Variables can be made to represent an externally defined variable using
    :meth:`Variable.set_binding()`. In this case, the right hand side set for
    a variable will be used as an alternative in situations where the external
    input is not available.
    
    Meta-data properties can be accessed via the property ``meta``, for example
    ``model.meta['key']= 'value'``.
    """
    def __init__(self, parent, name):
        super(Variable, self).__init__(parent, name)
        # Indice, only set if this is a state variable
        self._indice = None
        # This variable's unit, if given, else dimensionless
        self._unit = None
        # This variable's label, if given
        self._label = None        
        # This variable's binding label, if given
        self._binding = None
        # Used (created and deleted) by _validate_solvability()
        #   self._used
        # Cached return values for is_constant, etc
        # For internal use only!
        self._is_nested = isinstance(self._parent, Variable)
        self._is_bound = False
        self._is_constant = True
        self._is_intermediary = False
        self._is_literal = True
        self._is_state = False
        # Cached lists of references by and to
        # References are only counted for the variable's defining equation (IE
        # the expression returned by rhs()). References to values of state
        # variables are stored separately. Bound variables are treated as if
        # they were unbound.
        self._refs_by = set()   # Vars that refer to this var
        self._refs_to = set()   # Vars that this var refers to
        self._srefs_by = set()  # Vars that refer to this state var's value
        self._srefs_to = set()  # State var values that this var refers to
        # Left-hand side representation (name or dot)
        self._lhs = myokit.Name(self)
        # Right-hand side representation
        self._rhs = None
    def binding(self):
        """
        Returns this variable's binding label or ``None`` if no binding is set.
        """
        return self._binding
    def _clone1(self, parent):
        """
        Performs step 1 of cloning this variable into the newly created
        :class:`Component` or :class:`Variable` given as ``parent``. Creates a
        variable and the hierarchy of nested variables, but doesn't fill in the
        details.
        """
        v = parent.add_variable(self.name())
        self._clone_modelpart_data(v)
        for k in self.variables():
            k._clone1(v)
    def _clone2(self, v, lhsmap):
        """
        Performs step 2 of cloning this variable into ``v``. Adds equations,
        bindings, unit etc.
        
        The argument ``lhsmap`` should be a dictionary mapping old
        :class:`LhsExpression` objects their equivalents in the new model.
        """
        # _indice is set by promoting (done by model)
        # _binding
        if self._binding:
            v.set_binding(self._binding)
        # _label
        if self._label:
            v.set_label(self._label)
        # _unit (Units are immutable, no need to clone)
        v._unit = self._unit
        # Cached values are updated automatically.
        # Cached references are set by set_rhs 
        # Set RHS
        if self._rhs:
            v.set_rhs(self._rhs.clone(subst=lhsmap))
        # Clone child variables
        for k in self.variables():
            k._clone2(v[k.name()], lhsmap)
    def _code(self, b, t):
        """
        Create the code for this variable and any child variables.
        """
        # Create header line
        c = self.parent(Component)
        lhs = self._lhs.code(c) if self._lhs else 'UNNAMED'
        rhs = self._rhs.code(c) if self._rhs else 'UNDEFINED'
        head = lhs + ' = ' + rhs
        # Get description from meta data
        try:
            desc = self.meta['desc']
        except KeyError:
            desc = None
        # Add bind and description shortcuts
        omit = []
        unit = self._unit
        bind = self._binding
        label = self._label
        if len(head) < 40:
            if self._rhs and self._rhs.is_literal():
                # Append bind
                if bind:
                    text = head + ' bind ' + bind
                    if len(text) < 79:
                        head = text
                        bind = None
                # Append label
                if label:
                    text = head + ' label ' + label
                    if len(text) < 79:
                        head = text
                        label = None                   
            # Append desc
            if desc:
                text = head + ' : ' + desc
                if len(text) < 79 and '\n' not in desc and '\r' not in desc:
                    head = text
                    omit.append('desc')
        # Append header line
        pre = t*TAB
        eol = '\n'
        b.write(pre + head + eol)
        # Indent!
        t += 1
        pre = t*TAB
        # Append unit
        if unit is not None:
            b.write(pre + 'in ' + str(unit) + eol)
        # Append binding
        if bind:
            b.write(pre + 'bind ' + bind + eol)
        # Append label
        if label:
            b.write(pre + 'label ' + label + eol)
        # Append meta properties
        self._code_meta(b, t, ignore=omit)
        # Append nested variables
        for var in self.variables(sort=True):
            var._code(b, t)
    def _delete(self, recursive=False, whole_component=False):
        """
        Tells this variable that it's going to be deleted.
        
        Errors will be raised if other variables depend on this one, unless
        specified differently using the following arguments:
        
        ``recursive``
            If set to ``True``, no errors will be raised if child variables of
            this variable depend on it.
        ``whole_component``
            If set to ``True``, no errors will be raised if other variables in
            the same component depend on it. This is used when deleting whole
            components.
        
        """
        kids = [x for x in self.variables()]
        if kids and not (recursive or whole_component):
            raise myokit.IntegrityError('Variable <' + self.qname()
                + '> can not be removed: it has children ' + ' and '.join(
                ['<' + v.qname() + '>' for v in kids]) + '.')        
        if self._refs_by or self._srefs_by:
            refs = self._refs_by.union(self._srefs_by)
            if recursive:
                # Refs from child variables are allowed
                okay = set([x for x in refs if x.has_ancestor(self)])
                refs = refs.difference(okay)
                del(okay)
                # Nested variables can not be referred to by outside variables,
                # so this action doesn't have to be repeated for the nested
                # variables.
            if whole_component:
                # Refs from within the same component are okay
                comp = self.parent(Component)
                okay = set([x for x in refs if x.parent(Component)==comp])
                refs = refs.difference(okay)
                del(okay)            
            if refs:
                raise myokit.IntegrityError('Variable <' + self.qname()
                    + '> can not be removed: it is used by ' + ' and '.join(
                    ['<' + v.qname() + '>' for v in refs]) + '.')
        # Tell other variables it no longer depends on them
        for var in self._refs_to:
            var._refs_by.remove(self)
        for var in self._srefs_to:
            var._srefs_by.remove(self)
        # State variable? Then demote first
        if self.is_state():
            self.demote()
        # Delete child variables
        if recursive:
            for kid in kids:
                # Call this method for each kid (and cascade to kid-kids)
                kid._delete(recursive=True, whole_component=whole_component)
                # Remove kid from list of nested variables
                self._remove_variable_internal(kid)
        # Remove any aliases
        m = self.parent(Model)
        for c in m.components():
            c.remove_aliases_for(self)
        # Remove parent links
        super(Variable, self)._delete()
    def demote(self):
        """
        Turns a state variable into an intermediary variable.

        This will reset the validation status of the model this variable
        belongs to.
        """
        if self._indice is None:
            raise Exception('Variable is not a state variable.')
        model = self.model()
        try:
            # Remove initial value
            del(model._current_state[self._indice])
            # Remove this variable from the state
            del(model._state[self._indice])
            # Set lhs to name expression
            self._lhs = myokit.Name(self)
            # Remove this variable's indice
            self._indice = None
            # Reset other states' indices
            model._reset_indices()
        finally:
            model._reset_validation()
            self._reset_cache(bubble=True)
    def eq(self):
        """
        Returns this variable's defining equation.

        For state variables this will be an equation for the variable's
        derivative. For ordinary variables the equation for the variable's
        value is returned.
        """
        return Equation(self._lhs, self._rhs)
    def eval(self):
        """
        Evaluates this variable's defining equation and returns the result.
        """
        return self._rhs.eval()
    def indice(self):
        """
        For state variables, this will return their index in the state vector.
        For all other variables, this will raise an exception.
        """
        if self._indice is None:
            raise Exception('Only state variables have initial values.')
        return self._indice
    def is_bound(self):
        """
        Returns ``True`` if a binding label has been added to this variable.
        """
        return self._is_bound
    def is_constant(self):
        """
        Returns ``True`` if this variable is constant.

        Myokit doesn't discern between mathematical and physical constants,
        parameters etc. Anything that doesn't change during a simulation is
        termed a constant. Note that this specifically excludes variables bound
        to external inputs.
        """
        return self._is_constant
    def is_intermediary(self):
        """
        Returns ``True`` if this variable is an intermediary variable, i.e. not
        a constant or a state variable (and not bound to an external variable).
        """
        return self._is_intermediary
    def is_labelled(self):
        """
        Returns ``True`` if a label has been added to this variable.
        """
        return self._label is not None
    def is_literal(self):
        """
        Returns ``True`` if this variable's expression contains only literal
        values.
        """
        return self._is_literal
    def is_nested(self):
        """
        Returns ``True`` if this variable's parent is another variable.
        """
        return self._is_nested
    def is_referenced(self):
        """
        Returns ``True`` if other variables reference this variable's ``lhs``.
        For states, this means it only returns ``True`` if other variables
        depend on this variable's derivative.
        """
        return len(self._refs_by) > 0
    def is_state(self):
        """
        Returns ``True`` if this variable is a state variable.
        """
        return self._is_state
    def label(self):
        """
        Returns this variable's label or ``None`` if no label is set.
        """
        return self._label
    def lhs(self):
        """
        Returns the left-hand side of the equation defining this variable.

        For state variables this will be a :class:`myokit.Derivative`, for
        all others this should be a :class:`myokit.Name`.
        """
        return self._lhs
    def promote(self, state_value=0):
        """
        Turns this variable into a state variable with a current state value
        given by ``state_value``.

        This will reset the validation status of the model this variable
        belongs to.
        """
        if self._indice is not None:
            raise Exception('Variable is already a state variable')
        if not isinstance(self._parent, Component):
            raise Exception('State variables can only be added to Components.')
        if self._binding is not None:
            raise Exception('State variables cannot be bound to an external'
                ' value.')
        if isinstance(state_value, myokit.Expression):
            if not state_value.is_literal():
                raise myokit.NonLiteralValueError('Expressions for state'
                    ' values can not contain references to other variables.')
        model = self.model()
        try:
            # Set lhs to derivative expression
            self._lhs = myokit.Derivative(myokit.Name(self))
            # Get new indice
            self._indice = len(model._state)
            # Add to list of states
            model._state.append(self)
            # Add value to list of current values
            model._current_state.append(float(state_value))
        finally:
            model._reset_validation()
            self._reset_cache(bubble=True)
    def pyfunc(self, use_numpy=True, arguments=False):
        """
        Returns a python function that evaluates this variable as a function of
        the state variables it depends on.

        The argument names used by the returned function will be the variables'
        unames, ordered alphabetically.

        If the function runs in "numpy-mode", which is enabled by default, the
        vector ready versions of ``log``, ``exp`` etc are used and piecewise
        statements are evaluated using an array ready piecewise function. To
        disable this functionality, set ``use_numpy=False``.

        To obtain more information about the arguments in the created function
        handle, set the optional argument ``arguments`` to ``True``. With this
        setting, the function will return a tuple ``(handle, vars)`` where
        ``handle`` is  the function handle and ``vars`` is a list of
        :class:`myokit.LhsExpression` objects in the same order as the function
        arguments.
        """
        # Get expression writer
        if use_numpy:
            import numpy
            w = myokit.numpywriter()
        else:
            import math
            w = myokit.pywriter()
        # Get arguments, equations
        eqs, args = self.model().expressions_for(self)
        # Handle function arguments
        func = [w.ex(x) for x in args]
        func, args = zip(*sorted(zip(func, args))) # Sort both following func
        # Create function text
        func = ['def var_pyfunc_generated(' + ','.join(func) + '):']
        tab = '\t'
        if use_numpy:
            func.append(tab + 'with numpy.errstate(all=\'ignore\'):')
            tab += '\t'
        for eq in eqs:
            func.append(tab + w.eq(eq))
        func.append(tab + 'return ' + w.ex(eqs[-1].lhs))
        func = '\n'.join(func) + '\n'
        # Create function
        local = {}
        if use_numpy:
            exec(func, {'numpy':numpy}, local)
        else:
            exec(func, {'math':math}, local)
        handle = local['var_pyfunc_generated']
        # Return
        if arguments == True:
            return (handle, args)
        else:
            return handle
    def refs_by(self, state_refs=False):
        """
        Returns an iterator over the set of :class:`Variables <Variable>`  that
        refer to this variable in their defining equation.
        
        For state variables, there are two things to take into account:
        
         1. By default, only references to this variable's defining
            :class:`LhsExpression` (I.E. the one returned by
            :meth:`lhs() <Variable.lhs()>`) are returned. For a state
            variable ``x``, this means the returned result contains all
            variables referring to ``dot(x)``. To get an iterator over the
            variables referring to ``x`` add the optional attribute
            ``state_refs=True``. For non-state variables this setting will
            trigger a :class:`ValueError`.
         2. In cases where ``dot(x) = f(x)``, the returned iterator will _not_
            include ``x`` in the list of variables depending on ``x``.
            
        """
        if state_refs:
            if not self._is_state:
                raise ValueError('The argument "state_refs=True" can only be'
                    ' used on state variables.')
            return iter(self._srefs_by)
        return iter(self._refs_by)
    def refs_to(self, state_refs=False):
        """
        Returns an iterator over the set of :class:`Variables <Variable>` that
        this variable's defining equation refers to.
        
        By default, this will _not_ include references to a state variable's
        value. To obtain a list of state variables whose value is referenced,
        use ``state_refs=True``.
        """
        if state_refs:
            return iter(self._srefs_to)
        else:        
            return iter(self._refs_to)
    def rename(self, new_name):
        """
        Renames this variable.
        """
        p = self._parent
        if not p:
            raise Exception('Unable to rename variable without parent.')
        p.move_variable(self, p, new_name)
    def __repr__(self):
        if self._indice is not None:
            return '<State(' + self.qname() + ')>'
        else:
            return '<Var(' + self.qname() + ')>'
    def _reset_cache(self, bubble=False):
        """
        Updates this variable's cached attributes. If ``bubble`` is set to
        ``True`` and this variable's cache state changes, a cache reset will be
        triggered in all dependent variables.
        """
        if bubble:
            s_old = (self._is_bound, self._is_state, self._is_intermediary,
                     self._is_literal, self._is_constant, self._is_nested)
        self._is_bound = self._binding is not None
        self._is_state = self._indice is not None
        self._is_nested = isinstance(self._parent, Variable)
        if self._is_state or self._is_bound or self._rhs is None:
            self._is_constant = False
            self._is_literal = False
            self._is_intermediary = False
        else:
            self._is_constant = self._rhs.is_constant()
            self._is_literal = self._rhs.is_literal()
            self._is_intermediary = not self._is_constant
        if bubble:
            if s_old != (self._is_bound, self._is_state, self._is_intermediary,
                         self._is_literal, self._is_constant, self._is_nested):
                for ref in self._refs_by:
                    ref._reset_cache(bubble=True)
    def rhs(self):
        """
        Returns the right-hand side of the equation defining this variable.

        For state variables this will be the expression for their derivative,
        for all others an expression for their value is returned.
        """
        return self._rhs
    def set_binding(self, binding):
        """
        Adds a unique binding label to this variable, indicating it can be used
        as an entry point for external inputs.
        
        To remove a binding, call ``set_binding(None)``.

        Adding or removing binding labels resets the model's validation status.
        """
        if binding is not None:
            # Check name
            binding = check_name(binding)
            # Check for existing binding 
            if self._binding is not None:
                raise myokit.InvalidBindingError('The variable <'
                    + self.qname() + '> is already bound to "' + self._binding
                    + '".')
            # Check if not a state
            if self._indice is not None:
                raise myokit.InvalidBindingError('State variables cannot be'
                    ' bound to an external value.')
        # Set binding (model checks uniqueness)
        model = self.model()
        try:
            if binding is None:
                if self._binding is not None:
                    model._register_binding(self._binding, None)
            else:
                model._register_binding(binding, self)
            self._binding = binding
        finally:
            # Clear cache and cache of dependent variables
            self._reset_cache(bubble=True)
            # Reset model validation
            model._reset_validation()
    def set_label(self, label=None):
        """
        Adds a unique ``label`` for this variable, indicated that its value can
        be read by external users.
        
        To remove a label call ``set_label(None)``.
        """
        # Remove label?
        if label is None:
            if self._label is not None:
                self.model()._register_label(self._label, None)
                self._label = None
            return
        # Check name
        label = check_name(label)
        # Check for existing label or binding
        if self._label:
            raise myokit.InvalidLabelError('The variable <' + self.qname()
                + '> already has a label "' + self._label + '".')
        # Set label (model checks uniqueness)
        self.model()._register_label(label, self)
        self._label = label
    def set_rhs(self, rhs):
        """
        Changes the expression for this variable to the given float or
        ``myokit.Expression`` object.

        For state variables, this updates their derivative. For all others this
        will update the expression for their value.

        This will reset the validation status of the model this variable
        belongs to.
        """
        # Handle string and number rhs's
        if not isinstance(rhs, myokit.Expression):
            if type(rhs) in (str, unicode):
                rhs = myokit.parse_expression(rhs, context=self)
            else:
                rhs = myokit.Number(rhs)
        # Update the refs-by stored in the old dependencies
        for ref in self._refs_to:
            ref._refs_by.remove(self)
        # Get new references made by this variable, filter out references to
        # to values of state variables.
        self._refs_to = set([r.var() for r in rhs.references()
            if not r.is_state_value()])
        self._srefs_to = set([r.var() for r in rhs.references()
            if r.is_state_value()])
        try:
            # Ignore references like dot(y) = y
            self._srefs_to.remove(self)
        except KeyError:
            pass
        # Update the refs-by stored in the new dependencies of this var
        for ref in self._refs_to:
            ref._refs_by.add(self)
        for ref in self._srefs_to:
            ref._srefs_by.add(self)
        # Set rhs
        self._rhs = rhs
        self.model()._reset_validation()
        self._reset_cache(bubble=True)
    def set_state_value(self, value):
        """
        If this variable is a state variable, its current value will be
        updated. For all other variables this raises an exception.
        """
        if not self._is_state:
            raise Exception('Only state variables have state values.')
        model = self.model()
        if isinstance(value, myokit.Expression):
            if not value.is_literal():
                raise myokit.NonLiteralValueError('Expressions for state'
                    ' values can not contain references to other variables.')
        model._current_state[self._indice] = float(value)
        # No need to reset validation status or cache here.
    def set_unit(self, unit=None):
        """
        Changes this variable's unit. The unit can be set to any valid Unit
        object, or ``None`` to remove the unit.
        """
        if unit is None or isinstance(unit, myokit.Unit):
            self._unit = unit
        elif type(unit) in (str, unicode):
            self._unit = myokit.Unit.parse_simple(unit)
        else:
            raise TypeError('Method set_unit() expects a myokit.Unit or None.')
        # No need to reset validation status or cache. Units are checked only
        # by the model.
    def state_value(self):
        """
        For state variables, this will return their current value.
        For all other variables, this will raise an exception.
        """
        if not self._is_state:
            raise Exception('Only state variables have initial values.')
        return self.model()._current_state[self._indice]
    def unit(self):
        """
        Returns the unit specified for this variable. If no unit was set, the
        unit "dimensionless" is returned.
        
        Variables' units are set using :meth:`Variable.set_unit()` or using
        the ``in`` keyword in ``.mmt`` files. The unit set for a variable is
        the unit this variable's expression _should_ have. This allows
        expressions to be validated by computing the resulting unit of an
        expression and comparing it with the value set for the variable.
        """
        return self._unit
    def validate(self, fix_crudely=False):
        """
        Attempts to check this variable's validity, raises errors if it isn't.

        An attempt to crudely "fix" the variable can be made by setting
        ``fix_crudely`` to True.
        """
        # Validate rhs
        if self._rhs is None:
            raise myokit.MissingRhsError(self)
        self._rhs.validate()
        # Check state variables
        is_state = self._indice is not None
        is_deriv = self.lhs().is_derivative()
        if is_state:
            if not is_deriv:
                raise myokit.IntegrityError('Variable <' + self.qname()
                    + '> is listed as a state variable but its lhs is not a'
                    ' derivative.')
            if self._is_nested:
                raise myokit.IntegrityError('State variables should not be'
                    ' nested: <' + str(self.qname()) + '>.')
            m = self.model()
            if not m._state[self._indice] == self:
                raise myokit.IntegrityError('State variable not listed in'
                    ' model state vector at correct indice: <'
                    + self.qname() + '>.')
        elif is_deriv:
            raise myokit.IntegrityError('A derivative was set for <'
                + self.qname() + '> but this is not a state variable.')
        # Check for component as parent
        m = self.parent(Component)
        if m is None:
            raise myokit.IntegrityError('No component found in hierarchy for <'
                + self.qname() + '>.')
        # Validate references
        for ref in self._refs_to:
            # References can be made to:
            #  1. This variable's children or any of this variable's
            #     ancestor's children.
            #  2. Any component variable.
            # Quick checks first: sibling or direct child
            if ref._parent == self._parent or ref._parent == self:
                continue
            # Component variable?
            if isinstance(ref._parent, Component):
                continue
            # Child of ancestor of this variable
            if ref._parent.is_ancestor(self):
                continue
            raise myokit.IllegalReferenceError(ref, self)
        # Validate child variables
        for v in self.variables():
            if v._parent != self:
                msg = 'Child variable\'s parent does not match with actual' \
                      ' parent: parent of <' + v.qname() + '> is set to <'\
                      + str(v._parent.qname()) + '>, but the variable is' \
                      ' stored in <' + self.qname() + '>.'
                if fix_crudely:
                    self.model()._warn(msg + ' Attempting fix.')
                    v._parent = self
                else:
                    raise myokit.IntegrityError(msg, v._token)
            # Deep validation
            v.validate(fix_crudely)
    def value(self):
        """
        Will return the value of this variable's defining right-hand side
        expression.
        """
        return self._rhs.eval()
class Equation(object):
    """
    Defines an equation: a statement that a left-hand side is equal to a
    right-hand side.

    The sides of an equation are stored in the properties ``lhs`` and ``rhs``.
    """
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def __eq__(self, other):
        if not isinstance(other, Equation):
            return False
        else:
            return self.lhs == other.lhs and self.rhs == other.rhs
    def code(self):
        b = StringIO()
        self.lhs._code(b, None)
        b.write(' = ')
        self.rhs._code(b, None)
        return b.getvalue()
    def __hash__(self):
        #Note: Hash should never change during object's lifetime!
        return hash(self.code())
    def __iter__(self):
        # Having this allows "lhs, rhs = eq"
        return iter((self.lhs, self.rhs))
    def __ne__(self, other):
        return not self.__eq__(other)
    def __str__(self):
        return self.code()
class EquationList(list, VarProvider):
    """
    Represents an ordered list of :class:`Equation` objects
    """
    def _create_variable_stream(self, deep, sort):
        # Always sorted
        def stream(lst):
            for eq in lst:
                yield eq.lhs.var()
        return stream(self)
class UserFunction(object):
    """
    Defines a user function. User functions are not ``Expression`` objects, but
    template expressions that are converted upon parsing. They allow common
    functions (for example a boltzman function) to be used in string
    expressions.
    
    Arguments:
    
    ``name``
        The user function's name
    ``arguments``
        A list of function argument names
    ``template``
        The :class:`Expression` evaluating this function.
        
    """
    def __init__(self, name, arguments, template):
        self._name = str(name)
        self._arguments = list(arguments)
        self._template = template
    def arguments(self):
        """
        Returns an iterator over this user function's arguments.
        """
        return iter(self._arguments)
    def convert(self, arguments):
        """
        Returns an :class:`Expression` object.
        """
        if len(arguments) != len(self._arguments):
            raise ValueError('Wrong number of input arguments.')
        for arg in self._arguments:
            if arg not in arguments:
                raise ValueError('Missing input argument: <' + str(arg) + '>.')
        return self._template.clone(arguments)
