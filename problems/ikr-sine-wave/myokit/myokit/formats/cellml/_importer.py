#
# Imports a model definition from a CellML file
# Only partial CellML support is provided.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import os
import shutil
import textwrap
import xml.dom.minidom
from collections import OrderedDict as odict
import myokit
import myokit.units
from myokit.mxml import dom_child, dom_next
from myokit.formats.mathml import parse_mathml_rhs
class CellMLError(myokit.ImportError):
    """
    Raised when a fatal error occurs when importing CellML
    """
class CellMLImporter(myokit.formats.Importer):
    """
    This :class:`Importer <myokit.formats.Importer>` imports a model definition
    from CellML.
    """
    def __init__(self, verbose=False):
        super(CellMLImporter, self).__init__()
        self._generated_names = None
    def _flatten(self, node):
        """
        Reduces a node's contents to flat text and returns it.
        """
        def text(node, buff=None):
            if buff is None:
                buff = []
            if node.nodeType == node.TEXT_NODE:
                t = node.nodeValue
                t = t.replace('\n', ' ')
                t = t.replace('\r', ' ')
                t = t.replace('\f', ' ')
                t = t.strip()
                if t != '':
                    buff.append(str(t.encode('ascii', errors='replace')))
            else:
                for kid in node.childNodes:
                    text(kid, buff)
            return buff
        return textwrap.fill('\n'.join(text(node)), 75,
            replace_whitespace=False)
    def info(self):
        """
        Returns a string containing information about this importer.
        """
        return "Loads a Model definition from a CellML file."
    def model(self, path):
        """
        Reads a CellML file and returns a :class:`myokit.Model`.
        """
        # Reset list of generated names
        self._generated_names = {}
        # Parse xml file to dom
        path = os.path.abspath(os.path.expanduser(path))
        dom = xml.dom.minidom.parse(path)
        # Parse dom to model
        model = self._parse_model(dom)
        # Run model validation, order variables etc
        try:
            model.validate()
        except myokit.IntegrityError as e:
            self.warn('Integrity error found in model: ' + e.message)
        except Exception as f:
            import traceback
            self.warn('Exception occurred when validating model. '
                + traceback.format_exc())
        return model
    def _parse_model(self, dom):
        """
        Parses a dom tree into a myokit model.
        """
        model_tag = dom.getElementsByTagName('model')[0]
        name = self._sanitise_name(model_tag.getAttribute('name'))
        model = myokit.Model(name)
        model.meta['author'] = 'Cellml converter'
        # Parse model meta information
        desc = []
        for tag in model_tag.getElementsByTagName('documentation'):
            desc.append(self._flatten(tag).strip())
        desc = ('\n'.join(desc)).strip()
        if desc:
            model.meta['desc'] = desc
        # Check for unsupported CellML features
        # Check for <import> (allowed in CellML 1.1+)
        if model_tag.getElementsByTagName('import'):
            raise CellMLError('The CellML <import> tag is not supported.')            
        # Check for <reaction> (allowed in any CellML)
        if model_tag.getElementsByTagName('import'):
            raise CellMLError('The CellML <import> tag is not supported.')            
        # Check for <factorial> (allowed in all CellML, but why?)
        if model_tag.getElementsByTagName('factorial'):
            self.warn('The <factorial> tag is not supported.')
        # Check for MathML features not currently allowed in CellML
        # Check for <partialdiff> (not allowed in any CellML)
        if model_tag.getElementsByTagName('partialdiff'):
            self.warn('The <partialdiff> tag is not supported.')
        # Check for <sum> (not allowed in any CellML)
        if model_tag.getElementsByTagName('sum'):
            self.warn('The <sum> tag is not supported.')
        # Parse unit definitions
        si_units, munits, cunits = self._parse_units(model_tag)
        def convert_unit(unit):
            """
            Parses a CellML unit (string) and returns a :class:`myokit.Unit` or
            ``None`` if successful, returns a string if conversion failed.
            """
            if unit:
                if str(unit) in si_units:
                    unit = si_units[unit]
                elif unit in munits:
                    unit = munits[unit]
                elif unit in cunits[cname]:
                    unit = cunits[cname][unit]
                else:
                    unit = str(unit)
            return unit
        # Parse components
        components = {} # Dict of (component name, component) pairs
        for tag in model_tag.getElementsByTagName('component'):
            cname = tag.getAttribute('name')
            name = self._sanitise_name(cname)
            self.log('Parsing component: ' + name)
            comp = model.add_component(name)
            components[cname] = comp
        # Parse group relationships
        # There are a number of different group types. Myokit handles the
        # "encapsulation" type of grouping, the rest can be ignored without
        # introducing errors.
        # Dict of encapsulation relations (component, parent component) pairs
        parents = {}
        def scan_encapsulated_children(parent, pcomp):
            """
            Reads parent/child relationships from a <group> or <component_ref>
            tag and adds them to the dict ``parents``.

            Argument ``parent`` should be a <component_ref> tag and ``pcomp``
            should be the corresponding cellml component object.
            """
            kid = dom_child(parent, 'component_ref')
            while kid is not None:
                # Get cellml component from name
                try:
                    comp = components[kid.getAttribute('component')]
                except KeyError:
                    raise CellMLError('Group registered for unknown'
                        ' component: ' + kid.getAttribute('component'))
                # Log relationship
                self.log('Component <' + comp.qname() + '> is encapsulated'
                    ' in <' + pcomp.qname() + '>.')
                # Add relationship
                parents[comp] = pcomp
                # Scan kid for children
                scan_encapsulated_children(kid, comp)
                # Move to next kid
                kid = dom_next(kid, 'component_ref')
        for group in model_tag.getElementsByTagName('group'):
            # Filter out encapsulation groups
            is_encapsulation = False
            for ref in group.getElementsByTagName('relationship_ref'):
                if ref.getAttribute('relationship') == 'encapsulation':
                    is_encapsulation = True
                    break;
            if not is_encapsulation:
                continue
            # Parse and store relationships
            parent = dom_child(group, 'component_ref')
            while parent is not None:
                # Get cellml component from name
                try:
                    pcomp = components[parent.getAttribute('component')]
                except KeyError:
                    raise CellMLError('Group registered for unknown'
                        ' component: ' + parent.getAttribute('component'))
                # Add kids
                scan_encapsulated_children(parent, pcomp)
                # Search for next parent
                parent = dom_next(parent, 'component_ref')
        # Parse variables
        references = {} # Dict (component name, (var name, var))
        interfaces = {} # Dict (component name, (var name,(pub, pri, unit)))
        variables  = {} # Dict (component name, (var name, variable))
        values     = {} # Dict (component name, (var name, variable value))
        for ctag in model_tag.getElementsByTagName('component'):
            cname = ctag.getAttribute('name')
            comp  = components[cname]
            references[cname] = rfs = {}
            interfaces[cname] = ifs = {}
            variables[cname]  = vrs = {}
            values[cname]     = vls = {}
            for vtag in ctag.getElementsByTagName('variable'):
                vname = vtag.getAttribute('name')
                # Get public and private interface
                pub = vtag.getAttribute('public_interface')
                pri = vtag.getAttribute('private_interface')
                if pub not in ('in', 'out'): pub = None
                if pri not in ('in', 'out'): pri = None
                # Get unit
                unit = convert_unit(vtag.getAttribute('units'))
                # Native variable? Then create
                if not (pub == 'in' or pri == 'in'):
                    name = self._sanitise_name(vname)
                    self.log('Parsing variable: ' + name)
                    var = comp.add_variable(name)
                    vrs[vname] = var
                    init = str(vtag.getAttribute('initial_value'))
                    if init != '':
                        vls[vname] = init
                    # Set unit
                    if type(unit) == str:
                        var.meta['cellml_unit'] = unit
                    else:
                        var.set_unit(unit)
                    # Add resolved reference
                    rfs[vname] = var
                else:
                    # Otherwise, store as unresolved reference
                    rfs[vname] = None
                # Store reference information
                ifs[vname] = (pub, pri, unit)
        # Parse connections
        for tag in model_tag.getElementsByTagName('connection'):
            # Find linked components
            map_components = tag.getElementsByTagName('map_components')[0]
            cname1 = map_components.getAttribute('component_1')
            cname2 = map_components.getAttribute('component_2')
            for comp in (cname1, cname2):
                if not comp in components:
                    raise CellMLError('Connection found for unlisted'
                        ' component: <' + comp + '>.')
            comp1 = components[cname1]
            comp2 = components[cname2]
            # If component is encapsulated, find parent
            try:
                par1 = parents[comp1]
            except KeyError:
                par1 = None
            try:
                par2 = parents[comp2]
            except KeyError:
                par2 = None
            # Get relevant lists for components
            ifs1 = interfaces[cname1]
            ifs2 = interfaces[cname2]
            rfs1 = references[cname1]
            rfs2 = references[cname2]
            # Find all references
            for pair in tag.getElementsByTagName('map_variables'):
                ref1 = pair.getAttribute('variable_1')
                ref2 = pair.getAttribute('variable_2')
                # Check interfaces
                try:
                    int1 = ifs1[ref1]
                except KeyError:
                    self.warn('No interface found for variable <' + str(ref1)
                        + '>, unable to resolve connection.')
                    break
                try:
                    int2 = ifs2[ref2]
                except KeyError:
                    self.warn('No interface found for variable <' + str(ref2)
                        + '>, unable to resolve connection.')
                    break
                # Determine direction of reference
                ref_to_one = None
                if int2[0] == 'in' and (par1 == par2 or par2 == comp1):
                    # Reference from comp2 to its parent or sibling comp1
                    ref_to_one = True
                elif int1[0] == 'in' and (par1 == par2 or par1 == comp2):
                    # Reference from comp1 to its parent or sibling comp2
                    ref_to_one = False
                elif int2[1] == 'in' and par1 == comp2:
                    # Reference from comp2 to its child comp1
                    ref_to_one = True
                elif int1[1] == 'in' and par2 == comp1:
                    # Reference from comp1 to its child comp2
                    ref_to_one = False
                else:
                    self.warn('Unable to resolve connection between <'
                        + str(ref1) + '> in ' + str(comp1) + '('
                        + str(int1[0]) + ', ' + str(int1[1]) + ') and <'
                        + str(ref2) + '> in ' + str(comp2) + '('
                        + str(int2[0]) + ', ' + str(int2[1]) + ').')
                    continue
                # Check units
                if int1[2] != int2[2]:
                    self.warn('Unit mismatch between <' + str(ref1) + '> in '
                        + str(int1[2]) + ' and <' + str(ref2) + '> given in '
                        + str(int2[2]) + '.')
                # Now point reference at variable or reference in other comp
                try:
                    ref = rfs1[ref1] if ref_to_one else rfs2[ref2]
                except KeyError:
                    a, b = ref2, ref1 if ref_to_one else ref1, ref2
                    self.warn('Unable to resolve reference of ' + str(a)
                        + ' to ' + str(b) + '.')
                    continue
                if ref_to_one:
                    rfs2[ref2] = (cname1, ref1)
                    self.log('Variable <' + str(ref2) + '> in <' + str(cname2)
                        + '> points at <' + str(ref1) + '> in <' + str(cname1)
                        + '>.')
                else:
                    rfs1[ref1] = (cname2, ref2)
                    self.log('Variable <' + str(ref1) + '> in <' + str(cname1)
                        + '> points at <' + str(ref2) + '> in <' + str(cname2)
                        + '>.')
        # Check for references that are never connected
        for cname, rfs in references.iteritems():
            for vname, ref in rfs.iteritems():
                if ref is None:
                    self.warn('Unresolved reference <' + str(vname) + '> in'
                        ' component <' + str(cname) + '>. Creating a dummy'
                        ' variable with this name.')
                    c = components[cname]
                    v = c.add_variable(vname)
                    v.set_rhs(0)
                    rfs[vname] = v
        # The references should now all point to either a variable or a
        # reference to another variable. In the next step, these are resolved.
        for cname, rfs in references.iteritems():
            for vname, ref in rfs.iteritems():
                if type(ref) == tuple:
                    while True:
                        ref = references[ref[0]][ref[1]]
                        if type(ref) != tuple:
                            break
                    rfs[vname] = ref
        # MathML number post-processor to extract unit
        def npp(node, number):
            unit = convert_unit(node.getAttribute('cellml:units'))
            if unit:
                return myokit.Number(number.eval(), unit)
            else:
                return number
        # MathML derivative post-processor to check if we're only using time
        # derivatives
        global time
        time = None  
        def dpp(lhs):
            var = lhs.var()
            global time
            if time is None:
                time = var
            elif time != var:
                raise CellMLError('Found derivatives to two different'
                    ' variables: <' + str(time) + '> and <' + str(var) + '>.')
        # MathML expression parser            
        def mathml(node, rfs):
            return parse_mathml_rhs(node,
                var_table=rfs,
                logger=self,
                number_post_processor=npp,
                derivative_post_processor=dpp)
        # Parse expressions
        for ctag in model_tag.getElementsByTagName('component'):
            cname = ctag.getAttribute('name')
            math = ctag.getElementsByTagName('math')
            vrs = variables[cname]
            vls = values[cname]
            rfs = references[cname]
            n = 0
            for m in math:
                tag = dom_child(m)
                while tag:
                    if tag.tagName != 'apply':
                        raise CellMLError('Unexpected tag in expression: <'
                            + tag.tagName +'>, expecting <apply>.')
                    # First child of tag should be <eq />
                    eq = dom_child(tag, 'eq')
                    if not eq:
                        raise CellMLError('Unexpected content in math of'
                            ' component <' + cname + '>.')
                    # Get lhs and rhs tags
                    lhs_tag = dom_next(eq)
                    rhs_tag = dom_next(lhs_tag)
                    # Check for partial derivatives
                    if lhs_tag.tagName == 'apply':
                        if dom_child(lhs_tag) == 'partialdiff':
                            raise CellMLError('Unexpected tag in expression:'
                                ' expecting <diff>, found <partialdiff>.'
                                ' Partial derivatives are not supported.')
                    # Parse lhs
                    lhs = mathml(lhs_tag, rfs)
                    if not isinstance(lhs, myokit.LhsExpression):
                        raise CellMLError('Error parsing equation: Expecting'
                            ' <ci> or <apply> after <eq> in "' + cname
                            + '", got <' + str(lhs_tag.tagName) + '> instead.'
                            ' Differential algebraic equations are not'
                            ' supported).')
                    # Check variable
                    var = lhs.var()
                    if var not in vrs.values():
                        raise CellMLError('Error: Equation found for unknown'
                            ' variable <' + str(var) + '>.')
                    # Check derivatives
                    if lhs.is_derivative():
                        # Get CellML variable name
                        vname = dom_child(lhs_tag, 'ci')
                        vname = vname.firstChild.data.strip()
                        # Promote variable
                        try:
                            i = float(vls[vname])
                            del(vls[vname])
                        except KeyError:
                            self.warn('No initial value found for <'
                                + var.qname() + '>.')
                            i = 0
                        var.promote(i)
                    # Parse rhs
                    var.set_rhs(mathml(rhs_tag, rfs))
                    # Continue
                    tag = dom_next(tag)
                    n += 1
            self.log('Found ' + str(n) + ' equations in ' + cname + '.')
        # Use remaining initial values (can be used to set constants)
        for cname, vls in values.iteritems():
            vrs = variables[cname]
            for vname, val in vls.iteritems():
                vrs[vname].set_rhs(myokit.Number(val))
        # Bind time variable to engine time
        if time is not None:
            time.set_rhs(0)
            time.set_binding('time')
        # Check for variables with no rhs that are never referenced
        no_rhs = [v for v in model.variables(deep=True) if v.rhs() is None]
        no_rhs = set(no_rhs)
        for var in no_rhs:
            refs = set([x for x in var.refs_by()])
            if len(refs) == 0 or refs in no_rhs:
                self.warn('No expression for variable <' + var.qname() + '> is'
                    ' defined and no other variables reference it. The'
                    ' variable will be removed.')
                var.parent().remove_variable(var)
            else:
                self.warn('No expression for variable <' + var.qname() + '> is'
                    ' defined. This variable will be set to zero.')
                var.set_rhs(myokit.Number(0))
        return model
    def _parse_units(self, model_tag):
        """
        Parses all cellml units into myokit units.

        Returns a tuple (munits, cunits) where munits is a dict mapping cellml
        unit names to myokit unit objects (or None objects if a unit couldn't
        be parsed). The cunits part maps cellml component names to dicts of the
        same structure.
        """
        # <units> Can be placed inside <model>, <component> or <import>
        # for <model> and <import> the units are global.
        # The <import> tag is not supported by this importer.
        # A Units tag can set base_units="yes" to define a new base unit: this
        # is not supported.
        si_units = {
            'dimensionless' : myokit.units.dimensionless,
            'ampere'        : myokit.units.A,
            'farad'         : myokit.units.F,
            'katal'         : myokit.units.kat,
            'lux'           : myokit.units.lux,
            'pascal'        : myokit.units.Pa,
            'tesla'         : myokit.units.T,
            'becquerel'     : myokit.units.Bq,
            'gram'          : myokit.units.g,
            'kelvin'        : myokit.units.K,
            'meter'         : myokit.units.m,
            'radian'        : myokit.units.rad,
            'volt'          : myokit.units.V,
            'candela'       : myokit.units.cd,
            'gray'          : myokit.units.Gy,
            'kilogram'      : myokit.units.kg,
            'metre'         : myokit.units.m,
            'second'        : myokit.units.s,
            'watt'          : myokit.units.W,
            'celsius'       : myokit.units.C,
            'henry'         : myokit.units.H,
            'liter'         : myokit.units.L,
            'mole'          : myokit.units.mol,
            'siemens'       : myokit.units.S,
            'weber'         : myokit.units.Wb,
            'coulomb'       : myokit.units.C,
            'hertz'         : myokit.units.Hz,
            'litre'         : myokit.units.L,
            'newton'        : myokit.units.N,
            'sievert'       : myokit.units.Sv,
            'joule'         : myokit.units.J,
            'lumen'         : myokit.units.lm,
            'ohm'           : myokit.units.R,
            'steradian'     : myokit.units.sr,
        }
        si_prefixes = {
            'yotta' : 1e24,
            'zetta' : 1e21,
            'exa'   : 1e18,
            'peta'  : 1e15,
            'tera'  : 1e12,
            'giga'  : 1e9,
            'mega'  : 1e6,
            'kilo'  : 1e3,
            'hecto' : 1e2,
            'deka'  : 1e1,
            'deci'  : 1e-1,
            'centi' : 1e-2,
            'milli' : 1e-3,
            'micro' : 1e-6,
            'nano'  : 1e-9,
            'pico'  : 1e-12,
            'femto' : 1e-15,
            'atto'  : 1e-18,
            'zepto' : 1e-21,
            'yocto' : 1e-24,
        }
        class Unit:
            def __init__(self, name):
                self.name = name
                self.parts = []
        class Part:
            def __init__(self, base):
                self.base = base
                self.prefix = None
                self.multiplier = None
                self.exponent = None
        def parse(tag):
            """
            Parses <units> tags into Unit objects.
            """
            name = tag.getAttribute('name')
            self.log('Parsing unit: ' + name)
            unit = Unit(name)
            for part in tag.getElementsByTagName('unit'):
                if part.hasAttribute('offset'):
                    self.warn('The "offset" attribute for <unit> tags is not'
                        ' supported.')
                p = Part(part.getAttribute('units'))
                x = part.getAttribute('prefix')
                if x: p.prefix = str(x)
                x = part.getAttribute('multiplier')
                if x <> '': p.multiplier = float(x)
                x = part.getAttribute('exponent')
                if x <> '': p.exponent = float(x)
                unit.parts.append(p)
            return unit
        # Parse units in model
        munits = []
        tag = dom_child(model_tag, 'units')
        while tag:
            munits.append(parse(tag))
            tag = dom_next(tag, 'units')
        # Parse units in components
        cunits = {}
        for tag in model_tag.getElementsByTagName('component'):
            cunits[tag.getAttribute('name')] = units = []
            for unit in tag.getElementsByTagName('units'):
                units.append(parse(unit))
        # Order units (units can refer to each other in a DAG form)
        def order(units, global_units=None):
            """
            Orders a list of (name, parts) tuples so that none of the parts
            refer to a unit defined later in the list. Returns an odict mapping
            names to (name, parts) tuples.
            """
            todo = units
            units = odict()
            # List units that can already be referenced at this point
            okay = si_units.keys()
            if global_units:
                for name, unit in global_units:
                    okay.append(name)
            # Run through todo list
            while todo:
                done = []
                for unit in todo:
                    ok = True
                    for part in unit.parts:
                        if part.base not in okay:
                            ok = False
                            break
                    if ok:
                        done.append(unit)
                        okay.append(unit.name)
                for unit in done:
                    units[unit.name] = unit
                    todo.remove(unit)
                if len(done) == 0:
                    break
            if todo:
                # Unable to resolve all units
                for unit in todo:
                    self.warn('Unable to resolve unit: ' + str(unit.name))
                    units[unit.name] = unit
            return units
        munits = order(munits)
        for name, units in cunits.iteritems():
            cunits[name] = order(units)
        # Convert units
        def convert(obj, local_map=None):
            """
            Converts a Unit object to a myokit unit.
            """
            base = myokit.units.dimensionless
            for part in obj.parts:
                # Get simple unit
                if str(part.base) in si_units:
                    unit = si_units[part.base]
                elif part.base in munits:
                    unit = munits[part.base]
                elif local_map and part.base in local_map:
                    unit = local_map[part.base]
                else:
                    self.warn('Unknown base unit: ' + str(part.base))
                    return None
                # Add prefix
                if part.prefix is not None:
                    try:
                        unit *= si_prefixes[part.prefix]
                    except KeyError:
                        try:
                            if str(part.prefix) == str(int(part.prefix)):
                                unit *= 10 ** int(part.prefix)
                            else:
                                raise ValueError
                        except ValueError:
                            self.warn('Unknown prefix in unit specification: "'
                                + str(part.prefix) + '".')
                            return None
                # Exponent (prefix part is exponentiated, multiplier is not)
                if part.exponent is not None:
                    e = int(part.exponent)
                    if e - part.exponent > 1e-15:
                        self.warn('Non-integer exponents in unit specification'
                            + ' are not supported: ' + str(part.exponent))
                        return None
                    unit **= e
                # Multiplier
                if part.multiplier is not None:
                    unit *= part.multiplier
                # Multiply base unit with this one
                base *= unit
            self.log('Converted unit "' + obj.name + '" to ' + str(base))
            return base
        # Convert all units in <model>
        for name, obj in munits.iteritems():
            munits[name] = convert(obj)
        # Convert all units in components
        for cname, units in cunits.iteritems():
            for name, obj in units.iteritems():
                units[name] = convert(obj, units)
        # Return unit maps
        return si_units, munits, cunits
    def _sanitise_name(self, name):
        """
        Tests if a name is a valid myokit name. Adapts it if it isn't.
        """
        name = str(name.encode('ascii', errors='replace'))
        try:
            myokit.check_name(name)
            return name
        except myokit.InvalidNameError as e:
            try:
                return self._generated_names[name]
            except KeyError:
                self.warn('Invalid name: ' + e.message)
                clean = 'generated_name_' + str(1 + len(self._generated_names))
                self._generated_names[name] = clean
                return clean
    def supports_model(self):
        """
        Returns True.
        """
        return True
