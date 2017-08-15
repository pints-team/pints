#
# Exports to CellML
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
import myokit
import myokit.units
import myokit.formats
import xml.etree.cElementTree as et
class CellMLExporter(myokit.formats.Exporter):
    """
    This :class:`Exporter <myokit.formats.Exporter>` creates a CellML model.
    """
    def __init__(self):
        super(CellMLExporter, self).__init__()
    def custom_unit_name(self, unit):
        """
        Creates an almost readable name for a custom Myokit unit.
        """
        # Get name, strip brackets
        name = str(unit)[1:-1]
        # Split unit from multiplier part
        if ' ' in name:
            name, multiplier = name.split(' ')
        else:
            name, multiplier = name, ''
        # Treat unit parts
        name = name.replace('^', '')
        name = name.replace('/', '_per_')
        name = name.replace('*', '_')
        if name[:2] == '1_':
            name = name[2:]
        # Add multiplier (if any)
        if multiplier:
            # Strip brackets
            multiplier = multiplier[1:-1]
            # Remove characters not allowed in CellML identifiers
            multiplier = multiplier.replace('-', '_minus_')
            name += '_times_' + multiplier
        return name
    def info(self):
        import inspect
        return inspect.getdoc(self)
    def model(self, path, model, protocol=None, add_hardcoded_pacing=True,
            pretty_xml=True):
        """
        Writes a CellML model to the given filename.
        
        Arguments:
        
        ``path``
            The path/filename to write the generated code too.
        ``model``
            The model to export
        ``protocol``
            This argument will be ignored: protocols are not supported by
            CellML.
        ``add_hardcoded_pacing``
            Set this to ``True`` to add a hardcoded pacing signal to the model
            file. This requires the model to have a variable bound to `pace`.
        ``pretty_xml``
            Set this to ``True`` to write the output in formatted "pretty"
            xml.

        Notes about CellML export:
        
        * CellML expects a unit for every number present in the model. Since
          Myokit allows but does not enforce this, the resulting CellML file
          may only validate with unit checking disabled.
        * Files downloaded from the CellML repository typically have a pacing
          stimulus embedded in them, while Myokit views models and pacing
          protocols as separate things. To generate a model file with a simple
          embbeded protocol, add the optional argument
          ``add_hardcoded_pacing=True``.
        
        """
        path = os.path.abspath(os.path.expanduser(path))
        import myokit.formats.cellml as cellml
        # Replace the pacing variable with a hardcoded stimulus protocol
        if add_hardcoded_pacing:
            # Check for pacing variable
            if model.binding('pace') is None:
                self.warn('No variable bound to "pace", unable to add'
                    ' hardcoded stimulus protocol.')
            else:
                # Clone model before changes
                model = model.clone()
                # Get pacing variable and time
                time = model.time()
                pace = model.binding('pace')
                # Set basic properties for pace
                pace.set_unit(myokit.units.dimensionless)
                pace.set_rhs(0)
                pace.set_binding(None)
                pace.set_label(None) # Should already be true...
                # Get time unit
                time_unit = time.unit()
                if time_unit is None:
                    time_unit = myokit.units.ms
                # Scale if seconds are used, in all other cases, assume ms
                time_factor = 1
                if time_unit == myokit.units.s:
                    time_factor = 0.001
                # Remove any child variables pace might have
                for kid in pace.variables():
                    pace.remove_variable(kid, recursive=True)
                # Create new component for the pacing variables
                component = 'stimulus'
                if model.has_component(component):
                    root = component
                    number = 1
                    while model.has_component(component):
                        number += 1
                        component = root + '_' + str(number)
                component = model.add_component(component)
                # Move pace. This will be ok any references: since pace was
                # bound it cannot be a nested variable.
                # While moving, update its name to avoid conflicts with the
                # hardcoded names.
                pace.parent().move_variable(pace, component, new_name='pace')
                # Add variables defining pacing protocol
                period = component.add_variable('period')
                period.set_unit(time_unit)
                period.set_rhs(str(1000 * time_factor) + ' ' + str(time_unit))
                offset = component.add_variable('offset')
                offset.set_unit(time_unit)
                offset.set_rhs(str(100 * time_factor) + ' ' + str(time_unit))
                duration = component.add_variable('duration')
                duration.set_unit(time_unit)
                duration.set_rhs(str(2 * time_factor) + ' ' + str(time_unit))
                # Add corrected time variable
                ctime = component.add_variable('ctime')
                ctime.set_unit(time_unit)
                ctime.set_rhs(time.qname()
                    + ' - floor(' + time.qname() + ' / period) * period')
                # Set new RHS for pace
                pace.set_rhs(
                    'if(ctime >= offset and ctime < offset + duration, 1, 0)')
        # Validate model
        model.validate()
        # Get time variable
        time = model.time()
        # Create model xml element
        emodel = et.Element('model')
        emodel.attrib['xmlns'] = 'http://www.cellml.org/cellml/1.0#'
        emodel.attrib['xmlns:cellml'] = 'http://www.cellml.org/cellml/1.0#'
        emodel.attrib['name'] = 'generated_model'
        if 'name' in model.meta:
            dtag = et.SubElement(emodel, 'documentation')
            dtag.attrib['xmlns'] = 'http://cellml.org/tmp-documentation'
            atag = et.SubElement(dtag, 'article')
            ttag = et.SubElement(atag, 'title')
            ttag.text = model.meta['name']
        # Add custom units, create unit map
        exp_si = [si_units[x] for x in myokit.Unit.list_exponents()]
        unit_map = {} # Add si units later
        def add_unit(unit):
            """
            Checks if the given unit needs to be added to the list of custom
            units and adds it if necessary.
            """
            # Check if already defined
            if unit is None or unit in unit_map or unit in si_units:
                return
            # Create unit name
            name = self.custom_unit_name(unit)
            # Create unit tag
            utag = et.SubElement(emodel, 'units')
            utag.attrib['name'] = name
            # Add part for each of the 7 SI units
            m = unit.multiplier()
            for k, e in enumerate(unit.exponents()):
                if e != 0:
                    tag = et.SubElement(utag, 'unit')
                    tag.attrib['units'] = exp_si[k]
                    tag.attrib['exponent'] = str(e)
                    if m != 1:
                        tag.attrib['multiplier'] = str(m)
                        m = 1
            # Or... if the unit doesn't contain any of those seven, it must be
            # a dimensionless unit with a multiplier. These occur in CellML
            # definitions when unit mismatches are "resolved" by adding
            # conversion factors as units. This has no impact on the actual
            # equations...
            if m != 1:
                tag = et.SubElement(utag, 'unit')
                tag.attrib['units'] = si_units[myokit.units.dimensionless]
                tag.attrib['exponent'] = str(1)
                tag.attrib['multiplier'] = str(m)
                m = 1
            # Add the new unit to the list
            unit_map[unit] = name
        # Add variable and expression units
        for var in model.variables(deep=True):
            add_unit(var.unit())
            for e in var.rhs().walk(myokit.Number):
                add_unit(e.unit())
        # Add si units to unit map
        for unit, name in si_units.iteritems():
            unit_map[unit] = name
        # Add components
        #TODO: Order to components
        # Components can correspond to Myokit components or variables with
        # children!
        ecomps = {} # Components/Variables : elements (tags)
        cnames = {} # Components/Variables : names (strings)
        def uname(name):
            # Create a unique component name
            i = 1
            r = name + '-'
            while name in cnames:
                i += 1
                name = r + str(i)
            return name
        def export_nested_var(parent_tag, parent_name, var):
            # Create unique component name
            cname = uname(parent_name + '_' + var.uname())
            cnames[var] = cname
            # Create element
            ecomp = et.SubElement(emodel, 'component')
            ecomp.attrib['name'] = cname
            ecomps[var] = ecomp
            # Check for nested variables with children
            for kid in var.variables():
                if kid.has_variables():
                    export_nested_var(ecomp, cname, kid)
        for comp in model.components():
            # Create unique name
            cname = uname(comp.name())
            cnames[comp] = cname
            # Create element
            ecomp = et.SubElement(emodel, 'component')
            ecomp.attrib['name'] = cname
            ecomps[comp] = ecomp
            # Check for variables with children
            for var in comp.variables():
                if var.has_variables():
                    export_nested_var(ecomp, cname, var)
        # Add variables
        evars = {}
        for parent, eparent in ecomps.iteritems():
            for var in parent.variables():
                evar = et.SubElement(eparent, 'variable')
                evars[var] = evar
                evar.attrib['name'] = var.uname()
                # Add units
                unit = var.unit()
                unit = unit_map[unit] if unit else 'dimensionless'
                evar.attrib['units'] = unit
                # Add initial value
                init = None
                if var.is_literal():
                    init = var.rhs().eval()
                elif var.is_state():
                    init = var.state_value()
                if init is not None:
                    evar.attrib['initial_value'] = myokit.strfloat(init)
        # Add variable interfaces, connections
        deps = model.map_shallow_dependencies(omit_states=False,
            omit_constants=False)
        for var, evar in evars.iteritems():
            # Scan all variables, iterate over the vars they depend on
            par = var.parent()
            lhs = var.lhs()
            dps = set(deps[lhs])
            if var.is_state():
                # States also depend on the time variable
                dps.add(time.lhs())
            for dls in dps:
                dep = dls.var()
                dpa = dep.parent()
                # Parent mismatch: requires connection
                if par != dpa:
                    # Check if variable tag is present
                    epar = ecomps[par]
                    tag = epar.find('variable[@name="'+dep.uname()+'"]')
                    if tag is None:
                        # Create variable tag
                        tag = et.SubElement(epar, 'variable')
                        tag.attrib['name'] = dep.uname()
                        # Add unit
                        unit = dep.unit()
                        unit = unit_map[unit] if unit else 'dimensionless'
                        tag.attrib['units'] = unit
                        # Set interfaces
                        tag.attrib['public_interface'] = 'in'
                        edpa = ecomps[dpa]
                        tag = edpa.find('variable[@name="' + dep.uname()
                            + '"]')
                        tag.attrib['public_interface'] = 'out'
                        # Add connection for this variable
                        comp1 = cnames[par]
                        comp2 = cnames[dpa]
                        vname = dep.uname()
                        # Sort components in connection alphabetically to
                        # ensure uniqueness
                        if comp2 < comp1:
                            comp1, comp2 = comp2, comp1
                        # Find or create connection
                        ctag = None
                        for con in emodel.findall('connection'):
                            ctag = con.find('map_components[@component_1="'
                                + comp1 + '"][@component_2="' + comp2 + '"]')
                            if ctag is not None:
                                break
                        if ctag is None:
                            con = et.SubElement(emodel, 'connection')
                            ctag = et.SubElement(con, 'map_components')
                            ctag.attrib['component_1'] = comp1
                            ctag.attrib['component_2'] = comp2
                        vtag = con.find('map_variables[@variable_1="' + vname
                            + '"][variable_2="' + vname + '"]')
                        if vtag is None:
                            vtag = et.SubElement(con, 'map_variables')
                            vtag.attrib['variable_1'] = vname
                            vtag.attrib['variable_2'] = vname
        # Create CellMLWriter
        writer = cellml.CellMLExpressionWriter(units=unit_map)
        writer.set_element_tree_class(et)        
        writer.set_time_variable(time)
        # Add equations
        def add_child_equations(parent):
            # Add the equations to a cellml component
            try:
                ecomp = ecomps[parent]
            except KeyError:
                return
            maths = et.SubElement(ecomp, 'math')
            maths.attrib['xmlns'] = 'http://www.w3.org/1998/Math/MathML'
            for var in parent.variables():
                if var.is_literal() or var == time:
                    continue
                writer.eq(var.eq(), maths)
                add_child_equations(var)
        for comp in model.components():
            add_child_equations(comp)
        # Write xml to file
        doc = et.ElementTree(emodel)
        doc.write(path, encoding='utf-8', method='xml')
        if pretty_xml:
            # Create pretty XML
            import xml.dom.minidom as m
            xml = m.parse(path)
            with open(path, 'w') as f:
                f.write(xml.toprettyxml(encoding='utf-8'))
        # Log any generated warnings
        self.log_warnings()
    def supports_model(self):
        """
        Returns ``True``.
        """
        return True
#class CellMLExportModel(object):
#    """
#    CellML Model class used only for exporting.
#    """
#    def __init__(self, myokit_model):
#        pass
# SI Units supported by CellML
si_units = {
    myokit.units.dimensionless : 'dimensionless',
    myokit.units.A   : 'ampere',
    myokit.units.F   : 'farad',
    myokit.units.kat : 'katal',
    myokit.units.lux : 'lux',
    myokit.units.Pa  : 'pascal',
    myokit.units.T   : 'tesla',
    myokit.units.Bq  : 'becquerel',
    myokit.units.g   : 'gram',
    myokit.units.K   : 'kelvin',
    myokit.units.m   : 'meter',
    myokit.units.V   : 'volt',
    myokit.units.cd  : 'candela',
    myokit.units.Gy  : 'gray',
    myokit.units.kg  : 'kilogram',
    myokit.units.m   : 'metre',
    myokit.units.s   : 'second',
    myokit.units.W   : 'watt',
    myokit.units.C   : 'celsius',
    myokit.units.H   : 'henry',
    myokit.units.L   : 'liter',
    myokit.units.mol : 'mole',
    #myokit.units.rad : 'radian',   # this overwrites dimensionless
}
si_exponents = {
    -24 : 'yocto',
    -21 : 'zepto',
    -18 : 'atto',
    -15 : 'femto',
    -12 : 'pico',
     -9 : 'nano',
     -6 : 'micro',
     -3 : 'milli',
     -2 : 'centi',
     -1 : 'deci',
      1 : 'deka',
      2 : 'hecto',
      3 : 'kilo',
      6 : 'mega',
      9 : 'giga',
     12 : 'tera',
     15 : 'peta',
     18 : 'exa',
     21 : 'zetta',
     24 : 'yotta',
    }
