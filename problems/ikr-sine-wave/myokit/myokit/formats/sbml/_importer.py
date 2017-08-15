#
# Imports a model from an SBML file.
# Only partial SBML support (Based on SBML level 3) is provided.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
# Python imports
from collections import OrderedDict as odict
import xml.dom.minidom
import shutil
import os
import re
# Myokit import
import myokit
import myokit.units
import myokit.formats
from myokit.mxml import html2ascii
from myokit.mxml import dom_child, dom_next
from myokit.formats.mathml import parse_mathml_rhs
# Global vars, general importer stuff
info = \
"""
Loads a Model definition from an SBML file. Warning: This importer hasn't been
extensively tested.
"""
# The main class
class SBMLImporter(myokit.formats.Importer):
    """
    This :class:`Importer <myokit.formats.Importer>` load model definitions
    from files in SBML format.
    """
    def __init__(self):
        super(SBMLImporter, self).__init__()
        self.re_name = re.compile(r'^[a-zA-Z]+[a-zA-Z0-9_]*$')
        self.re_alpha = re.compile(r'[\W]+')
        self.re_white = re.compile(r'[ \t\f\n\r]+')
        self.units = {}
    def _convert_name(self, name):
        """
        Converts a name to something acceptable by myokit.
        """
        if not self.re_name.match(name):
            org_name = name
            name = self.re_white.sub('_', name)
            name = self.re_alpha.sub('_', name)
            if not self.re_name.match(name):
                name = 'x_' + name2
            self.warn('Converting name <' + org_name + '> to <' + name + '>.')
        return name
    def _convert_unit(self, unit):
        """
        Converts an SBML unit to a myokit one using the lookup tables generated
        when parsing the XML file.
        """
        if unit in self.units:
            return self.units[unit]
        elif unit in unit_map:
            return unit_map[unit]
        else:
           raise SBMLError('Unit not recognized: ' + str(unit))
    def info(self):
        return info
    def model(self, path):
        # Parse xml file
        path = os.path.abspath(os.path.expanduser(path))
        self.log('Reading ' + str(path))
        dom = xml.dom.minidom.parse(path)
        xmodel = dom.getElementsByTagName('model')[0]
        # Get model node
        if xmodel.getAttribute('name'):
            name = str(xmodel.getAttribute('name'))
        elif xmodel.getAttribute('id'):
            name = str(xmodel.getAttribute('id'))
        else:
            name = 'Imported SBML model'
        # Create myokit model
        model = myokit.Model(self._convert_name(name))
        self.log('Reading model "' + model.meta['name'] + '"')
        # Create one giant component to hold all variables
        comp = model.add_component('sbml')
        # Handle notes, if given
        x = dom_child(xmodel, 'notes')
        if x:
            self.log('Converting <model> notes to ascii')
            model.meta['desc'] = html2ascii(x.toxml(), width=75) #79-4 for tab!
        # Warn about missing functionality
        x = dom_child(xmodel, 'listOfCompartments')
        if x: self.warn('Compartments are not supported.')
        x = dom_child(xmodel, 'listOfSpecies')
        if x: self.warn('Species are not supported.')
        x = dom_child(xmodel, 'listOfConstraints')
        if x: self.warn('Constraints are not supported.')
        x = dom_child(xmodel, 'listOfReactions')
        if x: self.warn('Reactions are not supported.')
        x = dom_child(xmodel, 'listOfEvents')
        if x: self.warn('Events are not supported.')
        # Handle custom functions TODO???
        x = dom_child(xmodel, 'listOfFunctionDefinitions')
        if x: self.warn('Custom math functions are not (yet) implemented.')
        # Parse custom units
        x = dom_child(xmodel, 'listOfUnitDefinitions')
        if x: self._parse_units(model, comp, x)
        # Parse parameters (constants + parameters)
        x = dom_child(xmodel, 'listOfParameters')
        if x: self._parse_parameters(model, comp, x)
        # Parse rules (equations)
        x = dom_child(xmodel, 'listOfRules')
        if x: self._parse_rules(model, comp, x)
        # Parse extra initial assignments
        x = dom_child(xmodel, 'listOfInitialAssignments')
        if x: self._parse_initial_assignments(model, comp, x)
        # Write warnings to log
        self.log_warnings()
        # Run model validation, order variables etc
        try:
            model.validate()
        except myokit.IntegrityError as e:
            self.log_line()
            self.log('WARNING: Integrity error found in model:')
            self.log(e.message)
            self.log_line()
        # Return finished model
        return model
    def _parse_initial_assignments(self, model, comp, node):
        """
        Parses any initial values specified outside of the rules section.
        """
        node = dom_child(node, 'initialAssignment')
        while node:
            var = str(node.getAttribute('symbol')).strip()
            var = self._convert_name(var)
            if var in comp:
                self.log('Parsing initial assignment for "' + var + '".')
                var = comp[var]
                expr = parse_mathml_rhs(dom_child(node, 'math'), comp, self)
                if var.is_state():
                    # Initial value
                    var.set_state_value(expr, default=True)
                else:
                    # Change of value
                    var.set_rhs(expr)
            else:
                raise SBMLError('Initial assignment found for unknown'
                    ' parameter <' + var + '>.')
            node = dom_next(node, 'initialAssignment')
    def _parse_parameters(self, model, comp, node):
        """
        Parses parameters
        """
        node = dom_child(node, 'parameter')
        while node:
            # Create variable
            name = self._convert_name(str(node.getAttribute('id')))
            self.log('Found parameter "' + name + '"')
            if name in comp:
                self.warn('Skipping duplicate parameter name: ' + str(name))
            else:
                # Create variable
                unit = None
                if node.hasAttribute('units'):
                    foreign_unit = node.getAttribute('units')
                    if foreign_unit:
                        unit = self._convert_unit(foreign_unit)
                value = None
                if node.hasAttribute('value'):
                    value = node.getAttribute('value')
                var = comp.add_variable(name)
                var.set_unit(unit)
                var.set_rhs(value)
            node = dom_next(node, 'parameter')
    def _parse_rules(self, model, comp, node):
        """
        Parses the rules (equations) in this model
        """
        parent = node
        formulas = {}
        # Create variables with assignment rules (all except derivatives)
        node = dom_child(parent, 'assignmentRule')
        while node:
            var = self._convert_name(str(node.getAttribute('variable')).strip())
            if var in comp:
                self.log('Parsing assignment rule for <' + str(var) + '>.')
                var = comp[var]
                var.set_rhs(parse_mathml_rhs(
                    dom_child(node, 'math'), comp, self))
            else:
                raise SBMLError('Assignment found for unknown parameter: "'
                    + var + '".')
            node = dom_next(node, 'assignmentRule')
        # Create variables with rate rules (states)
        node = dom_child(parent, 'rateRule')
        while node:
            var = self._convert_name(str(node.getAttribute('variable')).strip())
            if var in comp:
                self.log('Parsing rate rule for <' + var + '>.')
                var = comp[var]
                ini = var.rhs()
                ini = ini.eval() if ini else 0
                var.promote(ini)
                var.set_rhs(parse_mathml_rhs(
                    dom_child(node, 'math'), comp, self))
            else:
                raise SBMLError('Derivative found for unknown parameter: <'
                    + var + '>.')
            node = dom_next(node, 'rateRule')
    def _parse_units(self, model, comp, node):
        """
        Parses custom unit definitions, creating a look-up table that can be
        used to convert these units to myokit ones.
        """
        node = dom_child(node, 'unitDefinition')
        while node:
            name = node.getAttribute('id')
            self.log('Parsing unit definition for "' + name + '".')
            unit = myokit.units.dimensionless
            node2 = dom_child(node, 'listOfUnits')
            node2 = dom_child(node2, 'unit')
            while node2:
                kind = str(node2.getAttribute('kind')).strip()
                u2 = self._convert_unit(kind)
                if node2.hasAttribute('multiplier'):
                    m = float(node2.getAttribute('multiplier'))
                else:
                    m = 1.0
                if node2.hasAttribute('scale'):
                    m *= 10 ** float(node2.getAttribute('scale'))
                u2 *= m
                if node2.hasAttribute('exponent'):
                    u2 **= float(node2.getAttribute('exponent'))
                unit *= u2
                node2 = dom_next(node2, 'unit')
            self.units[name] = unit
            node = dom_next(node, 'unitDefinition')
    def supports_model(self):
        return True
class SBMLError(myokit.ImportError):
    """
    Thrown if an error occurs when importing SBML
    """
unit_map = {
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
