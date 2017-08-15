#
# Imports a channel model from a ChannelML file.
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
import textwrap
from xml.dom import minidom
from cStringIO import StringIO
import myokit
from myokit import formats
from myokit import Name, Number, Minus, Multiply, Divide, Power
from myokit.mxml import dom_child, dom_next
class ChannelMLError(myokit.ImportError):
    """
    Raised if a fatal error occurs when importing ChannelML.
    """
class ChannelMLImporter(formats.Importer):
    """
    This :class:`Importer <myokit.formats.Importer>` imports model definitions
    from the ChannelML format.
    """
    def __init__(self, verbose=False):
        super(ChannelMLImporter, self).__init__()
        self.generated_name_index = 0
    def component(self, path, model):
        """
        Imports a channel component from the ChannelML file at ``path`` and
        adds it to the given model.
        
        The model **must** contain a variable labelled "membrane_potential".
        
        The created :class:`myokit.Component` is returned.
        """
        return self._parse(path, model)
    def info(self):
        return "Loads a channel model definition from a ChannelML file."
    def model(self, path):
        """
        Imports a ChannelML file from ``path`` and returns the channel as a
        :class:`myokit.Model`.
        """
        # Create model
        model = myokit.Model('channelml')
        # Add time variable
        c = model.add_component('engine')
        v = c.add_variable('time')
        v.set_rhs(0)
        v.set_binding('time')
        # Add membrane potential variable
        c = model.add_component('membrane')
        v = c.add_variable('v')
        v.set_rhs('-80')
        v.set_label('membrane_potential')
        # Add the channel component
        c = self._parse(path, model)
        # Return
        return model
    def _parse(self, path, model):
        """
        Parses a ChannelML channel and adds it to the given model.
        
        Returns the new :class:`myokit.Component`.
        """
        # Check model: get membrane potential varialbe
        vvar = model.label('membrane_potential')
        if vvar is None:
            raise ChannelMLError('No variable labelled "membrane_potential"'
                ' was found. This is required when adding ChannelML channels'
                ' to existing models.')
        # Parse XML
        path = os.path.abspath(os.path.expanduser(path))
        dom = minidom.parse(path)
        # Get channelml tag
        root = dom.getElementsByTagName('channelml')
        try:
            root = root[0]
        except IndexError:
            raise ChannelMLError('Unknown root element in xml document.'
                ' Expecting a tag of type <channelml>.')
        # Extract meta data
        meta = self._rip_meta(root)
        # Get channeltype tag
        root = root.getElementsByTagName('channel_type')
        try:
            root = root[0]
        except IndexError:
            raise ChannelMLError('No <channel_type> element found inside'
                ' <channelml> element. Import of <synapse_type> and'
                ' <ion_concentration> is not supported.')
        # Add channel component
        name = self._sanitise_name(root.getAttribute('name'))
        if name in model:
            root = name
            i = 2
            while name in model:
                name = root + '_' + str(i)
                i += 1
        component = model.add_component(name)
        # Add alias to membrane potential
        component.add_alias('v', vvar)
        # Add meta-data
        component.meta['desc'] = meta
        # Find current-voltage relation
        cvr = root.getElementsByTagName('current_voltage_relation')
        if len(cvr) < 1:
            raise ChannelMLError('Channel model must contain a current voltage'
                ' relation.')
        elif len(cvr) > 1:
            self.warn('Multiple current voltage relations found, ignoring all'
                ' but first.')
        cvr = cvr[0]
        # Check for q10
        try:
            q10 = cvr.getElementsByTagName('q10_settings')[0]        
            component.meta['experimental_temperature'] = str(
                q10.getAttribute('experimental_temp'))
        except IndexError:
            pass
        # Add reversal potential
        E = 0
        if cvr.hasAttribute('default_erev'):
            E = float(cvr.getAttribute('default_erev'))
        evar = component.add_variable('E')
        evar.meta['desc'] = 'Reversal potential'
        evar.set_rhs(E)
        # Get maximum conductance
        gmax = 1.0
        if cvr.hasAttribute('default_gmax'):
            gmax = float(cvr.getAttribute('default_gmax'))
        gmaxvar = component.add_variable('gmax')
        gmaxvar.set_rhs(gmax)
        gmaxvar.meta['desc'] = 'Maximum conductance'
        # Add gates
        gvars = []
        for gate in cvr.getElementsByTagName('gate'):
            gname = self._sanitise_name(gate.getAttribute('name'))
            gvar = component.add_variable(gname)
            gvar.promote(0)
            cstate = gate.getElementsByTagName('closed_state')
            cstate = cstate[0].getAttribute('id')
            ostate = gate.getElementsByTagName('open_state')
            ostate = ostate[0].getAttribute('id')
            # Transitions
            trans = gate.getElementsByTagName('transition')
            if len(trans) > 0:
                # Use "transitions" definition
                if len(trans) != 2:
                    raise ChannelMLError('Expecting exactly 2 transitions for'
                        ' gate <' + gname + '>.')
                # Get closed-to-open state
                tco = None
                for t in trans:
                    if (t.getAttribute('to') == ostate and 
                            t.getAttribute('from') == cstate):
                        tco = t
                        break
                if tco is None:
                    raise ChannelMLError('Unable to find closed-to-open'
                        ' transition for gate <' + gname + '>')
                # Get open-to-closed state
                toc = None
                for t in trans:
                    if (t.getAttribute('to') == cstate and
                            t.getAttribute('from') == ostate):
                        toc = t
                        break
                if toc is None:
                    raise ChannelMLError('Unable to find open-to-closed'
                        ' transition for gate <' + gname + '>')
                # Add closed-to-open transition
                tname = self._sanitise_name(tco.getAttribute('name'))
                tcovar = gvar.add_variable(tname)
                expr = str(tco.getAttribute('expr'))
                try:
                    tcovar.set_rhs(self._parse_expression(expr, tcovar))
                except myokit.ParseError as e:
                    self.warn('Error parsing expression for closed-to-open'
                        ' transition in gate <' + gname + '>: '
                        + myokit.format_parse_error(e))
                    tcovar.meta['expression'] = str(expr)
                # Add open-to-closed transition
                tname = self._sanitise_name(toc.getAttribute('name'))
                tocvar = gvar.add_variable(tname)
                expr = str(toc.getAttribute('expr'))
                try:
                    tocvar.set_rhs(self._parse_expression(expr, tocvar))
                except myokit.ParseError as e:
                    self.warn('Error parsing expression for open-to-closed'
                        ' transition in gate <' + gname + '>: '
                        + myokit.format_parse_error(e))
                    tocvar.meta['expression'] = str(expr)
                # Write equation for gate
                gvar.set_rhs(Minus(
                    Multiply(Name(tcovar), Minus(Number(1), Name(gvar))),
                    Multiply(Name(tocvar), Name(gvar))))
            else:
                # Use "steady-state & time_course" definition
                ss = gate.getElementsByTagName('steady_state')
                tc = gate.getElementsByTagName('time_course')
                if len(ss) < 1 or len(tc) < 1:
                    raise ChannelMLError('Unable to find transitions or'
                        ' steady state + time_course for gate <' + gname
                        + '>.')
                ss = ss[0]
                tc = tc[0]
                # Add steady-state variable
                ssname = self._sanitise_name(ss.getAttribute('name'))
                ssvar = gvar.add_variable(ssname)
                expr = str(ss.getAttribute('expr'))
                try:
                    ssvar.set_rhs(self._parse_expression(expr, ssvar))
                except myokit.ParseError as e:
                    self.warn('Error parsing expression for steady state in'
                        ' gate <' + gname + '>: '
                        + myokit.format_parse_error(e))
                    ssvar.meta['expression'] = str(expr)
                # Add time course variable
                tcname = self._sanitise_name(tc.getAttribute('name'))
                tcvar = gvar.add_variable(tcname)
                expr = str(tc.getAttribute('expr'))
                try:
                    tcvar.set_rhs(self._parse_expression(expr, tcvar))
                except myokit.ParseError as e:
                    self.warn('Error parsing expression for time course in'
                        ' gate <' + gname + '>: '
                        + myokit.format_parse_error(e))
                    tcvar.meta['expression'] = str(expr)
                # Write expression for gate
                gvar.set_rhs(Divide(Minus(Name(ssvar),Name(gvar)),Name(tcvar)))
            power = int(gate.getAttribute('instances'))
            if power > 1:
                gvars.append(Power(Name(gvar), Number(power)))
            else:
                gvars.append(Name(gvar))
        if len(gvars) < 1:
            raise ChannelMLError('Current voltage relation requires at least'
                ' one gate.')
        # Add current variable
        ivar = component.add_variable('I')
        ivar.meta['desc'] = 'Current'
        expr = Name(gmaxvar)
        while gvars:
            expr = Multiply(expr, gvars.pop())
        expr = Multiply(expr, Minus(Name(vvar), Name(evar)))
        ivar.set_rhs(expr)
        # Done, return component
        return component
    def _parse_expression(self, s, var=None):
        """
        Attempts to read an expression from the given string ``s`` and parse it
        using the context of variable ``var`` to resolve any references.
        """
        s = str(s)        
        # Pre-process some basic html entities
        entities = {
            '&eq;'  : '==',
            '&ne;'  : '!=',
            '&neq;' : '!=', # Not a html entity!
            '&lt;'  : '<',
            '&gt;'  : '>',
            '&le;'  : '<=',
            '&ge;'  : '>=',
            }
        s = s.replace('&amp;', '&')
        for k, v in entities.iteritems():
            s = s.replace(k, v)
        # Attempt to handle single a?b:c construct
        if '?' in s:
            c, v = s.split('?', 1)
            if ':' in v:
                v = v.split(':', 1)
            else:
                v = (v, '0')
            return myokit.If(
                myokit.parse_expression(c, var),
                myokit.parse_expression(v[0], var),
                myokit.parse_expression(v[1], var))
        return myokit.parse_expression(s, var)
    def _rip_meta(self, root):
        """
        Coarsely extracts any meta data from the model and returns it as a text
        file.
        """
        if not root.hasAttribute('xmlns:meta'):
            return None
        ns = root.getAttribute('xmlns:meta')
        b = StringIO()
        in_meta_tag = False
        def scan(parent):
            kid = dom_child(parent)
            while kid is not None:
                t = None
                if kid.namespaceURI == ns:
                    t = self._flatten(kid)
                if t:
                    b.write(t)
                    b.write('\n')
                else:   
                    scan(kid)
                kid = dom_next(kid)
        scan(root)
        return b.getvalue().strip()
    def _flatten(self, node):
        """
        Reduces a node's contents to flat text content and returns it.
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
                    buff.append(t)
            else:
                for kid in node.childNodes:
                    text(kid, buff)
            return buff
        return textwrap.fill(str('\n'.join(text(node))), 70,
            replace_whitespace=False)
    def _sanitise_name(self, name):
        """
        Tests if a name is a valid myokit name. Adapts it if it isn't.
        """
        name = str(name).strip()
        name = name.replace('.', '_')
        name = name.replace(' ', '_')
        name = name.replace('-', '_')
        try:
            myokit.check_name(name)
        except myokit.InvalidNameError as e:
            self.warn('Invalid name: ' + e.message)
            self.generated_name_index += 1
            name = 'generated_name_' + str(self.generated_name_index)
        return name
    def supports_component(self):
        return True
    def supports_model(self):
        return True
def tag(element):
    t = element.tag
    if '{' in t:
        return t[t.find('}')+1:]
    return t
