#
# Exports to MathML based formats.
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
import xml.etree.cElementTree as et
from _ewriter import MathMLExpressionWriter
class XMLExporter(myokit.formats.Exporter):
    """
    This :class:`Exporter <myokit.formats.Exporter>` generates an XML file
    containing a model's equations, encoded in Content MathML. This is an XML
    format containing the bare equations, without any formatting. It can be 
    used to exchange equations with MathML supporting applications.
    """
    def info(self):
        import inspect
        return inspect.getdoc(self)
    def model(self, path, model, protocol=None):
        """
        Export the model to an xml document.
        """
        path = os.path.abspath(os.path.expanduser(path))
        # Create model xml element
        root = et.Element('math')
        root.attrib['xmlns'] = 'http://www.w3.org/1998/Math/MathML'
        # Create expression writer
        writer = MathMLExpressionWriter()
        writer.set_element_tree_class(et)
        writer.set_mode(presentation=False)
        writer.set_time_variable(model.time())            
        # Write equations
        for var in model.variables(deep=True):
            writer.eq(var.eq(), root)
        # Write xml to file
        doc = et.ElementTree(root)
        doc.write(path, encoding='utf-8', method='xml')
        if True:
            # Pretty output
            import xml.dom.minidom as m
            xml = m.parse(path)
            with open(path, 'w') as f:
                f.write(xml.toprettyxml(encoding='utf-8'))
    def supports_model(self):
        """
        Returns ``True``.
        """
        return True
class HTMLExporter(myokit.formats.Exporter):
    """
    This :class:`Exporter <myokit.formats.Exporter>` generates a HTML file
    displaying a model's equations. The equations are encoded using 
    Presentation MathML. This format can be viewed in most modern browsers, but
    is less suitable as an exchange format.
    """
    def info(self):
        import inspect
        return inspect.getdoc(self)
    def model(self, path, model, protocol=None):
        """
        Export to a html document.
        """
        # Get model name
        try:
            name = model.meta['name']
        except KeyError:
            name = 'Generated model'
        # Create model html element
        html = et.Element('html')
        head = et.SubElement(html, 'head')
        title = et.SubElement(head, 'title')
        title.text = name
        body = et.SubElement(html, 'body')
        heading = et.SubElement(body, 'h1')
        heading.text = name
        # Create expression writer
        writer = MathMLExpressionWriter()
        writer.set_element_tree_class(et)
        writer.set_mode(presentation=True)
        writer.set_time_variable(model.time())            
        # Write equations, per component
        for component in model.components():
            div = et.SubElement(body, 'div')
            div.attrib['class'] = 'component'
            heading = et.SubElement(div, 'h2')
            heading.text = component.qname()
            for var in component.variables(deep=True):
                div2 = et.SubElement(div, 'div')
                div2.attrib['class'] = 'variable'
                math = et.SubElement(div2, 'math')
                math.attrib['xmlns'] = 'http://www.w3.org/1998/Math/MathML'    
                writer.eq(var.eq(), math)
        # Write xml to file
        doc = et.ElementTree(html)
        doc.write(path, encoding='utf-8', method='xml')
        if True:
            # Pretty output
            import xml.dom.minidom as m
            xml = m.parse(path)
            with open(path, 'w') as f:
                f.write(xml.toprettyxml(encoding='utf-8'))
    def supports_model(self):
        """
        Returns ``True``.
        """
        return True
