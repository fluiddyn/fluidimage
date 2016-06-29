# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:35:12 2016

@author: campagne8a
"""

import re
import xml.sax.handler
import numpy as np
from fluiddyn.util.paramcontainer import ParamContainer, tidy_container


class DataObject(object):
    pass


def get_number_from_string(string):
    return map(float, re.findall(r"[-+]?\d*\.\d+|\d+", string))


def calib_parameters_from_uvmat(fic):
    calib = ParamContainer(path_file=fic)
    tidy_container(calib)

    GeometryCalib = calib['geometry_calib']

    calib_param = DataObject()
    calib_param.f = map(
        float, np.asarray(get_number_from_string(GeometryCalib['fx_fy'])))
    calib_param.C = np.asarray(get_number_from_string(GeometryCalib['cx__cy']))
    calib_param.kc = np.asarray(GeometryCalib['kc'])
    calib_param.T = np.asarray(
        get_number_from_string(GeometryCalib['tx__ty__tz']))

    calib_param.R = []
    for i in range(3):
        calib_param.R = np.hstack([
            calib_param.R,
            get_number_from_string(GeometryCalib['r_{}'.format(i+1)])])

    calib_param.omc = np.asarray(get_number_from_string(GeometryCalib['omc']))

    if GeometryCalib['nb_slice'] is not None:
        slices = DataObject()

        slices.nbSlice = np.asarray(GeometryCalib['nb_slice'])
        slices.zsliceCoord = np.zeros([slices.nbSlice, 3])

        for i in range(slices.nbSlice):
            slices.zsliceCoord[i][:] = get_number_from_string(
                GeometryCalib['slice_coord_{}'.format(i+1)])

        if GeometryCalib['slice_angle_1'] is not None:
            slices.sliceAngle = np.zeros([slices.nbSlice, 3])
            for i in range(slices.nbSlice):
                slices.sliceAngle[i][:] = get_number_from_string(
                    GeometryCalib['slice_angle_{}'.format(i+1)])

        calib_param.slices = slices

    return calib_param


def xml2obj(src):
    """
    A simple function to converts XML data into native Python object.
    """

    non_id_char = re.compile('[^_0-9a-zA-Z]')

    def _name_mangle(name):
        return non_id_char.sub('_', name)

    class DataNode(object):
        def __init__(self):
            self._attrs = {}    # XML attributes and child elements
            self.data = None    # child text data

        def __len__(self):
            # treat single element as a list of 1
            return 1

        def __getitem__(self, key):
            if isinstance(key, basestring):
                return self._attrs.get(key, None)
            else:
                return [self][key]

        def __contains__(self, name):
            return name in self._attrs

        def __nonzero__(self):
            return bool(self._attrs or self.data)

        def __getattr__(self, name):
            if name.startswith('__'):
                # need to do this for Python special methods???
                raise AttributeError(name)
            return self._attrs.get(name, None)

        def _add_xml_attr(self, name, value):
            if name in self._attrs:
                # multiple attribute of the same name are represented by a list
                children = self._attrs[name]
                if not isinstance(children, list):
                    children = [children]
                    self._attrs[name] = children
                children.append(value)
            else:
                self._attrs[name] = value

        def __str__(self):
            return self.data or ''

        def __repr__(self):
            items = sorted(self._attrs.items())
            if self.data:
                items.append(('data', self.data))
            return u'{%s}' % ', '.join([u'%s:%s' % (
                k, repr(v)) for k, v in items])

    class TreeBuilder(xml.sax.handler.ContentHandler):
        def __init__(self):
            self.stack = []
            self.root = DataNode()
            self.current = self.root
            self.text_parts = []

        def startElement(self, name, attrs):
            self.stack.append((self.current, self.text_parts))
            self.current = DataNode()
            self.text_parts = []
            # xml attributes --> python attributes
            for k, v in attrs.items():
                self.current._add_xml_attr(_name_mangle(k), v)

        def endElement(self, name):
            text = ''.join(self.text_parts).strip()
            if text:
                self.current.data = text
            if self.current._attrs:
                obj = self.current
            else:
                # a text only node is simply represented by the string
                obj = text or ''
            self.current, self.text_parts = self.stack.pop()
            self.current._add_xml_attr(_name_mangle(name), obj)

        def characters(self, content):
            self.text_parts.append(content)

    builder = TreeBuilder()
    if isinstance(src, basestring):
        xml.sax.parseString(src, builder)
    else:
        xml.sax.parse(src, builder)
    return builder.root._attrs.values()[0]
