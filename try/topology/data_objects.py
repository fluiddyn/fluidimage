
from __future__ import print_function


class DataObject(object):
    pass


class ArrayImage(DataObject):
    pass


class ArrayCouple(DataObject):
    def __init__(self, names, arrays):
        self.names = names
        self.arrays = arrays


class HeavyPIVResult(DataObject):
    def __init__(self, deltax, deltay):
        self.deltax = deltax
        self.deltay = deltay

    def save(self, path):
        print('save in path:\n', path)
        return self


class StructuredVelocityField(DataObject):
    pass
