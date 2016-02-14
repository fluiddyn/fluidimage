
from .display2 import display2


def common_start(sa, sb):
    """Returns the longest common substring from the beginning of 2 strings."""
    def _iter():
        for a, b in zip(sa, sb):
            if a == b:
                yield a
            else:
                return

    return ''.join(_iter())


class DataObject(object):
    pass


class ArrayCouple(DataObject):
    def __init__(self, names, arrays):
        self.names = tuple(names)
        self.arrays = tuple(arrays)

    def get_arrays(self):
        return self.arrays


class HeavyPIVResults(DataObject):
    def __init__(self, deltaxs, deltays, xs, ys, correls, couple):
        self.deltaxs = deltaxs
        self.deltays = deltays
        self.correls = correls
        self.couple = couple
        self.ys = ys
        self.xs = xs

    def get_images(self):
        return self.couple.get_arrays()

    def display(self):
        im0, im1 = self.couple.get_arrays()
        display2(
            im0, im1, self.xs, self.ys,
            self.deltaxs, self.deltays, self.correls)

    def save(self, path):

        base_name = common_start(*self.couple.names)
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
