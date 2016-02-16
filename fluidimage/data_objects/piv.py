
import os

import h5py

from .display2 import display2


class DataObject(object):
    pass


class ArrayCouple(DataObject):
    def __init__(self, names, arrays, serie=None):
        self.names = tuple(names)
        self.arrays = tuple(arrays)
        self.serie = serie

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
        serie = self.couple.serie

        str_ind0 = serie._compute_strindices_from_indices(
            *[inds[0] for inds in serie.get_index_slices()])

        str_ind1 = serie._compute_strindices_from_indices(
            *[inds[1] - 1 for inds in serie.get_index_slices()])

        name = ('piv_' + serie.base_name + str_ind0 + '-' + str_ind1 + '.h5')

        path_file = os.path.join(path, name)
        print(path_file)

        keys_to_be_saved = ['xs', 'ys', 'deltaxs', 'deltays']
        with h5py.File(path_file, 'w') as f:
            for k in keys_to_be_saved:
                f.create_dataset(k, data=self.__dict__[k])

        return self

    def load(self, path):
        raise NotImplementedError
