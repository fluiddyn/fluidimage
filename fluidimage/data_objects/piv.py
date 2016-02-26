
import os

import h5py

from fluiddyn.util.paramcontainer import ParamContainer

from .display import display
from .. import imread


class DataObject(object):
    pass


class ArrayCouple(DataObject):
    def __init__(
            self, names=None, arrays=None, serie=None,
            path_load=None, hdf5_parent=None):

        if path_load is not None:
            self._load(path=path_load)
            return

        if hdf5_parent is not None:
            self._load(hdf5_object=hdf5_parent['couple'])
            return

        if serie is not None and arrays is None:
            if serie.get_nb_files() != 2:
                raise ValueError('serie has to contain 2 arrays.')
            names = serie.get_name_files()
            paths = serie.get_path_files()
            arrays = serie.get_arrays()
            self.paths = tuple(paths)
        self.names = tuple(names)
        self.arrays = tuple(arrays)
        self.serie = serie

    def get_arrays(self):
        if not hasattr(self, 'arrays'):
            self.arrays = (imread(path) for path in self.paths)

        return self.arrays

    def save(self, path=None, hdf5_parent=None):
        if path is not None:
            raise NotImplementedError

        if not isinstance(hdf5_parent, (h5py.File,)):
            raise NotImplementedError

        hdf5_parent.create_group('couple')
        group = hdf5_parent['couple']
        group.attrs['names'] = self.names
        group.attrs['paths'] = self.paths

    def _load(self, path=None, hdf5_object=None):

        if path is not None:
            raise NotImplementedError

        self.names = tuple(hdf5_object.attrs['names'])
        self.paths = tuple(hdf5_object.attrs['paths'])


class HeavyPIVResults(DataObject):

    _keys_to_be_saved = [
        'xs', 'ys', 'deltaxs', 'deltays', 'correls_max']

    def __init__(self, deltaxs=None, deltays=None,
                 xs=None, ys=None, errors=None,
                 correls_max=None, correls=None,
                 couple=None, params=None,
                 path_load=None):

        if path_load is not None:
            self._load(path_load)
            return

        self.deltaxs = deltaxs
        self.deltays = deltays
        self.ys = ys
        self.xs = xs
        self.errors = errors
        self.correls_max = correls_max
        self.correls = correls
        self.couple = couple
        self.params = params

    def get_images(self):
        return self.couple.get_arrays()

    def display(self):
        im0, im1 = self.couple.get_arrays()
        return display(
            im0, im1, self)

    def save(self, path=None):
        serie = self.couple.serie

        str_ind0 = serie._compute_strindices_from_indices(
            *[inds[0] for inds in serie.get_index_slices()])

        str_ind1 = serie._compute_strindices_from_indices(
            *[inds[1] - 1 for inds in serie.get_index_slices()])

        name = ('piv_' + serie.base_name + str_ind0 + '-' + str_ind1 + '.h5')

        if path is not None:
            path_file = os.path.join(path, name)
        else:
            path_file = name

        print(path_file)

        with h5py.File(path_file, 'w') as f:
            self.params._save_as_hdf5(hdf5_parent=f)
            self.couple.save(hdf5_parent=f)
            for k in self._keys_to_be_saved:
                f.create_dataset(k, data=self.__dict__[k])

            g = f.create_group('errors')
            g.create_dataset('keys', data=self.errors.keys())
            g.create_dataset('values', data=self.errors.values())

        return self

    def _load(self, path):

        with h5py.File(path, 'r') as f:
            self.params = ParamContainer(hdf5_object=f['params'])
            self.couple = ArrayCouple(hdf5_parent=f)
            for k in self._keys_to_be_saved:
                dataset = f[k]
                self.__dict__[k] = dataset[:]

            g = f['errors']
            keys = g['keys']
            values = g['values']
            self.errors = {k: v for k, v in zip(keys, values)}
