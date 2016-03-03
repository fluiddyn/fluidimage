
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
            str_path=None, hdf5_parent=None):

        if str_path is not None:
            self._load(path=str_path)
            return

        if hdf5_parent is not None:
            self._load(hdf5_object=hdf5_parent['couple'])
            return

        if serie is not None:
            if serie.get_nb_files() != 2:
                raise ValueError('serie has to contain 2 arrays.')
            names = serie.get_name_files()
            paths = serie.get_path_files()
            self.paths = tuple(os.path.abspath(p) for p in paths)

            if arrays is None:
                arrays = serie.get_arrays()

        self.names = tuple(names)
        self.arrays = tuple(arrays)
        self.serie = serie

    def get_arrays(self):
        if not hasattr(self, 'arrays'):
            self.arrays = tuple(imread(path) for path in self.paths)

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
                 str_path=None, hdf5_object=None):

        if hdf5_object is not None:
            if couple is not None:
                self.couple = couple

            if params is not None:
                self.params = params

            self._load_from_hdf5_object(hdf5_object)
            return

        if str_path is not None:
            self._load(str_path)
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

    def _get_name(self):

        serie = self.couple.serie

        str_ind0 = serie._compute_strindices_from_indices(
            *[inds[0] for inds in serie.get_index_slices()])

        str_ind1 = serie._compute_strindices_from_indices(
            *[inds[1] - 1 for inds in serie.get_index_slices()])

        name = ('piv_' + serie.base_name + str_ind0 + '-' + str_ind1 + '.h5')
        return name

    def save(self, path=None):

        name = self._get_name()

        if path is not None:
            path_file = os.path.join(path, name)
        else:
            path_file = name

        with h5py.File(path_file, 'w') as f:
            self._save_in_hdf5_object(f)

        return self

    def _save_in_hdf5_object(self, f, tag='piv0'):

        if 'class_name' not in f.attrs.keys():
            f.attrs['class_name'] = 'HeavyPIVResults'
            f.attrs['module_name'] = 'fluidimage.data_objects.piv'

        if 'params' not in f.keys():
            self.params._save_as_hdf5(hdf5_parent=f)

        if 'couple' not in f.keys():
            self.couple.save(hdf5_parent=f)

        g_piv = f.create_group(tag)
        g_piv.attrs['class_name'] = 'HeavyPIVResults'
        g_piv.attrs['module_name'] = 'fluidimage.data_objects.piv'

        for k in self._keys_to_be_saved:
            g_piv.create_dataset(k, data=self.__dict__[k])

        g = g_piv.create_group('errors')
        g.create_dataset('keys', data=self.errors.keys())
        g.create_dataset('values', data=self.errors.values())

    def _load(self, path):

        with h5py.File(path, 'r') as f:
            self.load_from_hdf5_object(f['piv0'])

    def _load_from_hdf5_object(self, g_piv):

        f = g_piv.parent

        if not hasattr(self, 'params'):
            self.params = ParamContainer(hdf5_object=f['params'])

        if not hasattr(self, 'couple'):
            self.couple = ArrayCouple(hdf5_parent=f)

        for k in self._keys_to_be_saved:
            dataset = g_piv[k]
            self.__dict__[k] = dataset[:]

        g = g_piv['errors']
        keys = g['keys']
        values = g['values']
        self.errors = {k: v for k, v in zip(keys, values)}


class MultipassPIVResults(DataObject):

    def __init__(self, str_path=None):
        self.passes = []

        if str_path is not None:
            self._load(str_path)

    def display(self, i=-1):
        r = self.passes[i]
        return r.display()

    def __getitem__(self, key):
        return self.passes[key]

    def append(self, results):
        i = len(self.passes)
        self.passes.append(results)
        self.__dict__['piv{}'.format(i)] = results

    def _get_name(self):
        r = self.passes[0]
        return r._get_name()

    def save(self, path=None):

        name = self._get_name()

        if path is not None:
            path_file = os.path.join(path, name)
        else:
            path_file = name

        with h5py.File(path_file, 'w') as f:
            f.attrs['class_name'] = 'MultipassPIVResults'
            f.attrs['module_name'] = 'fluidimage.data_objects.piv'

            f.attrs['nb_passes'] = len(self.passes)

            for i, r in enumerate(self.passes):
                r._save_in_hdf5_object(f, tag='piv{}'.format(i))

    def _load(self, path):

        with h5py.File(path, 'r') as f:
            self.params = ParamContainer(hdf5_object=f['params'])
            self.couple = ArrayCouple(hdf5_parent=f)

            nb_passes = f.attrs['nb_passes']

            for ip in range(nb_passes):
                g = f['piv{}'.format(ip)]
                self.append(HeavyPIVResults(
                    hdf5_object=g, couple=self.couple, params=self.params))
