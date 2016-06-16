"""Preprocessing data objects
=============================

.. autoclass:: ArraySeries
   :members:
   :private-members:

.. autoclass:: PreprocResults
   :members:
   :private-members:

"""

import os
import h5py

from scipy.misc import imsave
from .piv import ArrayCouple, LightPIVResults
from .. import ParamContainer


class ArraySerie(ArrayCouple):
    def __init__(
            self, names=None, arrays=None, serie=None,
            str_path=None, hdf5_parent=None):

        if str_path is not None:
            self._load(path=str_path)
            return

        if hdf5_parent is not None:
            self._load(hdf5_object=hdf5_parent['serie'])
            return

        if serie is not None:
            names = serie.get_name_files()
            paths = serie.get_path_files()
            self.paths = tuple(os.path.abspath(p) for p in paths)

            if arrays is None:
                arrays = serie.get_arrays()

        self.names = tuple(names)
        self.arrays = tuple(arrays)
        self.serie = serie

    def save(self, path=None, hdf5_parent=None):
        if path is not None:
            raise NotImplementedError

        if not isinstance(hdf5_parent, (h5py.File,)):
            raise NotImplementedError

        hdf5_parent.create_group('serie')
        group = hdf5_parent['serie']
        group.attrs['names'] = self.names
        group.attrs['paths'] = self.paths


class PreprocResults(LightPIVResults):

    def __init__(self, serie=None, params=None,
                 str_path=None, hdf5_object=None):

        self._keys_to_be_saved = ['data']
        if hdf5_object is not None:
            if serie is not None:
                self.serie = serie

            if params is not None:
                self.params = params

            self._load_from_hdf5_object(hdf5_object)
            return

        if str_path is not None:
            self._load(str_path)
            return

        self.serie = serie
        self.params = params
        self.data = {}

    def _get_name(self):

        name = ' TODO'
        return name

    def save(self, path=None, out_format='img'):

        name = self._get_name()

        if path is not None:
            path_file = os.path.join(path, name)
        else:
            path_file = name

        if out_format == 'hdf5':
            with h5py.File(path_file, 'w') as f:
                f.attrs['class_name'] = 'PreprocResults'
                f.attrs['module_name'] = 'fluidimage.data_objects.pre_proc'

                self._save_in_hdf5_object(f, tag='pre_proc')
        elif out_format == 'img':
            for k, v in self.data.items():
                imsave(path_file + k, v)

        return self

    def _save_in_hdf5_object(self, f, tag='pre_proc'):

        if 'class_name' not in f.attrs.keys():
            f.attrs['class_name'] = 'PreprocResults'
            f.attrs['module_name'] = 'fluidimage.data_objects.pre_proc'
        if 'params' not in f.keys():
            self.params._save_as_hdf5(hdf5_parent=f)
        if 'serie' not in f.keys():
            self.serie.save(hdf5_parent=f)

        g_piv = f.create_group(tag)
        g_piv.attrs['class_name'] = 'PreprocResults'
        g_piv.attrs['module_name'] = 'fluidimage.data_objects.pre_proc'

        for k in self._keys_to_be_saved:
            data = self.__dict__[k]
            if isinstance(data, dict):
                g_piv.create_group(k)
                for key, value in data.items():
                    print(key, type(value))
                    g_piv[k].create_dataset(key, value)
            else:
                g_piv.create_dataset(k, data)

    def _load(self, path):
        with h5py.File(path, 'r') as f:
            self.params = ParamContainer(hdf5_object=f['params'])
            self.couple = ArrayCouple(hdf5_parent=f)

        with h5py.File(path, 'r') as f:
            self._load_from_hdf5_object(f['piv'])

    def _load_from_hdf5_object(self, g_piv):

        f = g_piv.parent

        if not hasattr(self, 'params'):
            self.params = ParamContainer(hdf5_object=f['params'])

        if not hasattr(self, 'serie'):
            self.serie = ArraySerie(hdf5_parent=f)

        for k in self._keys_to_be_saved:
            dataset = g_piv[k]
            self.__dict__[k] = dataset[:]
