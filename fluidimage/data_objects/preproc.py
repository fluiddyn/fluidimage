"""Preprocessing data objects (:mod:`fluidimage.data_objects.preproc`)
======================================================================

.. autoclass:: ArraySerie
   :members:
   :private-members:

.. autoclass:: PreprocResults
   :members:
   :private-members:

"""

import math
import os

import h5py

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage.util import imsave, imsave_h5

from .piv import ArrayCouple, LightPIVResults


def get_Ni_Nj(serie):
    """Returns number of images in the first and second indices
    of the series.

    """
    if not isinstance(serie, SerieOfArraysFromFiles):
        raise ValueError(
            "serie must be an instance of class SerieOfArraysFromFiles"
        )

    nb_indices = serie.nb_indices
    slices = serie.get_index_slices()
    Ni = slices[0][1] - slices[0][0]
    if nb_indices == 1:
        Nj = 1
    elif nb_indices == 2:
        if len(slices[1]) == 1:
            Nj = 1
        else:
            Nj = slices[1][1] - slices[1][0]
    else:
        raise NotImplementedError(
            "Cannot evaluate series with more than 2 indices"
        )

    return Ni, Nj


def get_ind_middle(serie):
    Ni, Nj = get_Ni_Nj(serie)
    ind_middle_start = int(math.floor(Ni / 2.0)) * Nj
    ind_middle_end = int(math.ceil(Ni / 2.0)) * Nj
    if Ni % 2 == 0:
        ind_middle_start -= 1

    return ind_middle_start, ind_middle_end


def get_name_preproc(serie, name_files, ind_serie, nb_series, out_format):
    ind_middle_start, ind_middle_end = get_ind_middle(serie)

    if ind_serie == 0 and nb_series == 1:
        s = -1
    elif ind_serie == 0:
        s = ind_middle_start
    elif ind_serie == nb_series - 1:
        s = -1
    else:
        s = ind_middle_start

    if out_format == "img":
        return name_files[s]

    else:
        fname, ext = os.path.splitext(name_files[s])
        fname += "." + out_format
        return fname


class ArraySerie(ArrayCouple):
    def __init__(
        self,
        names=None,
        arrays=None,
        serie=None,
        ind_serie=0,
        nb_series=1,
        str_path=None,
        hdf5_parent=None,
    ):

        if str_path is not None:
            self._load(path=str_path)
            return

        if hdf5_parent is not None:
            self._load(hdf5_object=hdf5_parent["serie"])
            return

        if serie is not None:
            names = serie.get_name_arrays()
            paths = serie.get_path_arrays()
            self.paths = tuple(os.path.abspath(p) for p in paths)

            if arrays is None:
                arrays = serie.get_arrays()

        self.ind_serie = ind_serie
        self.nb_series = nb_series
        self.names = tuple(names)
        self.arrays = tuple(arrays)
        self.serie = serie

    def _clear_data(self):
        self.arrays = tuple()

    def save(self, path=None, hdf5_parent=None):
        if path is not None:
            raise NotImplementedError

        if not isinstance(hdf5_parent, (h5py.File,)):
            raise NotImplementedError

        hdf5_parent.create_group("serie")
        group = hdf5_parent["serie"]
        group.attrs["names"] = self.names
        group.attrs["paths"] = self.paths


class PreprocResults(LightPIVResults):
    def __init__(self, params=None, str_path=None, hdf5_object=None):

        self._keys_to_be_saved = ["data"]
        if hdf5_object is not None:
            if params is not None:
                self.params = params

            self._load_from_hdf5_object(hdf5_object)
            return

        if str_path is not None:
            self._load(str_path)
            return

        self.params = params
        self.data = {}

    def _clear_data(self):
        self.data = {}

    def save(self, path=None):
        out_format = self.params.saving.format
        for k, v in self.data.items():
            path_file = os.path.join(path, k)
            if out_format == "img":
                imsave(path_file, v, as_int=True)
            elif out_format == "h5":
                attrs = {
                    "class_name": "PreprocResults",
                    "module_name": self.__module__,
                }
                imsave_h5(path_file, v, self.params, attrs, as_int=True)
            else:
                # Try to save in formats supported by PIL.Image
                imsave(path_file, v, format=out_format, as_int=True)

        self._clear_data()
        return self
