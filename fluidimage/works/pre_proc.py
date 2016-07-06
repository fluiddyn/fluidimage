"""Works Preprocess (:mod:`fluidimage.works.pre_proc`)
====================================================
To preprocess series of images using topology.

.. currentmodule:: fluidimage.works.pre_proc

Provides:

.. autoclass:: WorkPreproc
   :members:
   :private-members:

"""
import numpy as np

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage.pre_proc.base import PreprocBase
from ..data_objects.pre_proc import ArraySerie, PreprocResults, get_ind_middle


class WorkPreproc(PreprocBase):

    def calcul(self, serie):
        """Apply all enabled preprocessing tools on the series of arrays
        and returns the result as a data object.

        """
        if isinstance(serie, SerieOfArraysFromFiles):
            serie = ArraySerie(serie=serie)

        if not isinstance(serie, ArraySerie):
            raise ValueError('serie must be an instance of class ArraySerie')

        result = PreprocResults(self.params)
        images = np.array(serie.get_arrays())
        images = self.tools(images)
        serie._clear_data()
        dico = self._make_dict_to_save(serie, images)
        result.data.update(dico)
        return result

    def _make_dict_to_save(self, array_serie, images):
        name_files = array_serie.names
        nb_series = array_serie.nb_series
        ind_serie = array_serie.ind_serie

        ind_middle_start, ind_middle_end = get_ind_middle(array_serie.serie)
        if ind_serie == 0 and nb_series == 1:
            s = slice(None, None)
        elif ind_serie == 0:
            s = slice(0, ind_middle_end)
        elif ind_serie == nb_series - 1:
            s = slice(ind_middle_start, None)
        else:
            s = slice(ind_middle_start, ind_middle_end)

        return dict(zip(name_files[s], images[s]))
