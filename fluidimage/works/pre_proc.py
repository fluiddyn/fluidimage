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
from fluidimage.util.util import logger, log_memory_usage
from fluidimage.pre_proc.base import PreprocBase
from fluidimage.data_objects.pre_proc import ArraySerie, PreprocResults


def get_Ni_Nj(serie):
    """Returns number of images in the first and second indices of the series."""

    if not isinstance(serie, SerieOfArraysFromFiles):
        raise ValueError('serie must be an instance of class SerieOfArraysFromFiles')

    nb_indices = serie.nb_indices
    slices = serie.get_index_slices()
    print('slices=', slices)
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
            'Cannot evaluate series with more than 2 indices')

    return Ni, Nj


class WorkPreproc(PreprocBase):

    def calcul(self, serie):
        """Apply all enabled preprocessing tools on the series of arrays
        and returns the result as a data object.

        """
        if isinstance(serie, SerieOfArraysFromFiles):
            serie = ArraySerie(serie=serie)

        if not isinstance(serie, ArraySerie):
            raise ValueError('serie must be an instance of class ArraySerie')

        result = PreprocResults(serie, self.params)
        images = np.array(serie.get_arrays())
        images = self.tools(images)
        serie._clear_data()
        dico = self._make_dict_to_save(serie, images)
        result.data.update(dico)
        log_memory_usage('Memory usage after preprocessing:')

        return result

    def _make_dict_to_save(self, array_serie, images):
        name_files = array_serie.names
        nb_series = array_serie.nb_series
        ind_serie = array_serie.ind_serie

        Ni, Nj = get_Ni_Nj(array_serie.serie)
        ind_middle_start = int(np.floor(Ni / 2.)) * Nj
        ind_middle_end = int(np.ceil(Ni / 2.)) * Nj

        if ind_serie == 0 and nb_series == 1:
            logger.info('Preprocessed single serie, 1 out of 1')
            s = slice(None, None)
        elif ind_serie == 0:
            logger.info('Preprocessed first serie, %d out of %d',
                        ind_serie + 1, nb_series)
            s = slice(0, ind_middle_end)
        elif ind_serie == nb_series - 1:
            logger.info('Preprocessed last serie, %d out of %d',
                        ind_serie + 1, nb_series)
            s = slice(ind_middle_start, None)
        else:
            logger.info('Preprocessed next serie, %d out of %d',
                        ind_serie + 1, nb_series)
            s = slice(ind_middle_start, ind_middle_end)

        return dict(zip(name_files[s], images[s]))
