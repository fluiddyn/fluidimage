"""Works Preprocess (:mod:`fluidimage.works.pre_proc`)
====================================================
To preprocess series of images using topology.

.. currentmodule:: fluidimage.works.pre_proc

Provides:

.. autoclass:: WorkPreproc
   :members:
   :private-members:

"""
import logging
import numpy as np

from fluiddyn.util import terminal_colors as term
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage import log_memory_usage
from fluidimage.pre_proc.base import PreprocBase
from fluidimage.data_objects.pre_proc import ArraySerie, PreprocResults


logger = logging.getLogger('fluidimage')


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
        name_files = serie.names
        images = np.array(serie.get_arrays())
        images = self.tools(images)
        serie._clear_data()
        data = self._make_data_to_save(serie, name_files, images)
        result.data.update(data)
        log_memory_usage('Memory usage after preprocessing:')

        return result

    def _make_data_to_save(self, serie, name_files, images_out):
        nb_series = serie.nb_series
        ind_serie = serie.ind_serie
        nb_img = len(name_files)
        ind_middle_img = nb_img // 2

        if ind_serie == 0 and nb_series == 1:
            logger.info('Preprocessed single serie, 1 out of 1')
            s = slice(None, None)
        elif ind_serie == 0:
            logger.info('Preprocessed first serie, %d out of %d',
                        ind_serie + 1, nb_series)
            s = slice(0, ind_middle_img + 1)
        elif ind_serie == nb_series - 1:
            logger.info('Preprocessed last serie, %d out of %d',
                        ind_serie + 1, nb_series)
            s = slice(ind_middle_img, None)
        else:
            logger.info('Preprocessed next serie, %d out of %d',
                        ind_serie + 1, nb_series)
            s = slice(ind_middle_img, ind_middle_img + 1)

        return dict(zip(name_files[s], images_out[s]))
