"""Works Preprocess (:mod:`fluidimage.works.pre_proc`)
====================================================
To preprocess series of images using topology.

.. currentmodule:: fluidimage.works.pre_proc

Provides:

.. autoclass:: WorkPreproc
   :members:
   :private-members:

"""

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage.pre_proc.base import PreprocBase
from fluidimage.data_objects.pre_proc import ArraySerie, PreprocResults


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
        for i, img in enumerate(serie.get_arrays()):
            name = name_files[i]
            img_out = self.tools(img)
            result.data.update({name: img_out})

        return result
