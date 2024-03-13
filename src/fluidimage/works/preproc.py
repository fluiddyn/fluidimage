"""Works Preprocess
===================

To preprocess series of images using topology.

Provides:

.. autoclass:: WorkPreproc
   :members:
   :private-members:

"""

import sys

import numpy as np

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage import ParamContainer
from fluidimage.data_objects.display_pre import DisplayPreProc
from fluidimage.data_objects.preproc import (
    ArraySerie,
    PreprocResults,
    get_ind_middle,
)
from fluidimage.util import print_memory_usage

from . import BaseWorkFromSerie


def _make_doc_with_filtered_params_doc(cls):
    params = cls.create_default_params()
    strings = ("Parameters", "References", "----------")
    return "\n".join(
        line
        for line in params._get_formatted_docs().split("\n")
        if not any(line.endswith(string) for string in strings)
    )


def _get_backend_class(backend):
    if backend == "python":
        from fluidimage.preproc.toolbox import PreprocToolsPy

        return PreprocToolsPy
    elif backend == "opencv":
        from fluidimage.preproc.toolbox import PreprocToolsCV

        return PreprocToolsCV

    raise ImportError(f"Unknown backend: {backend}")


class WorkPreproc(BaseWorkFromSerie):
    """Work for preprocessing.

    Preprocess series of images with various tools.

    """

    @classmethod
    def create_default_params(cls, backend="python"):
        """Class method returning the default parameters.

        Parameters
        ----------

        backend: {'python', 'opencv'}

            Specifies which backend to use.

        """
        params = ParamContainer(tag="params")
        params._set_attrib("backend", backend)
        BaseWorkFromSerie._complete_params_with_default(params)
        params.series.str_subset = "all1by1"

        Tools = _get_backend_class(backend)
        Tools.create_default_params(params)
        return params

    def __init__(self, params=None):
        """Set path for results and loads images as SerieOfArraysFromFiles."""
        if params is None:
            params = type(self).create_default_params()
        super().__init__(params)
        self.params = params
        Tools = _get_backend_class(params.backend)
        self.tools = Tools(params)

    def calcul(self, serie):
        """Apply all enabled preprocessing tools on the series of arrays
        and returns the result as a data object.

        """
        if isinstance(serie, SerieOfArraysFromFiles):
            serie = ArraySerie(serie=serie)
        elif isinstance(serie, dict):
            serie = ArraySerie(**serie)

        if not isinstance(serie, ArraySerie):
            raise ValueError("serie must be an instance of class ArraySerie")

        images = np.array(serie.get_arrays())
        images = self.tools.apply(images)
        serie.clear_data()
        result = PreprocResults(self.params)
        result.data.update(self._make_dict_to_save(serie, images))
        print_memory_usage(
            f"Memory usage after preprocessing {serie.ind_serie + 1}/{serie.nb_series} series"
        )
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

    def display(self, ind=0, hist=False):
        """Display figures to study the preprocessing"""

        serie0 = self.get_serie(ind)
        serie1 = self.get_serie(ind + 1)

        result0 = self.calcul(serie0)
        result1 = self.calcul(serie1)

        key0 = serie0.get_name_arrays()[0]
        key1 = serie1.get_name_arrays()[0]

        arr_input0 = serie0.get_array_from_name(key0)
        arr_input1 = serie0.get_array_from_name(key1)

        arr_output0 = result0.data[key0]
        arr_output1 = result1.data[key1]

        return DisplayPreProc(
            arr_input0, arr_input1, arr_output0, arr_output1, hist=hist
        )


Work = WorkPreproc

if "sphinx" in sys.modules:
    __doc__ += _make_doc_with_filtered_params_doc(Work)
