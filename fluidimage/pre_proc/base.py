"""Preprocess base (:mod:`fluidimage.pre_proc.base`)
====================================================
To preprocess series of images.

.. currentmodule:: fluidimage.pre_proc.base

Provides:

.. autoclass:: PreprocBase
   :members:
   :private-members:

"""

from .toolbox import PreprocTools
from .. import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays


class PreprocBase(object):
    """Preprocess series of images with spatial filters. """

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters."""
        params = ParamContainer(tag='params')
        params._set_child('preproc')
        params.preproc._set_child('series', attribs={'path': ''})

        PreprocTools._complete_class_with_tools(params)

        return params

    def __init__(self, params=None):
        """Set path for results and loads images as SeriesOfArraysFromFiles."""
        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.serie_arrays = SerieOfArraysFromFiles(params.preproc.series.path)
        self.tools = PreprocTools(params)
        self.results = {}

    def __call__(self, sequence=None):
        """Apply all enabled preprocessing tools on the series of arrays
        and saves them in self.results.

        """
        name_files = self.serie_arrays.get_name_files()
        for i, img in enumerate(self.serie_arrays.iter_arrays()):
            name = name_files[i]
            self.results[name] = self.tools(img, sequence)


class PreprocBaseTemporalFilters(PreprocBase):
    """Preprocess series of images with spatial and temporal filters. """

    def __init__(self, params=None):
        """Loads images as SeriesOfArrays."""
        super(PreprocBaseTimeFiltering, self).__init__(params)
        attribs = {'strcouple': 'i:i+2',
                   'ind_start': 0,
                   'ind_stop': None,
                   'ind_step': 1}
        self.params.series.set_attribs(attribs)
        self.series = SeriesOfArrays(
            self.serie_arrays, params.preproc.series.strcouple,
            ind_start=params.preproc.series.ind_start,
            ind_stop=params.preproc.series.ind_stop)
