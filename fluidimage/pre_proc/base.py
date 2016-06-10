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
    """
    Preprocess series of images.

    """

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters."""
        # params = super(PreprocBase, cls).create_default_params()
        params = ParamContainer(tag='params')
        params._set_child('preproc')
        params.preproc._set_child('series', attribs={'path': '',
                                                     'strcouple': 'i:i+2',
                                                     'ind_start': 0,
                                                     'ind_stop': None,
                                                     'ind_step': 1})

        PreprocTools._complete_class_with_tools(params)

        return params

    def __init__(self, params=None):
        """Set path for results and loads images as SeriesOfArrays."""
        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        serie_arrays = SerieOfArraysFromFiles(params.preproc.series.path)
        self.series = SeriesOfArrays(
            serie_arrays, params.preproc.series.strcouple,
            ind_start=params.preproc.series.ind_start,
            ind_stop=params.preproc.series.ind_stop)

        self.tools = PreprocTools(params)
        self.results = {}

    def __call__(self, sequence=None):
        """Apply all enabled preprocessing tools on the series of arrays
        and saves them in self.results.

        """
        for serie in self.series:
            name_files = serie.get_name_files()
            for i, img in enumerate(serie.iter_arrays()):
                name = name_files[i]
                self.results[name] = self.tools(img, sequence)
