"""Preprocess base (:mod:`fluidimage.pre_proc.base`)
====================================================
To preprocess series of images.

.. currentmodule:: fluidimage.pre_proc.base

Provides:

.. autoclass:: PreprocBase
   :members:
   :private-members:

.. autoclass:: PreprocSpecific
   :members:
   :private-members:

"""

import logging
from fluidimage.data_objects.piv import set_path_dir_result

from .toolbox import PreprocTools
from .. import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays


logger = logging.getLogger('fluidimage')


class PreprocBase(object):
    """
    Preprocess series of images

    .. TODO: Requires reorganizing. Has major similarities with `TopologyPIV` class in 
             Both requires image loading.
    """

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters."""
        # params = super(PreprocBase, cls).create_default_params()
        params = ParamContainer(tag='params')
        params._set_child('preproc')
        params.preproc._set_child('series', attribs={'path': '',
                                                     'strcouple':'i:i+2',
                                                     'ind_start': 0,
                                                     'ind_stop': None,
                                                     'ind_step': 1})

        params.preproc._set_child('saving', attribs={'path': None,
                                                     'how': 'ask',
                                                     'postfix': 'pre'})

        params.preproc.saving._set_doc(
            "`how` can be 'ask', 'new_dir', 'complete' or 'recompute'.")
        
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
        
        path_dir = params.preproc.series.path
        self.path_dir_result, self.how_saving = set_path_dir_result(path_dir,
                                                                    params.preproc.saving.path,
                                                                    params.preproc.saving.postfix,
                                                                    params.preproc.saving.how)
        self.results = {}


class PreprocSpecific(PreprocBase):

    @classmethod
    def create_default_params(cls):
        params = super(PreprocSpecific, cls).create_default_params()

        PreprocTools._complete_class_with_tools(params)
        return params

    def __init__(self, params=None):
        super(PreprocSpecific, self).__init__(params)
        self.tools = PreprocTools(params)

