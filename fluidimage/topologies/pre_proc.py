"""Topology for preprocessing images (:mod:`fluidimage.topologies.pre_proc`)
============================================================================
To preprocess series of images using multiprocessing and waiting queues.

.. currentmodule:: fluidimage.topologies.pre_proc

Provides:

.. autoclass:: TopologyPreproc
   :members:
   :private-members:

"""

import logging

from fluidimage.data_objects.piv import set_path_dir_result
from fluidimage.pre_proc.base import PreprocBase
from .base import TopologyBase
from .waiting_queues.base import (
    WaitingQueueMultiprocessing, WaitingQueueThreading,
    WaitingQueueMakeCouple, WaitingQueueLoadImage)


logger = logging.getLogger('fluidimage')


class TopologyPreproc(PreprocBase, TopologyBase):
    """Preprocess series of images and provides interface for I/O and
    multiprocessing.

    """

    @classmethod
    def create_default_params(cls):
        params = super(TopologyPreproc, cls).create_default_params()
        params.preproc._set_child('saving', attribs={'path': None,
                                                     'how': 'ask',
                                                     'postfix': 'pre'})

        params.preproc.saving._set_doc(
            "`how` can be 'ask', 'new_dir', 'complete' or 'recompute'.")

        return params

    def __init__(self, params=None):
        super(TopologyPreproc, self).__init__(params)
        path_dir = params.preproc.series.path
        self.path_dir_result, self.how_saving = set_path_dir_result(
            path_dir, params.preproc.saving.path,
            params.preproc.saving.postfix, params.preproc.saving.how)

        self.results = {}

        self.work = None  # Something

        self.wq_result = WaitingQueueThreading(
            'save results', lambda o: o.save(self.path_dir_result),
            self.results, work_name='save', topology=self)

        self.wq_preproc = WaitingQueueMultiprocessing(
            'apply preprocessing', self.work.calcul,
            self.wq_result, work_name='preproc', topology=self)

        self.wq_images = WaitingQueueLoadImage(
            'image loader', self.wq_preproc, path_dir=path_dir, topology=self)

        self.queues = [self.wq_images, self.wq_preproc, self.wq_result]
        self.add_series(self.series)
