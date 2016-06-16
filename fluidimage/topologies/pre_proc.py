"""Topology for preprocessing images (:mod:`fluidimage.topologies.pre_proc`)
============================================================================
To preprocess series of images using multiprocessing and waiting queues.

.. currentmodule:: fluidimage.topologies.pre_proc

Provides:

.. autoclass:: TopologyPreproc
   :members:
   :private-members:

"""

import os
import logging

from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.data_objects.piv import get_name_piv, set_path_dir_result
from fluidimage.works.pre_proc import WorkPreproc
from .base import TopologyBase
from .waiting_queues.base import (
    WaitingQueueMultiprocessing, WaitingQueueThreading,
    WaitingQueueMakeSerie, WaitingQueueLoadImage)


logger = logging.getLogger('fluidimage')


class TopologyPreproc(TopologyBase):
    """Preprocess series of images and provides interface for I/O and
    multiprocessing.

    """

    @classmethod
    def create_default_params(cls):
        params = WorkPreproc.create_default_params()
        params.preproc.series._set_attribs({'strcouple': 'i:i+2',
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
        if params is None:
            params = self.__class__.create_default_params()

        self.params = params.preproc
        self.preproc_work = WorkPreproc(params)
        serie_arrays = self.preproc_work.serie_arrays
        self.series = SeriesOfArrays(
            serie_arrays, params.preproc.series.strcouple,
            ind_start=params.preproc.series.ind_start,
            ind_stop=params.preproc.series.ind_stop)

        super(TopologyPreproc, self).__init__(params)
        path_dir = params.preproc.series.path
        self.path_dir_result, self.how_saving = set_path_dir_result(
            path_dir, params.preproc.saving.path,
            params.preproc.saving.postfix, params.preproc.saving.how)

        self.results = {}

        self.wq_preproc = WaitingQueueThreading(
            'save results', lambda o: o.save(self.path_dir_result),
            self.results, work_name='save', topology=self)

        self.wq_serie = WaitingQueueMultiprocessing(
            'apply preprocessing', self.preproc_work.calcul,
            self.wq_preproc, work_name='preproc', topology=self)

        self.wq_images = WaitingQueueMakeSerie(
            'make serie', self.wq_serie, topology=self)

        self.wq0 = WaitingQueueLoadImage(
            destination=self.wq_images, path_dir=path_dir, topology=self)

        self.queues = [self.wq0, self.wq_images, self.wq_serie, self.wq_preproc]
        self.add_series(self.series)

    def add_series(self, series):

        if len(series) == 0:
            print('Warning: add 0 couple. No PIV to compute.')
            return

        if self.how_saving == 'complete':
            names = []
            index_series = []
            for i, serie in enumerate(series):
                name_piv = get_name_piv(serie, prefix='piv')
                if os.path.exists(os.path.join(
                        self.path_dir_result, name_piv)):
                    continue
                names_serie = serie.get_name_files()
                for name in names_serie:
                    if name not in names:
                        names.append(name)

                index_series.append(i + series.ind_start)

            if len(index_series) == 0:
                print('Warning: topology in mode "complete" and '
                      'work already done.')
                return

            series.set_index_series(index_series)

            logger.debug(repr(names))
            logger.debug(repr([serie.get_name_files() for serie in series]))
        else:
            names = series.get_name_all_files()

        print('Add {} PIV fields to compute.'.format(len(series)))

        self.wq0.add_name_files(names)
        self.wq_images.add_series(series)

        k, o = self.wq0.popitem()
        im = self.wq0.work(o)
        self.wq0.fill_destination(k, im)
