"""Topology for preprocessing images (:mod:`fluidimage.topologies.pre_proc`)
============================================================================

To preprocess series of images using multiprocessing and waiting queues.

Provides:

.. autoclass:: TopologyPreproc
   :members:
   :private-members:

"""

import os
import json

from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluiddyn.io.image import imread

from ..works.pre_proc import WorkPreproc

from ..data_objects.piv import set_path_dir_result
from ..data_objects.display import DisplayPreProc
from ..data_objects.pre_proc import get_name_preproc

from ..util.util import logger

from .base import TopologyBase

from .waiting_queues.base import (
    WaitingQueueMultiprocessing, WaitingQueueThreading)
from .waiting_queues.series import (
    WaitingQueueMakeSerie, WaitingQueueLoadImageSeries)


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
                                            'ind_step': 1,
                                            'sequential_loading': True})

        params.preproc._set_child('saving', attribs={'path': None,
                                                     'how': 'ask',
                                                     'format': 'img',
                                                     'postfix': 'pre'})

        params.preproc.saving._set_doc(
            "`how` can be 'ask', 'new_dir', 'complete' or 'recompute'.\n" +
            "`format` can be 'img' or 'hdf5'")

        params._set_internal_attr(
            '_value_text',
            json.dumps({'program': 'fluidimage',
                        'module': 'fluidimage.topologies.pre_proc',
                        'class': 'TopologyPreproc'}))

        return params

    def __init__(self, params=None, logging_level='info', nb_max_workers=None):
        if params is None:
            params = self.__class__.create_default_params()

        self.params = params.preproc
        self.preproc_work = WorkPreproc(params)
        serie_arrays = self.preproc_work.serie_arrays
        self.series = SeriesOfArrays(
            serie_arrays, params.preproc.series.strcouple,
            ind_start=params.preproc.series.ind_start,
            ind_stop=params.preproc.series.ind_stop,
            ind_step=params.preproc.series.ind_step)

        self.nb_items_per_serie = serie_arrays.get_nb_files()

        if os.path.isdir(params.preproc.series.path):
            path_dir = params.preproc.series.path
        else:
            path_dir = os.path.dirname(params.preproc.series.path)
        self.path_dir_result, self.how_saving = set_path_dir_result(
            path_dir, params.preproc.saving.path,
            params.preproc.saving.postfix, params.preproc.saving.how)

        self.params.saving.path = self.path_dir_result
        self.results = self.preproc_work.results
        self.display = self.preproc_work.display

        def save_preproc_results_object(o):
            return o.save(path=self.path_dir_result)

        self.wq_preproc = WaitingQueueThreading(
            'save results', save_preproc_results_object,
            self.results, work_name='save', topology=self)

        self.wq_serie = WaitingQueueMultiprocessing(
            'apply preprocessing', self.preproc_work.calcul,
            self.wq_preproc, work_name='preprocessing', topology=self)

        self.wq_images = WaitingQueueMakeSerie(
            'make serie', self.wq_serie, topology=self)

        self.wq0 = WaitingQueueLoadImageSeries(
            destination=self.wq_images, path_dir=path_dir, topology=self,
            sequential=params.preproc.series.sequential_loading)

        super(TopologyPreproc, self).__init__(
            [self.wq0, self.wq_images, self.wq_serie, self.wq_preproc],
            path_output=self.path_dir_result, logging_level=logging_level,
            nb_max_workers=nb_max_workers)

        self.add_series(self.series)

    def add_series(self, series):

        if len(series) == 0:
            logger.warning(
                'encountered empty series. No images to preprocess.')
            return

        if self.how_saving == 'complete':
            names = []
            index_series = []
            for i, serie in enumerate(series):
                names_serie = serie.get_name_files()
                name_preproc = get_name_preproc(
                    serie, names_serie, i, series.nb_series,
                    self.params.saving.format)
                if os.path.exists(os.path.join(
                        self.path_dir_result, name_preproc)):
                    continue

                for name in names_serie:
                    if name not in names:
                        names.append(name)

                index_series.append(i + series.ind_start)

            if len(index_series) == 0:
                logger.warning('topology in mode "complete" and '
                               'work already done.')
                return

            series.set_index_series(index_series)

            logger.debug(repr(names))
            logger.debug(repr([serie.get_name_files() for serie in series]))
        else:
            names = series.get_name_all_files()

        print('Add {} image serie(s) to compute.'.format(len(series)))

        self.wq0.add_name_files(names)
        self.wq_images.add_series(series)

        k, o = self.wq0.popitem()
        im = self.wq0.work(o)
        self.wq0.fill_destination(k, im)

    def compare(self, indices=[0, 1], suffix=None, hist=False):
        if not suffix:
            suffix = '.' + self.params.saving.postfix
        pathbase = self.params.series.path + '/'

        im0 = imread(pathbase + self.series.get_name_all_files()[indices[0]])
        im1 = imread(pathbase + self.series.get_name_all_files()[indices[1]])
        im0p = imread(pathbase[:-1] + suffix + '/' +
                      self.series.get_name_all_files()[indices[0]])
        im1p = imread(pathbase[:-1] + suffix + '/' +
                      self.series.get_name_all_files()[indices[1]])
        return DisplayPreProc(
            im0, im1, im0p, im1p, hist=hist)

params = TopologyPreproc.create_default_params()

__doc__ += params._get_formatted_docs().replace(
    'Parameters\n    ----------', '').replace('References\n    ----------', '')
