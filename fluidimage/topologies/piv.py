"""Topology for PIV computation
===============================

.. autoclass:: TopologyPIV
   :members:
   :private-members:

"""
import os

from .. import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays

from .base import TopologyBase

from .waiting_queues.base import (
    WaitingQueueMultiprocessing, WaitingQueueThreading,
    WaitingQueueMakeCouple, WaitingQueueLoadImage)

from ..works.piv import WorkPIV
from ..data_objects.piv import get_name_piv, set_path_dir_result
from ..util.util import logger


class TopologyPIV(TopologyBase):
    """Topology for PIV.

    """

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters.

        For developers: cf. fluidsim.base.params

        """
        params = ParamContainer(tag='params')

        params._set_child('series', attribs={'path': '',
                                             'strcouple': 'i:i+2',
                                             'ind_start': 0,
                                             'ind_stop': None,
                                             'ind_step': 1})

        params._set_child('saving', attribs={'path': None,
                                             'how': 'ask',
                                             'postfix': 'piv'})

        params.saving._set_doc(
            "`how` can be 'ask', 'new_dir', 'complete' or 'recompute'.")

        WorkPIV._complete_params_with_default(params)
        return params

    def __init__(self, params=None, logging_level='info', nb_max_workers=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.piv_work = WorkPIV(params)

        serie_arrays = SerieOfArraysFromFiles(params.series.path)

        self.series = SeriesOfArrays(
            serie_arrays, params.series.strcouple,
            ind_start=params.series.ind_start,
            ind_stop=params.series.ind_stop,
            ind_step=params.series.ind_step)

        path_dir = self.series.serie.path_dir
        path_dir_result, self.how_saving = set_path_dir_result(
            path_dir, params.saving.path,
            params.saving.postfix, params.saving.how)

        self.path_dir_result = path_dir_result

        self.results = {}

        def save_piv_object(o):
            ret = o.save(path_dir_result)
            return ret

        self.wq_piv = WaitingQueueThreading(
            'delta', save_piv_object, self.results, topology=self)
        self.wq_couples = WaitingQueueMultiprocessing(
            'couple', self.piv_work.calcul, self.wq_piv,
            topology=self)
        self.wq_images = WaitingQueueMakeCouple(
            'array image', self.wq_couples, topology=self)
        self.wq0 = WaitingQueueLoadImage(
            destination=self.wq_images,
            path_dir=path_dir, topology=self)

        super(TopologyPIV, self).__init__(
            [self.wq0, self.wq_images, self.wq_couples, self.wq_piv],
            path_output=path_dir_result, logging_level=logging_level,
            nb_max_workers=nb_max_workers)

        self.add_series(self.series)

    def add_series(self, series):

        if len(series) == 0:
            logger.warning('add 0 couple. No PIV to compute.')
            return

        if self.how_saving == 'complete':
            names = []
            index_series = []
            for i, serie in enumerate(series):
                name_piv = get_name_piv(serie, prefix='piv')
                if os.path.exists(os.path.join(
                        self.path_dir_result, name_piv)):
                    continue
                for name in serie.get_name_files():
                    if name not in names:
                        names.append(name)

                index_series.append(i * series.ind_step + series.ind_start)

            if len(index_series) == 0:
                logger.warning(
                    'topology in mode "complete" and work already done.')
                return

            series.set_index_series(index_series)

            logger.debug(repr(names))
            logger.debug(repr([serie.get_name_files() for serie in series]))
        else:
            names = series.get_name_all_files()

        nb_series = len(series)
        print('Add {} PIV fields to compute.'.format(nb_series))

        for i, serie in enumerate(series):
            if i > 1:
                break
            print('Files of serie {}: {}'.format(i, serie.get_name_files()))

        self.wq0.add_name_files(names)
        self.wq_images.add_series(series)

        k, o = self.wq0.popitem()
        im = self.wq0.work(o)
        self.wq0.fill_destination(k, im)

        self.piv_work._prepare_with_image(im)

    def _print_at_exit(self, time_since_start):

        txt = 'Stop compute after t = {:.2f} s'.format(time_since_start)
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += (' ({} piv fields, {:.2f} s/field).'.format(
                nb_results, time_since_start / nb_results))
        else:
            txt += '.'

        txt += '\npath results:\n' + self.path_dir_result

        print(txt)
