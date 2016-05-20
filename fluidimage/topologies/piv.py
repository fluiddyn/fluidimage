"""Topology for PIV computation
===============================

.. autoclass:: TopologyPIV
   :members:
   :private-members:

"""
import os
import logging

from fluiddyn.util.query import query

from .. import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays

from .base import TopologyBase

from .waiting_queues.base import (
    WaitingQueueMultiprocessing, WaitingQueueThreading,
    WaitingQueueMakeCouple, WaitingQueueLoadImage)

from ..works.piv import WorkPIV
from ..data_objects.piv import get_name_piv

logger = logging.getLogger('fluidimage')


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

    def __init__(self, params=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.piv_work = WorkPIV(params)

        serie_arrays = SerieOfArraysFromFiles(params.series.path)

        self.series = SeriesOfArrays(
            serie_arrays, params.series.strcouple,
            ind_start=params.series.ind_start,
            ind_stop=params.series.ind_stop)

        path_dir = self.series.serie.path_dir
        if params.saving.path is not None:
            path_dir_result = params.saving.path
        else:
            path_dir_result = path_dir + '.' + params.saving.postfix

        how = params.saving.how
        if os.path.exists(path_dir_result):
            if how == 'ask':
                answer = query(
                    'The directory {} '.format(path_dir_result) +
                    'already exists. What do you want to do?\n'
                    'New dir, Complete, Recompute or Stop?\n')

                while answer.lower() not in ['n', 'c', 'r', 's']:
                    answer = query(
                        "The answer should be in ['n', 'c', 'r', 's']\n"
                        "Please type your answer again...\n")

                if answer == 's':
                    raise ValueError('Stopped by the user.')
                elif answer == 'n':
                    how = 'new_dir'
                elif answer == 'c':
                    how = 'complete'
                elif answer == 'r':
                    how = 'recompute'

            if how == 'new_dir':
                i = 0
                while os.path.exists(path_dir_result + str(i)):
                    i += 1
                path_dir_result += str(i)

        self.how_saving = how

        if not os.path.exists(path_dir_result):
            os.mkdir(path_dir_result)

        self.path_dir_result = path_dir_result

        self.results = {}
        self.wq_piv = WaitingQueueThreading(
            'delta', lambda o: o.save(path_dir_result), self.results,
            work_name='save', topology=self)
        self.wq_couples = WaitingQueueMultiprocessing(
            'couple', self.piv_work.calcul, self.wq_piv, work_name='PIV',
            topology=self)
        self.wq_images = WaitingQueueMakeCouple(
            'array image', self.wq_couples, topology=self)
        self.wq0 = WaitingQueueLoadImage(
            destination=self.wq_images,
            path_dir=path_dir, topology=self)

        super(TopologyPIV, self).__init__([
            self.wq0, self.wq_images, self.wq_couples, self.wq_piv
        ])

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

        self.piv_work._prepare_with_image(im)
