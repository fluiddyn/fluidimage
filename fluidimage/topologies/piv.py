
import os

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.serieofarrays import \
    SerieOfArraysFromFiles, SeriesOfArrays

from .base import TopologyBase

from ..waiting_queues.base import (
    WaitingQueueMultiprocessing, WaitingQueueThreading,
    WaitingQueueMakeCouple, WaitingQueueLoadImage)

from ..works.piv import WorkPIV


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
                                             'strcouple': 'i+1:i+3'})

        WorkPIV._complete_params_with_default(params)
        return params

    def __init__(self, params=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.piv_work = WorkPIV(params)

        serie_arrays = SerieOfArraysFromFiles(params.series.path)
        self.series = SeriesOfArrays(serie_arrays, params.series.strcouple)

        path_dir = self.series.serie.path_dir
        path_dir_result = path_dir + '.piv'

        if not os.path.exists(path_dir_result):
            os.mkdir(path_dir_result)

        self.results = {}
        self.wq_piv = WaitingQueueThreading(
            'delta', lambda o: o.save(path_dir_result), self.results,
            work_name='save')
        self.wq_couples = WaitingQueueMultiprocessing(
            'couple', self.piv_work.calcul, self.wq_piv, work_name='PIV')
        self.wq_images = WaitingQueueMakeCouple(
            'array image', self.wq_couples)
        self.wq0 = WaitingQueueLoadImage(
            destination=self.wq_images,
            path_dir=path_dir)

        super(TopologyPIV, self).__init__([
            self.wq0, self.wq_images, self.wq_couples, self.wq_piv
        ])

        self.add_couples(self.series)

    def add_couples(self, series):
        names = series.get_name_all_files()

        self.wq0.add_name_files(names)
        self.wq_images.add_couples(series)

        k, o = self.wq0.popitem()
        im = self.wq0.work(o)
        self.wq0.fill_destination(k, im)

        self.piv_work._prepare_with_image(im)
