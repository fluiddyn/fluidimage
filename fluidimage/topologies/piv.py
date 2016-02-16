
import os

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.serieofarrays import \
    SerieOfArraysFromFiles, SeriesOfArrays

from fluidimage.topologies.base import TopologyBase

from fluidimage.waiting_queues.base import (
    WaitingQueueMultiprocessing, WaitingQueueThreading,
    WaitingQueueMakeCouple, WaitingQueueLoadImage)

from fluidimage.works.piv import FirstPIVWork


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
        params._set_child('piv', attribs={
            'n_interrogation_window': 48,
            'overlap': 0.5})
        return params

    def __init__(self, params=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params

        n_interrogation_window = params.piv.n_interrogation_window
        overlap = params.piv.overlap

        self.piv_work = FirstPIVWork(
            n_interrogation_window=n_interrogation_window, overlap=overlap)

        serie_arrays = SerieOfArraysFromFiles(params.series.path)
        self.series = SeriesOfArrays(serie_arrays, params.series.strcouple)

        path_dir = self.series.serie.path_dir
        path_dir_result = path_dir + '.piv'

        if os.path.exists(path_dir_result):
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
            self.wq0, self.wq_images, self.wq_couples  # , self.wq_piv
        ])

        self.add_couples(self.series)

    def add_couples(self, series):
        couples = [serie.get_name_files() for serie in series]
        names = series.get_name_all_files()

        self.wq0.add_name_files(names)
        self.wq_images.add_couples(couples)

        k, o = self.wq0.popitem()
        im = self.wq0.work(o)
        self.wq0.fill_destination(k, im)

        self.piv_work.prepare_with_image(im)


if __name__ == '__main__':

    params = TopologyPIV.create_default_params()

    # path = '../../image_samples/Oseen/Images/Oseen_center*'
    path = '../../image_samples/Karman/Images'

    # path = '../../image_samples/Jet/Images/c*'
    # params.series.strcouple = 'i+60, 0:2'
    # params.series.strcouple = 'i+60:i+62, 0'

    params.series.path = path

    topology = TopologyPIV(params)

    topology.compute()

    topology.make_code_graphviz('topo.dot')
    # then the graph can be produced with the command:
    # dot topo.dot -Tpng -o topo.png
    # dot topo.dot -Tx11
