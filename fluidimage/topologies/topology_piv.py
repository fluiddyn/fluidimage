
from fluidimage.topologies.base import TopologyBase

from fluidimage.waiting_queues.base import (
    WaitingQueueMultiprocessing,  # , WaitingQueueThreading,
    WaitingQueueMakeCouple, WaitingQueueLoadImage)

from fluidimage.works.piv import FirstPIVWork


class TopologyPIV(TopologyBase):
    def __init__(self):

        self.piv_work = FirstPIVWork(n_interrogation_window=48, overlap=0.5)

        # results = {}
        # wq_result = WaitingQueueThreading(
        #     'delta', lambda o: o.save('Data/images.piv'), results,
        #     work_name='save')
        self.wq_result = {}

        self.wq_couples = WaitingQueueMultiprocessing(
            'couple', self.piv_work.calcul, self.wq_result, work_name='PIV')
        self.wq_images = WaitingQueueMakeCouple(
            'array image', self.wq_couples)
        self.wq0 = WaitingQueueLoadImage(
            destination=self.wq_images,
            path_dir=series.serie.path_dir)

        super(TopologyPIV, self).__init__([
            self.wq0, self.wq_images, self.wq_couples
            # , wq_result
        ])

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

    from fluiddyn.util.serieofarrays import \
        SerieOfArraysFromFiles, SeriesOfArrays

    # path = '../../image_samples/Oseen/Oseen_center*'
    # strcouple = 'i+1:i+3'

    path = '../../image_samples/Karman'
    strcouple = '2*i+1:2*i+3'

    serie_arrays = SerieOfArraysFromFiles(path)
    series = SeriesOfArrays(serie_arrays, strcouple)

    topology = TopologyPIV()
    topology.add_couples(series)

    topology.compute()

    # topology.make_code_graphviz('topo.dot')
    # then the graph can be produced with the command:
    # dot topo.dot -Tpng -o topo.png
