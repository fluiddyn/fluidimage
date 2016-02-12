
from base import TopologyBase

from waiting_queues import (
    WaitingQueueMultiprocessing, WaitingQueueThreading,
    WaitingQueueMakeCouple, WaitingQueueLoadFile)

from works import load_image, PIVWork


class TopologyPIV(TopologyBase):
    def __init__(self):

        piv_work = PIVWork()

        results = {}
        wq_result = WaitingQueueThreading(
            'delta', lambda o: o.save('Data/images.piv'), results,
            work_name='save')
        wq_couples = WaitingQueueMultiprocessing(
            'couple', piv_work.calcul, wq_result, work_name='PIV')
        self.wq_images = WaitingQueueMakeCouple(
            'array image', wq_couples)
        self.wq0 = WaitingQueueLoadFile(
            'file', load_image, self.wq_images,
            path_dir=series.serie.path_dir)

        super(TopologyPIV, self).__init__([
            self.wq0, self.wq_images, wq_couples, wq_result])

    def add_couples(self, series):
        couples = [serie.get_name_files() for serie in series]
        names = series.get_name_all_files()

        self.wq0.add_name_files(names)
        self.wq_images.add_couples(couples)


if __name__ == '__main__':

    from fluiddyn.util.serieofarrays import \
        SerieOfArraysFromFiles, SeriesOfArrays

    serie_arrays = SerieOfArraysFromFiles('Data/images/im*')
    series = SeriesOfArrays(serie_arrays, 'i+1:i+3')

    topology = TopologyPIV()
    topology.add_couples(series)

    topology.run()

    topology.make_code_graphviz('topo.dot')
    # then the graph can be produced with the command:
    # dot topo.dot -Tpng -o topo.png
