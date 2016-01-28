
class Topology(object):
    def __init__(self, queues):
        self.queues = queues

    def run(self):

        workers = []
        workers_cpu = []
        while any([not q.is_empty() for q in self.queues]) or len(workers) > 0:
            for q in self.queues:
                if not q.is_empty():
                    print(q.work)
                    new_workers = q.check_and_act()
                    if new_workers is not None:
                        for worker in new_workers:
                            workers.append(worker)
                            if hasattr(worker, 'do_use_cpu') and \
                               worker.do_use_cpu:
                                workers_cpu.append(worker)

            workers[:] = [w for w in workers
                          if not w.fill_destination()]

            workers_cpu[:] = [w for w in workers_cpu
                              if w.is_alive()]

    def make_code_graphviz(self, name_file):
        """Generate the graphviz / dot code."""

        code = 'digraph {\nrankdir = LR\ncompound=true\n'
        # waiting queues
        code += '\nnode [shape="record"]\n'

        txt_queue = '"{name}"\t[label="<f0> {name}|' + '|'.join(
            ['<f{}>'.format(i) for i in range(1, 5)]) + '"]\n'

        for queue in self.queues:
            code += txt_queue.format(name=queue.name)

        # works and links
        code += "\nnode [shape=\"ellipse\"]\n"

        txt_work = '"{name}"\t[label="{name}"]'
        for queue in self.queues:
            name_work = queue.work_name or str(queue.work)
            code += txt_work.format(name=name_work)
            code += '"{}" -> "{}"'.format(queue.name, name_work)
            if hasattr(queue.destination, 'name'):
                code += '"{}" -> "{}"'.format(
                    name_work, queue.destination.name)

        code += '}\n'

        with open(name_file, 'w') as f:
            f.write(code)


if __name__ == '__main__':
    from copy import copy

    from waiting_queues import (
        WaitingQueueMultiprocessing, WaitingQueueThreading,
        WaitingQueueMakeCouple, WaitingQueueLoadFile)

    from works import load_image, PIVWork

    from fluiddyn.util.serieofarrays import \
        SerieOfArraysFromFiles, SeriesOfArrays

    serie_arrays = SerieOfArraysFromFiles('Data/images/im*')

    def give_indslices_from_indserie(iserie):
        indslices = copy(serie_arrays._index_slices_all_files)
        indslices[0] = [iserie+1, iserie+3]
        return indslices

    series = SeriesOfArrays(serie_arrays, give_indslices_from_indserie)
    couples = [serie.get_name_files() for serie in series]
    names = series.get_name_all_files()

    piv_work = PIVWork()

    results = {}

    wq_result = WaitingQueueThreading(
        'delta', lambda o: o.save('Data/images.piv'), results,
        work_name='save')
    wq_couples = WaitingQueueMultiprocessing(
        'couple', piv_work.calcul, wq_result, work_name='PIV')
    wq_images = WaitingQueueMakeCouple(
        'array image', wq_couples, couples)
    wq0 = WaitingQueueLoadFile(
        'file', load_image, wq_images, path_dir=series.serie.path_dir)
    wq0.add_name_files(names)

    topology = Topology([wq0, wq_images, wq_couples, wq_result])
    topology.run()

    topology.make_code_graphviz('topo.dot')
