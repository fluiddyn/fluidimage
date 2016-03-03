
from __future__ import print_function

from time import sleep
from multiprocessing import cpu_count

nb_cores = cpu_count()
dt = 0.2  # s


class TopologyBase(object):

    def __init__(self, queues):
        self.queues = queues
        self.nb_cores = nb_cores
        self.nb_items_lim = max(nb_cores, 2)

    def compute(self, sequential=None):

        workers = []
        workers_cpu = []
        while any([not q.is_empty() for q in self.queues]) or len(workers) > 0:
            self.nb_workers_cpu = len(workers_cpu)
            self.nb_workers = len(workers)
            if self.nb_workers_cpu >= nb_cores:
                # slow down this loop...
                print('sleep {} s'.format(dt))
                sleep(dt)
            for q in self.queues:
                print(q)
                if not q.is_empty():
                    print('check_and_act for work:', q.work)
                    new_workers = q.check_and_act(sequential=sequential)
                    if new_workers is not None:
                        for worker in new_workers:
                            workers.append(worker)
                            if hasattr(worker, 'do_use_cpu') and \
                               worker.do_use_cpu:
                                workers_cpu.append(worker)
                    print('workers:', workers)
                    print('workers_cpu:', workers_cpu)

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
