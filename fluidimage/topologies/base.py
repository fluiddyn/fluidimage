"""Topology base
================

.. autoclass:: TopologyBase
   :members:
   :private-members:

"""
from __future__ import print_function

from time import sleep, time
from multiprocessing import cpu_count
from signal import signal
import re
import sys
from fluiddyn.util import terminal_colors as term
from fluidimage.util.util import logger, log_memory_usage

from ..config import get_config
from .waiting_queues.base import WaitingQueueThreading

config = get_config()

dt = 0.5  # s

nb_cores = cpu_count()
overloading_coef = 2

if config is not None:
    try:
        allow_hyperthreading = eval(config['topology']['allow_hyperthreading'])
    except KeyError:
        allow_hyperthreading = True

try:  # should work on UNIX

    # found in http://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python # noqa
    with open('/proc/self/status') as f:
        status = f.read()
    m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', status)
    if m:
        nb_cpus_allowed = bin(int(m.group(1).replace(',', ''), 16)).count('1')

    if nb_cpus_allowed > 0:
        nb_cores = nb_cpus_allowed
        print('Cpus_allowed: {}'.format(nb_cpus_allowed))

    if allow_hyperthreading is False:
        with open('/proc/cpuinfo') as f:
            cpuinfo = f.read()

        nb_proc_tot = 0
        siblings = None
        for line in cpuinfo.split('\n'):
            if line.startswith('processor'):
                nb_proc_tot += 1
            if line.startswith('siblings') and siblings is None:
                siblings = int(line.split()[-1])

        if nb_proc_tot == siblings * 2:
            print('We do not use hyperthreading.')
            nb_cores /= 2

except IOError:
    pass


if config is not None:
    try:
        nb_cores = eval(config['topology']['nb_cores'])
    except KeyError:
        pass

    try:
        overloading_coef = eval(config['topology']['overloading_coef'])
    except KeyError:
        pass


nb_max_workers = nb_cores * overloading_coef


class TopologyBase(object):

    def __init__(self, queues):
        self.queues = queues
        self.nb_cores = nb_cores
        self.nb_items_lim = max(nb_cores, 2)

        self._has_to_stop = False

        if sys.platform != 'win32':
            def handler_signals(signal_number, stack):
                print('signal {} received: set _has_to_stop to True'.format(
                    signal_number))
                self._has_to_stop = True

            signal(12, handler_signals)

    def compute(self, sequential=None, has_to_exit=True):

        t_start = time()

        print('Start compute.')
        log_memory_usage('Memory usage at the beginning of compute', 'OKGREEN')

        workers = []
        workers_cpu = []
        while (not self._has_to_stop and
               (any([not q.is_empty() for q in self.queues]) or
                len(workers) > 0)):
            self.nb_workers_cpu = len(workers_cpu)
            self.nb_workers = len(workers)

            # slow down this loop...
            sleep(0.05)
            if self.nb_workers_cpu >= nb_max_workers:
                logger.debug('{} Saturated workers: {}, sleep {} s {}'.format(
                    term.WARNING, self.nb_workers_cpu, dt, term.ENDC))
                sleep(dt)

            for q in self.queues:
                if not q.is_empty():
                    logger.debug(q)
                    logger.debug('check_and_act for work: ' + repr(q.work))
                    new_workers = q.check_and_act(sequential=sequential)
                    if new_workers is not None:
                        for worker in new_workers:
                            workers.append(worker)
                            if hasattr(worker, 'do_use_cpu') and \
                               worker.do_use_cpu:
                                workers_cpu.append(worker)
                    logger.debug('workers:' + repr(workers))
                    logger.debug('workers_cpu:' + repr(workers_cpu))

            workers[:] = [w for w in workers
                          if not w.fill_destination()]

            workers_cpu[:] = [w for w in workers_cpu
                              if w.is_alive()]

        if self._has_to_stop:
            logger.info(term.FAIL + 'Will exist because of signal 12. ' +
                        'Waiting for all workers to finish...' + term.ENDC)

            q = self.queues[-1]

            # if the last queue is a WaitingQueueThreading (saving),
            # it is also emptied.
            while (len(workers) > 0 or
                   (not q.is_empty() and
                    isinstance(q, WaitingQueueThreading))):

                sleep(0.2)

                if not q.is_empty() and isinstance(q, WaitingQueueThreading):
                    new_workers = q.check_and_act(sequential=sequential)
                    if new_workers is not None:
                        for worker in new_workers:
                            workers.append(worker)

                workers[:] = [w for w in workers
                              if not w.fill_destination()]

        self._print_at_exit(time() - t_start)
        log_memory_usage('Memory usage at the exit', 'OKGREEN')

        if self._has_to_stop and has_to_exit:
            logger.info(term.FAIL + 'Exit with signal 99.' + term.ENDC)
            exit(99)

    def _print_at_exit(self, time_since_start):

        txt = 'Stop compute after t = {:.2f} s'.format(time_since_start)
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += (' ({} results, {:.2f} s/result).'.format(
                nb_results, time_since_start / nb_results))
        else:
            txt += '.'

        print(txt)

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

        print('A graph can be produced with one of these commands:\n'
              'dot topo.dot -Tpng -o topo.png\n'
              'dot topo.dot -Tx11')
