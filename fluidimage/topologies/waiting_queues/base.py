from __future__ import print_function

import os
from copy import deepcopy
from time import time

import multiprocessing
import threading
import Queue

from fluidimage import logger, log_memory_usage
from fluiddyn.util import (terminal_colors as term, time_as_str)
from ...data_objects.piv import ArrayCouple
from ...works import load_image


class WaitingQueueBase(dict):
    def __init__(self, name, work, destination=None, work_name=None,
                 topology=None):
        self.name = name
        self.work = work
        self.destination = destination

        if work_name is None:
            if hasattr(work, 'im_class') and hasattr(work, 'func_name'):
                cls = work.im_class
                work_name = (cls.__module__ + '.' + cls.__name__ + '.' +
                             work.func_name)
            elif hasattr(work, 'func_name'):
                work_name = work.__module__ + '.' + work.func_name

        self.work_name = work_name
        self.topology = topology
        self._keys = []
        self._nb_processes = 0

    def __str__(self):
        return (term.OKBLUE + 'WaitingQueue ' + repr(self.name) + term.ENDC +
                ' with keys ' + repr(self._keys))

    def __setitem__(self, key, value):
        super(WaitingQueueBase, self).__setitem__(key, value)
        try:
            self._keys.remove(key)
        except ValueError:
            pass
        self._keys.append(key)

    def is_empty(self):
        return not bool(self)

    def check_and_act(self, sequential=None):
        k, o = self.popitem()
        logger.info()
        log_memory_usage(
            time_as_str(2) + ': launch work ' + self.work_name +
            '. mem usage')
        t_start = time()
        result = self.work(o)
        logger.info(
            'work {} ({}) done in {:.2f} s'.format(
                self.work_name, k, time() - t_start))
        self.fill_destination(k, result)

    def fill_destination(self, k, result):
        if self.destination is not None:
            self.destination[k] = result

    def update(self, d, keys=None):
        if keys is None:
            keys = list(d.keys())
        if not set(d.keys()) == set(keys):
            raise ValueError
        self._keys += keys
        super(WaitingQueueBase, self).update(d)

    def popitem(self):
        k = self._keys.pop(0)
        o = super(WaitingQueueBase, self).pop(k)
        return k, o

    def is_destination_full(self):
        return (isinstance(self.destination, WaitingQueueBase) and
                len(self.destination) + self._nb_processes >=
                self.topology.nb_items_lim)


def exec_work_and_comm(work, o, comm):
    result = work(o)
    comm.put(result)


class WaitingQueueMultiprocessing(WaitingQueueBase):
    do_use_cpu = True

    @staticmethod
    def _Queue(*args, **kwargs):
        return multiprocessing.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return multiprocessing.Process(*args, **kwargs)

    def enough_workers(self):
        return self.topology.nb_workers_cpu >= self.topology.nb_max_workers

    def check_and_act(self, sequential=False):

        if sequential:
            return WaitingQueueBase.check_and_act(self, sequential=sequential)

        workers = []
        i_launch = 0
        while (i_launch < self.topology.nb_max_launch and
               not self.is_empty() and
               not self.is_destination_full() and
               not self.enough_workers()):
            workers.append(self._launch_worker())
            i_launch += 1

        return workers

    def _launch_worker(self):

        k = self._keys.pop(0)
        o = self.pop(k)

        log_memory_usage(
            time_as_str(2) + ': launch work ' + self.work_name +
            ' ({}). mem usage'.format(k))

        comm = self._Queue()
        p = self._Process(target=exec_work_and_comm, args=(self.work, o, comm))
        t_start = time()
        p.start()
        self._nb_processes += 1
        if self.do_use_cpu:
            self.topology.nb_workers_cpu += 1
        else:
            self.topology.nb_workers_io += 1
        p.do_use_cpu = self.do_use_cpu

        def fill_destination():
            if isinstance(p, multiprocessing.Process):
                if p.exitcode:
                    logger.info(
                        'Error in work (process): '
                        'work_name = {}; key = {}; exitcode = {}'.format(
                            self.work_name, k, p.exitcode))
                    self._nb_processes -= 1
                    self.topology.nb_workers_cpu -= 1
                    return True

                try:
                    result = comm.get_nowait()
                    is_done = True
                except Queue.Empty:
                    return False
            else:
                if p.exitcode:
                    logger.info(
                        'Error in work (thread): '
                        'work_name = {}; key = {}; exitcode = {}'.format(
                            self.work_name, k, p.exitcode))
                    raise p.exception

                is_done = not p.is_alive()

            if not is_done:
                return False
            else:
                if not isinstance(p, multiprocessing.Process):
                    result = comm.get()
                    self.topology.nb_workers_io -= 1
                else:
                    self.topology.nb_workers_cpu -= 1
                logger.info(
                    'work {} ({}) done in {:.2f} s'.format(
                        self.work_name, k, time() - t_start))
                self.fill_destination(k, result)
                self._nb_processes -= 1
                return True

        p.fill_destination = fill_destination
        return p


class ThreadHandlingExceptions(threading.Thread):

    def __init__(self, *args, **kwargs):
        self.exitcode = None
        super(ThreadHandlingExceptions, self).__init__(*args, **kwargs)
        self.daemon = True

    def run(self):
        try:
            super(ThreadHandlingExceptions, self).run()
        except Exception as e:
            self.exitcode = 1
            self.exception = e


class WaitingQueueThreading(WaitingQueueMultiprocessing):
    do_use_cpu = False

    @staticmethod
    def _Queue(*args, **kwargs):
        return Queue.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return ThreadHandlingExceptions(*args, **kwargs)

    def enough_workers(self):
        return self.topology.nb_workers_io >= self.topology.nb_max_workers_io


class WaitingQueueLoadFile(WaitingQueueThreading):
    def __init__(self, *args, **kwargs):
        self.path_dir = kwargs.pop('path_dir')
        super(WaitingQueueLoadFile, self).__init__(*args, **kwargs)
        self.work_name = __name__ + '.load'

    def add_name_files(self, names):
        self.update({name: os.path.join(self.path_dir, name)
                     for name in names}, names)


class WaitingQueueLoadImage(WaitingQueueLoadFile):
    def __init__(self, *args, **kwargs):
        super(WaitingQueueLoadImage, self).__init__(
            'image file', load_image, *args, **kwargs)


class WaitingQueueMakeCouple(WaitingQueueBase):

    def __init__(self, name, destination,
                 work_name='make couples', topology=None):

        self.nb_couples_to_create = {}
        self.couples = set()
        self.series = {}
        self.topology = topology
        work = 'make_couples'

        super(WaitingQueueMakeCouple, self).__init__(
            name, work, destination, work_name, topology)

    def is_empty(self):
        return len(self.couples) == 0

    def add_series(self, series):

        self.series.update({serie.get_name_files(): deepcopy(serie)
                            for serie in series})

        couples = [serie.get_name_files() for serie in series]

        if len(couples) > 0 and len(couples[0]) != 2:
            raise ValueError(
                'A couple has to be of length 2.')

        self.couples.update(couples)
        nb = self.nb_couples_to_create

        for couple in couples:
            for name in couple:
                if name in nb:
                    nb[name] = nb[name] + 1
                else:
                    nb[name] = 1

    def check_and_act(self, sequential=None):
        if self.is_destination_full():
            return

        for k0 in self.keys():
            for k1 in self.keys():
                if (k1, k0) in self.couples:
                    k0, k1 = k1, k0

                if (k0, k1) in self.couples:
                    newk = k0 + '-' + k1
                    self.couples.remove((k0, k1))
                    serie = self.series.pop((k0, k1))

                    if self.nb_couples_to_create[k0] == 1:
                        v0 = self.pop(k0)
                        del self.nb_couples_to_create[k0]
                        self._keys.remove(k0)
                    else:
                        v0 = self[k0]
                        self.nb_couples_to_create[k0] = \
                            self.nb_couples_to_create[k0] - 1

                    if self.nb_couples_to_create[k1] == 1:
                        v1 = self.pop(k1)
                        del self.nb_couples_to_create[k1]
                        self._keys.remove(k1)
                    else:
                        v1 = self[k1]
                        self.nb_couples_to_create[k1] = \
                            self.nb_couples_to_create[k1] - 1

                    self.destination[newk] = ArrayCouple(
                        (k0, k1), (v0, v1), serie)
