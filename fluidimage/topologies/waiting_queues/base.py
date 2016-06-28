
import os
from copy import deepcopy, copy
import logging

import multiprocessing
import threading
import Queue

from fluiddyn.util.util import print_memory_usage
from fluiddyn.util import terminal_colors as term
from ...data_objects.piv import ArrayCouple
from ...data_objects.pre_proc import ArraySerie
from ...works import load_image


logger = logging.getLogger('fluidimage')


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
        result = self.work(o)
        self.fill_destination(k, result)

    def fill_destination(self, k, result):
        if self.destination is not None:
            self.destination[k] = result

    def update(self, d, keys):
        if not set(d.keys()) == set(keys):
            raise ValueError
        self._keys += keys
        super(WaitingQueueBase, self).update(d)

    def popitem(self):
        k = self._keys.pop(0)
        o = super(WaitingQueueBase, self).pop(k)
        return k, o


class WaitingQueueMultiprocessing(WaitingQueueBase):
    do_use_cpu = True

    @staticmethod
    def _Queue(*args, **kwargs):
        return multiprocessing.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return multiprocessing.Process(*args, **kwargs)

    def is_destination_full(self):
        cond_instance = isinstance(self.destination, WaitingQueueBase)
        cond_nb_items = len(self.destination) >= self.topology.nb_items_lim
        return (cond_instance and cond_nb_items)

    def check_and_act(self, sequential=None):

        if sequential:
            return WaitingQueueBase.check_and_act(self, sequential=sequential)

        if self.do_use_cpu and \
           self.topology.nb_workers_cpu >= self.topology.nb_cores:
            return

        if self.is_destination_full():
            return

        logger.info('launch work ' + self.work_name)

        k = self._keys.pop(0)
        o = self.pop(k)
        comm = self._Queue()

        def f(comm):
            result = self.work(o)
            comm.put(result)

        p = self._Process(target=f, args=(comm,))
        p.start()
        p.do_use_cpu = self.do_use_cpu

        def fill_destination():
            if isinstance(p, multiprocessing.Process):
                try:
                    result = comm.get_nowait()
                    is_done = True
                except Queue.Empty:
                    is_done = False
            else:
                is_done = not p.is_alive()

            if not is_done:
                return False
            else:
                if isinstance(p, multiprocessing.Process):
                    p.terminate()
                else:
                    result = comm.get()
                self.fill_destination(k, result)
                return True

        p.fill_destination = fill_destination
        print_memory_usage(
            term.WARNING + 'Memory usage after one work: ' +
            self.work_name + term.ENDC)
        return [p]


class WaitingQueueThreading(WaitingQueueMultiprocessing):
    do_use_cpu = False

    @staticmethod
    def _Queue(*args, **kwargs):
        return Queue.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return threading.Thread(*args, **kwargs)


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


class WaitingQueueLoadImageSeries(WaitingQueueLoadFile):

    def __init__(self, *args, **kwargs):
        super(WaitingQueueLoadImageSeries, self).__init__(
            'load image', load_image, *args, **kwargs)

    def is_destination_full(self):
        cond_instance = isinstance(self.destination, WaitingQueueBase)
        buffer_limit = max(self.topology.nb_items_lim,
                           self.topology.nb_items_per_serie)

        cond_nb_items = len(self.destination) >= buffer_limit
        return (cond_instance and cond_nb_items)


class WaitingQueueMakeSerie(WaitingQueueBase):
    """
    The difference from `WaitingQueueMakeCouple` is that
    the following attributes are replaced:
    .. `self.couples` --> `self.serie_set`
    .. `self.nb_couples_to_create` --> `self.nb_serie_to_create`

    Allowing the dictionary to contain a serie of images,
    and not just a couple (2 images).

    """
    def __init__(self, name, destination,
                 work_name='make serie', topology=None):

        self.nb_series = 0
        self.ind_series = []
        self.nb_serie_to_create = {}
        self.serie_set = set()
        self.series = {}
        self.topology = topology
        work = 'make_serie'
        super(WaitingQueueMakeSerie, self).__init__(
            name, work, destination, work_name, topology)

    def is_empty(self):
        return len(self.serie_set) == 0

    def add_series(self, series):

        self.series.update({serie.get_name_files(): deepcopy(serie)
                            for serie in series})

        serie_set = [serie.get_name_files() for serie in series]
        self.serie_set.update(serie_set)
        self.nb_series = len(serie_set)
        self.ind_series = range(self.nb_series)
        nb = self.nb_serie_to_create

        for names in serie_set:
            for name in names:
                if name in nb:
                    nb[name] = nb[name] + 1
                else:
                    nb[name] = 1

    def check_and_act(self, sequential=None):
        for names in copy(self.serie_set):
            if all([name in self for name in names]):
                logger.info('launch work with ' + self.work_name + repr(names))
                k0 = names[0]
                k1 = names[-1]
                newk = k0 + '-' + k1

                self.serie_set.remove(names)
                serie = self.series.pop(names)
                ind_serie = self.ind_series.pop(0)

                values = []
                logger.debug('Creating a serie for ' + repr(names))
                for name in names:
                    if self.nb_serie_to_create[name] == 1:
                        values.append(self.pop(name))
                        del self.nb_serie_to_create[name]
                        self._keys.remove(name)
                    else:
                        values.append(self[name])
                        self.nb_serie_to_create[name] -= 1

                self.destination[newk] = ArraySerie(
                    names, values, serie, ind_serie, self.nb_series)
