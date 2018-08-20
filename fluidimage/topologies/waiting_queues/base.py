"""Waiting queues classes (:mod:`fluidimage.topologies.waiting_queues.base`)
============================================================================

.. autoclass:: WaitingQueueBase
   :members:
   :private-members:

.. autoclass:: WaitingQueueMultiprocessing
   :members:
   :private-members:

.. autoclass:: ThreadWork
   :members:
   :private-members:

.. autoclass:: WaitingQueueThreading
   :members:
   :private-members:

.. autoclass:: WaitingQueueLoadFile
   :members:
   :private-members:

.. autoclass:: WaitingQueueLoadImage
   :members:
   :private-members:

.. autoclass:: WaitingQueueMakeCouple
   :members:
   :private-members:

"""


from __future__ import print_function

import os
from copy import deepcopy
from time import time

import multiprocessing
import threading

try:
    import queue
except ImportError:
    # python 2
    import Queue as queue

from ...data_objects.piv import ArrayCouple, ArrayCoupleBOS
from ...works import load_image
from ...util.util import logger, log_memory_usage, cstring


class WaitingQueueBase(dict):
    def __init__(
        self, name, work, destination=None, work_name=None, topology=None
    ):
        self.name = name
        self.work = work
        self.destination = destination

        if work_name is None:
            if hasattr(work, "im_class") and hasattr(work, "func_name"):
                cls = work.im_class
                work_name = (
                    cls.__module__ + "." + cls.__name__ + "." + work.func_name
                )
            elif hasattr(work, "func_name"):
                work_name = work.__module__ + "." + work.func_name

            else:
                try:
                    cls = work.__class__
                    work_name = (
                        cls.__module__ + "." + cls.__name__ + "." + work.__name__
                    )
                except AttributeError:
                    work_name = work.__module__ + "." + work.__name__

        self.work_name = work_name
        self.topology = topology
        self._keys = []
        self._nb_workers = 0

    def __str__(self):
        length = len(self._keys)
        if length == 0:
            keys = []
        elif length < 5:
            keys = self._keys
        else:
            index = range(min(length, 3))
            keys = [self._keys[i] for i in index]
            keys.extend(["...", self._keys[-1]])

        length = str(length)
        return cstring(
            "WaitingQueue",
            repr(self.name),
            "with keys",
            repr(keys),
            "(" + length + " items)",
        )

    def __setitem__(self, key, value):
        if key in self._keys:
            self._keys.remove(key)
        self._keys.append(key)
        super(WaitingQueueBase, self).__setitem__(key, value)

    def is_empty(self):
        return not bool(self)

    def check_and_act(self, sequential=None):
        if self.is_empty() or self.is_destination_full():
            return

        k, o = self.popitem()
        log_memory_usage(
            "{:.2f} s. ".format(time() - self.topology.t_start)
            + "Launch work "
            + self.work_name
            + " ({}). mem usage".format(k)
        )

        t_start = time()
        result = self.work(o)
        logger.info(
            "work {} ({}) done in {:.2f} s".format(
                self.work_name, k, time() - t_start
            )
        )
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
        return (
            isinstance(self.destination, WaitingQueueBase)
            and len(self.destination) + self._nb_workers
            >= self.topology.nb_items_lim
        )


def exec_work_and_comm(work, o, comm, comm_started):
    # try:
    #     name = o.name
    # except AttributeError:
    #     name = repr(o)
    # print('start', work, name)
    comm_started.put(True)
    result = work(o)
    # print('work finished, communication.', work, name)
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
        while (
            i_launch < self.topology.nb_max_launch
            and not self.is_empty()
            and not self.is_destination_full()
            and not self.enough_workers()
        ):
            workers.append(self._launch_worker())
            i_launch += 1

        if i_launch >= self.topology.nb_max_launch:
            logger.debug("stop launching because nb_max_launch.")

        elif self.is_empty():
            logger.debug("stop launching because the queue is empty.")

        elif self.is_destination_full():
            logger.debug("stop launching because the destination is full.")

        elif self.enough_workers():
            logger.debug("stop launching because the workers are saturated.")
        return workers

    def _launch_worker(self):
        k = self._keys[0]
        o = self[k]
        log_memory_usage(
            "{:.2f} s. ".format(time() - self.topology.t_start)
            + "Launch work "
            + self.work_name
            + " ({}). mem usage".format(k)
        )

        comm = self._Queue()
        comm_started = self._Queue()
        p = self._Process(
            target=exec_work_and_comm, args=(self.work, o, comm, comm_started)
        )
        p.t_start = t_start = time()
        p.comm = comm
        p.key = k
        p.work_name = self.work_name

        # to handle a bug py3 multiprocessing
        p.comm_started = comm_started
        p.really_started = False
        p.start()

        # we do this after p.start() because an error can be raised here
        assert k == self._keys.pop(0)
        assert o is self.pop(k)

        self._nb_workers += 1
        if self.do_use_cpu:
            self.topology.nb_workers_cpu += 1
        else:
            self.topology.nb_workers_io += 1
        p.do_use_cpu = self.do_use_cpu

        def fill_destination():
            if isinstance(p, multiprocessing.Process):
                if p.exitcode:
                    logger.error(
                        cstring(
                            "Error in work (Process): "
                            "work_name = {}; key = {}; exitcode = {}".format(
                                self.work_name, k, p.exitcode
                            ),
                            color="FAIL",
                        )
                    )
                    self._nb_workers -= 1
                    self.topology.nb_workers_cpu -= 1
                    return True

                try:
                    result = comm.get_nowait()
                    is_done = True
                except queue.Empty:
                    # strange bug
                    if not p.is_alive():
                        logger.exception(
                            cstring(
                                "not p.is_alive() but nothing in the communication"
                                " queue. Result (" + k + ") has been lost :-(",
                                color="FAIL",
                            )
                        )
                        self.topology.nb_workers_cpu -= 1
                        self._nb_workers -= 1
                        return True

                    return False

            else:
                if p.exitcode:
                    logger.error(
                        cstring(
                            "Error in work (thread): "
                            "work_name = {}; key = {}; exitcode = {}".format(
                                self.work_name, k, p.exitcode
                            ),
                            color="FAIL",
                        )
                    )
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
                    "work {} ({}) done in {:.2f} s".format(
                        self.work_name, k, time() - t_start
                    )
                )
                self.fill_destination(k, result)
                self._nb_workers -= 1
                return True

        p.fill_destination = fill_destination
        return p


class WaitingQueueMultiprocessingSpe(WaitingQueueMultiprocessing):
    def _launch_worker(self):

        k = self._keys[0]
        o = self[k]
        print(o)
        log_memory_usage(
            "{:.2f} s. ".format(time() - self.topology.t_start)
            + "Launch work "
            + self.work_name
            + " ({}). mem usage".format(k)
        )

        comm = self._Queue()
        comm_started = self._Queue()
        p = self._Process(
            target=exec_work_and_comm, args=(self.work, o, comm, comm_started)
        )
        p.t_start = t_start = time()
        p.comm = comm
        p.key = k
        p.work_name = self.work_name

        # to handle a bug py3 multiprocessing
        p.comm_started = comm_started
        p.really_started = False
        p.start()

        # we do this after p.start() because an error can be raised here

        self._nb_workers += 1
        if self.do_use_cpu:
            self.topology.nb_workers_cpu += 1
        else:
            self.topology.nb_workers_io += 1
        p.do_use_cpu = self.do_use_cpu


class ThreadWork(threading.Thread):
    def __init__(self, *args, **kwargs):
        self.exitcode = None
        super(ThreadWork, self).__init__(*args, **kwargs)

    # self.daemon = True

    def run(self):
        try:
            super(ThreadWork, self).run()
        except Exception as e:
            self.exitcode = 1
            self.exception = e


class WaitingQueueThreading(WaitingQueueMultiprocessing):
    do_use_cpu = False
    nb_max_workers = 6

    @staticmethod
    def _Queue(*args, **kwargs):
        return queue.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return ThreadWork(*args, **kwargs)

    def enough_workers(self):
        return (
            self._nb_workers >= self.nb_max_workers
            or self.topology.nb_workers_io >= self.topology.nb_max_workers_io
        )


class WaitingQueueLoadFile(WaitingQueueThreading):
    nb_max_workers = 8

    def __init__(self, *args, **kwargs):
        self.path_dir = kwargs.pop("path_dir")
        super(WaitingQueueLoadFile, self).__init__(*args, **kwargs)
        self.work_name = __name__ + ".load"

    def add_name_files(self, names):
        self.update(
            {name: os.path.join(self.path_dir, name) for name in names}, names
        )


class WaitingQueueLoadImage(WaitingQueueLoadFile):
    def __init__(self, *args, **kwargs):
        super(WaitingQueueLoadImage, self).__init__(
            "image file", load_image, *args, **kwargs
        )
        self.work_name = __name__ + ".load_image"


def load_image_path(path):
    im = load_image(path)
    return im, path


class WaitingQueueLoadImagePath(WaitingQueueLoadFile):
    def __init__(self, *args, **kwargs):
        super(WaitingQueueLoadImagePath, self).__init__(
            "image file", load_image_path, *args, **kwargs
        )
        self.work_name = __name__ + ".load_image"


class WaitingQueueMakeCouple(WaitingQueueBase):
    def __init__(
        self, name, destination, work_name="make couples", topology=None
    ):

        self.nb_couples_to_create = {}
        self.couples = set()
        self.series = {}
        self.topology = topology
        work = "make_couples"

        super(WaitingQueueMakeCouple, self).__init__(
            name, work, destination, work_name, topology
        )

    def is_empty(self):
        return len(self.couples) == 0

    def add_series(self, series):

        self.series.update(
            {serie.get_name_arrays(): deepcopy(serie) for serie in series}
        )

        couples = [serie.get_name_arrays() for serie in series]

        if len(couples) > 0 and len(couples[0]) != 2:
            raise ValueError("A couple has to be of length 2.")

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

        for k0 in list(self.keys()):
            for k1 in list(self.keys()):
                if (k1, k0) in self.couples:
                    k0, k1 = k1, k0

                if (k0, k1) in self.couples:
                    newk = k0 + "-" + k1
                    self.couples.remove((k0, k1))
                    serie = self.series.pop((k0, k1))

                    if self.nb_couples_to_create[k0] == 1:
                        v0 = self.pop(k0)
                        del self.nb_couples_to_create[k0]
                        self._keys.remove(k0)
                    else:
                        v0 = self[k0]
                        self.nb_couples_to_create[k0] = (
                            self.nb_couples_to_create[k0] - 1
                        )

                    if self.nb_couples_to_create[k1] == 1:
                        v1 = self.pop(k1)
                        del self.nb_couples_to_create[k1]
                        self._keys.remove(k1)
                    else:
                        v1 = self[k1]
                        self.nb_couples_to_create[k1] = (
                            self.nb_couples_to_create[k1] - 1
                        )

                    self.destination[newk] = ArrayCouple(
                        (k0, k1),
                        (v0, v1),
                        serie,
                        params_mask=self.topology.params.mask,
                    )


class WaitingQueueOneShot(WaitingQueueBase):
    def __init__(
        self,
        name,
        destination,
        work,
        work_name="make couples",
        topology=None,
        image_reference=None,
        path_reference=None,
        serie=None,
    ):
        self.topology = topology
        work = "make_couples"

        self.image_reference = image_reference
        self.path_reference = path_reference
        self.serie = serie
        self.work = work

        self.topology = topology
        work = "make_couples"
        self.image_reference = image_reference
        self.path_reference = path_reference
        self.serie = serie

        super(WaitingQueueOneShot, self).__init__(
            name, work, destination, work_name, topology
        )

    def is_empty(self):
        return WaitingQueueBase.is_empty(WaitingQueueBase)

    def check_and_act(self, sequential=None):
        if self.is_destination_full():
            return
        print(type(self.work))


class WaitingQueueMakeCoupleBOS(WaitingQueueBase):
    def __init__(
        self,
        name,
        destination,
        work_name="make couples",
        topology=None,
        image_reference=None,
        path_reference=None,
        serie=None,
    ):

        self.topology = topology
        work = "make_couples"

        self.image_reference = image_reference
        self.path_reference = path_reference
        self.serie = serie

        super(WaitingQueueMakeCoupleBOS, self).__init__(
            name, work, destination, work_name, topology
        )

        self.topology = topology
        work = "make_couples"

        self.image_reference = image_reference
        self.path_reference = path_reference
        self.serie = serie

        super(WaitingQueueMakeCoupleBOS, self).__init__(
            name, work, destination, work_name, topology
        )

    def check_and_act(self, sequential=None):
        if self.is_destination_full():
            return

        nb_images_per_check = 100
        ind_image = 0
        for k in list(self.keys()):
            v = self.pop(k)

            paths = (self.path_reference, os.path.join(self.serie.path_dir, k))

            self.destination[k] = ArrayCoupleBOS(
                ("ref", k),
                (self.image_reference, v),
                self.serie,
                paths=paths,
                params_mask=self.topology.params.mask,
            )

            ind_image += 1
            if ind_image > nb_images_per_check:
                break
