"""Executor async/await
=======================

This executer uses await/async with trio library to put topology tasks in
concurrency.

Two modes of computation are available:

- The single executor mode.

  A single executor (in one process) is created.  If CPU bounded tasks are limited
  by the Python GIL, the threads won't use at the same time the CPU.

  Meaning that the work will be done in a single thread, except if the topology
  computed has C code in it. In this case, the GIL is bypassed and computation can
  use many CPU.

- The multi-executor mode.

  Many executors are created, each executor works in a process from
  multiprocessing with a part of the work to do. The work split is done in the
  class "ExecutorAsyncMultiproc". Usefull for full python written topology. See
  "ExecutorAsyncMultiproc" for more information.

.. autoclass:: ExecutorAsync
   :members:
   :private-members:

"""

import time
import collections

import trio

from fluidimage.util import logger, log_memory_usage

from .base import ExecutorBase


def popitem(input_queue):
    """Get an item from the input_queue."""
    key, obj = input_queue.popitem()
    return key, obj


def push(key, obj, output_queue):
    """
    Add an item (key, obj) in the output_queue
    :param key: A dictionnary key
    :param obj: A dictionnary value
    :param output_queue: a dictionary
    """
    output_queue[key] = obj
    return


class ExecutorAsync(ExecutorBase):
    """Executor async/await.

    Work in a single thread, except if the computed topology has C code.

    Parameters
    ----------

    nb_max_workers : None, int

      Limits the numbers of workers working in the same time.

    nb_items_queue_max : None, int

      Limits the numbers of items that can be in a output_queue.

    sleep_time : None, float

      Defines the waiting time (from trio.sleep) of a function. Functions await
      "trio.sleep" when they have done a work on an item, and when there is
      nothing in there input_queue.

    """

    def __init__(
        self,
        topology,
        path_dir_result,
        nb_max_workers=None,
        nb_items_queue_max=None,
        sleep_time=0.1,
    ):
        super().__init__(
            topology, path_dir_result, nb_max_workers, nb_items_queue_max
        )

        # Object variables
        self.nb_working_worker = 0

        # Executor parameters
        self.sleep_time = sleep_time

        # fonction containers
        self.works = []
        self.async_funcs = collections.OrderedDict()
        self.funcs = collections.OrderedDict()

        # Functions definition
        self.store_async_works()
        self.define_functions()
        # Queue0

    def compute(self):
        """Compute the whole topology.

        Begin by executing one shot jobs, then execute multiple shots jobs
        implemented as async functions.  Warning, one shot jobs must be ancestors
        of multiple shots jobs in the topology.

        """

        self._init_compute()
        self.exec_one_shot_job()
        trio.run(self.start_async_works)
        self._finalize_compute()

    async def start_async_works(self):
        """Create a trio nursery and start all async functions.

        """
        async with trio.open_nursery() as self.nursery:
            for af in reversed(self.async_funcs.values()):
                self.nursery.start_soon(af)
        return

    def define_functions(self):
        """Define sync and async functions.

        Define sync (one shot functions) and async functions (multiple shot
        functions), and store them in `self.async_funcs`.

        The behavior of the executor is mostly defined here.  To sum up : Each
        "multiple shot" works from the topology, waits for an items to be
        available in there input_queue. Then :return:

        """
        for w in reversed(self.topology.works):
            # One shot functions
            if w.kind is not None and "one shot" in w.kind:

                def func(work=w):
                    work.func_or_cls(work.input_queue, work.output_queue)

                func.func_or_cls = w.func_or_cls

                self.funcs[w.name] = func
                continue

            # global functions
            elif w.kind is not None and "global" in w.kind:

                async def func(work=w):
                    item_number = 1
                    while True:
                        while len(work.output_queue) > self.nb_items_queue_max:
                            await trio.sleep(self.sleep_time)
                        t_start = time.time()
                        while not work.func_or_cls(
                            work.input_queue, work.output_queue
                        ):
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                            t_start = time.time()
                        item_number += 1
                        log_memory_usage(
                            "{:.2f} s. ".format(time.time() - self.t_start)
                            + "Launch work "
                            + work.name.replace(" ", "_")
                            + " ({}). mem usage".format(item_number)
                        )
                        logger.info(
                            "work {} ({}) done in {:.3f} s".format(
                                work.name.replace(" ", "_"),
                                "item" + str(item_number),
                                time.time() - t_start,
                            )
                        )
                        await trio.sleep(self.sleep_time)

            # I/O
            elif (
                w.kind is not None
                and "io" in w.kind
                and w.output_queue is not None
            ):

                async def func(work=w):
                    while True:
                        while (
                            not work.input_queue
                            or self.nb_working_worker >= self.nb_max_workers
                            or len(work.output_queue) >= self.nb_items_queue_max
                        ):
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                        t_start = time.time()
                        key, obj = popitem(work.input_queue)
                        log_memory_usage(
                            "{:.2f} s. ".format(time.time() - self.t_start)
                            + "Launch work "
                            + work.name.replace(" ", "_")
                            + " ({}). mem usage".format(key)
                        )
                        ret = await trio.run_sync_in_worker_thread(
                            work.func_or_cls, obj
                        )
                        push(key, ret, work.output_queue)
                        logger.info(
                            "work {} {} done in {:.3f} s".format(
                                work.name.replace(" ", "_"),
                                key,
                                time.time() - t_start,
                            )
                        )
                        await trio.sleep(self.sleep_time)

            # there is output_queue
            elif w.output_queue is not None:

                async def func(work=w):
                    while True:
                        while (
                            not work.input_queue
                            or self.nb_working_worker >= self.nb_max_workers
                            or len(work.output_queue) >= self.nb_items_queue_max
                        ):
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                        key, obj = popitem(work.input_queue)
                        self.nb_working_worker += 1
                        self.nursery.start_soon(self.async_worker, work, key, obj)
                        await trio.sleep(self.sleep_time)

            # There is no output_queue
            else:

                async def func(work=w):
                    while True:
                        while (
                            not work.input_queue
                            or self.nb_working_worker >= self.nb_max_workers
                        ):
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                        key, obj = popitem(work.input_queue)
                        self.nb_working_worker += 1
                        self.nursery.start_soon(self.async_worker, work, key, obj)
                        await trio.sleep(self.sleep_time)

            self.async_funcs[w.name] = func

    def exec_one_shot_job(self):
        """
        Execute all "one shot functions".

        """
        for key, func in reversed(self.funcs.items()):
            logger.info(
                "Does one_shot_job, key func : {} with function {}".format(
                    key, func.func_or_cls
                )
            )
            func()

    async def async_worker(self, work, key, obj):
        """A worker is destined to be started with a "trio.start_soon".

        It does the work on an item (key,obj) given in parameter, and add the
        result on work.output_queue.

        :param work: A work from the topology
        :param key: The key of the dictionnary item to be process
        :param obj: The value of the dictionnary item to be process

        """
        t_start = time.time()
        log_memory_usage(
            "{:.2f} s. ".format(time.time() - self.t_start)
            + "Launch work "
            + work.name.replace(" ", "_")
            + " ({}). mem usage".format(key)
        )
        ret = await trio.run_sync_in_worker_thread(work.func_or_cls, obj)
        if work.output_queue is not None:
            push(key, obj, work.output_queue)
            work.output_queue[key] = ret
        logger.info(
            "work {} ({}) done in {:.3f} s".format(
                work.name.replace(" ", "_"), key, time.time() - t_start
            )
        )
        self.nb_working_worker -= 1
        return

    def store_async_works(self):
        """
        Picks up async works and stores them in `self.works`.
        """
        for w in self.topology.works:
            if w.kind is None or "one shot" not in w.kind:
                self.works.append(w)

    def has_to_stop(self):
        """Work has to stop flag. Check if all works has been done.

        Return True if there are no workers in working and if there is no items in
        all queues.

        """
        return (
            self._has_to_stop
            or (not any([len(queue) != 0 for queue in self.topology.queues]))
            and self.nb_working_worker == 0
        )
