"""Executor async/await
=======================

This executer uses await/async with trio library to put topology tasks in
concurrency.  Depending on the topology, parameters (worker_limit, queues_limit,
sleep_time) can be set to improve the performance.  Two modes of computation are
available:

- The single executor mode.

  A single executor (in one process) is created.  If CPU bounded tasks are limited
  by the Python GIL, the threads won't use at the same time the CPU.


Meaning that the work will be
  done in a single thread, except if the topology computed has C code in it. In
  this case, the GIL is bypassed and computation can use many CPU.

- The multi-executor mode.

  Many executors are created, each executor works in a process from
  multiprocessing with a part of the work to do. The work split is done in the
  class "ExecutorAwaitMultiproc". Usefull for full python written topology. See
  "ExecutorAwaitMultiproc" for more information.

.. autoclass:: ExecutorAwait
   :members:
   :private-members:

.. autoclass:: ExecutorAwaitMultiprocs
   :members:
   :private-members:

"""

import time
import math
import collections
import copy
import trio
from fluiddyn import time_as_str
from fluidimage.util.util import logger, get_memory_usage, log_memory_usage
from fluidimage.experimental.executors.executor_base import ExecutorBase
from multiprocessing import Process
from itertools import islice


class ExecutorAwait(ExecutorBase):
    """Executor async/await.

    Work in a single thread, except if the computed topology has C code.

    Parameters
    ----------

    worker_limit : None, int

      Limits the numbers of workers working in the same time.

    queue_limit : None, int

      Limits the numbers of items that can be in a output_queue.

    sleep_time : None, float

      Defines the waiting time (from trio.sleep) of a function. Functions await
      "trio.sleep" when they have done a work on an item, and when there is
      nothing in there input_queue.

    """

    def __init__(
        self,
        topology,
        worker_limit=6,
        queues_limit=4,
        sleep_time=0.1,
        new_dict=False,
    ):
        super().__init__(topology)
        # If multi_executing with first queue split, change topology first queue
        if new_dict is not False:
            topology.queues[0].queue = new_dict

        # Oject variables
        self.t_start = time.time()
        self.nb_working_worker = 0

        # Executor parameters
        self.worker_limit = worker_limit
        self.sleep_time = sleep_time
        self.queues_limit = queues_limit

        # fonction containers
        self.works = []
        self.async_funcs = collections.OrderedDict()
        self.funcs = collections.OrderedDict()

        # Functions definition
        self.get_async_works()
        self.define_function()
        # Queue0

    def compute(self):
        """
        Compute the whole topology. Begin by executing one shot jobs,
        then execute multiple shots jobs implemented as async functions.
        Warning, one shot jobs must be ancestors of multiple shots jobs in the topology
        :return:
        """

        self.t_start = time.time()
        self.do_one_shot_job()
        trio.run(self.start_async_works)

    async def start_async_works(self):
        """
        Create a trio nursery and start soon all async functions (multiple shots functions)
        :return:
        """
        async with trio.open_nursery() as self.nursery:
            for key, af in reversed(self.async_funcs.items()):
                self.nursery.start_soon(af)
        return

    def define_function(self):
        """
        Define sync functions (one shot functions) and async functions (multiple shot functions),
        and store them in "self.async_funcs". The behavior of the executor is mostly defined here.
        To sum up : Each "multiple shot" works from the topology, waits
        for an items to be avaible in there intput_queue. Then
        :return:
        """
        for w in reversed(self.topology.works):
            # One shot functions
            if w.kind is not None and "one shot" in w.kind:

                def func(work=w):
                    work.func_or_cls(work.input_queue, work.output_queue)

                self.funcs[w.name] = func
                continue

            # global functions
            elif w.kind is not None and "global" in w.kind:

                async def func(work=w):
                    item_number = 1
                    while True:
                        while len(work.output_queue.queue) > self.queues_limit:
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
                            "{:.2f} s. ".format(
                                time.time() - self.topology.t_start
                            )
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
                            not work.input_queue.queue
                            or self.nb_working_worker >= self.worker_limit
                            or len(work.output_queue.queue) > self.queues_limit
                        ):
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                        t_start = time.time()
                        work.input_queue.queue["in_working"] = True
                        (key, obj) = await trio.run_sync_in_worker_thread(
                            self.pull, work.input_queue.queue
                        )
                        log_memory_usage(
                            "{:.2f} s. ".format(
                                time.time() - self.topology.t_start
                            )
                            + "Launch work "
                            + work.name.replace(" ", "_")
                            + " ({}). mem usage".format(key)
                        )
                        ret = await trio.run_sync_in_worker_thread(
                            work.func_or_cls, obj
                        )
                        self.push(key, ret, work.output_queue.queue)
                        # self.nursery.start_soon(self.worker, work, key, obj)
                        del work.input_queue.queue["in_working"]
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
                            not work.input_queue.queue
                            or self.nb_working_worker >= self.worker_limit
                            or len(work.output_queue.queue) > self.queues_limit
                        ):
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                        work.input_queue.queue["in_working"] = True
                        (key, obj) = await trio.run_sync_in_worker_thread(
                            self.pull, work.input_queue.queue
                        )
                        self.nb_working_worker += 1
                        self.nursery.start_soon(self.worker, work, key, obj)
                        del work.input_queue.queue["in_working"]
                        await trio.sleep(self.sleep_time)

            # There is no output_queue
            else:

                async def func(work=w):
                    while True:
                        while (
                            not work.input_queue.queue
                            or self.nb_working_worker >= self.worker_limit
                        ):
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                        start = time.time()
                        work.input_queue.queue["in_working"] = True
                        (key, obj) = await trio.run_sync_in_worker_thread(
                            self.pull, work.input_queue.queue
                        )
                        self.nb_working_worker += 1
                        self.nursery.start_soon(self.worker, work, key, obj)
                        del work.input_queue.queue["in_working"]
                        await trio.sleep(self.sleep_time)

            self.async_funcs[w.name] = func

    def do_one_shot_job(self):
        """
        Execute all "one shot functions".
        :return:
        """
        for key, func in reversed(self.funcs.items()):
            logger.info(
                "Does one_shot_job, key func : {} with function {}".format(
                    key, func
                )
            )
            func()

    async def worker(self, work, key, obj):
        """
        A worker is destined to be started with a "trio.start_soon".
        It does the work on an item (key,obj) given in parameter,  and add the result on work.output_queue.
        :param work: A work from the topology
        :param key: The key of the dictionnary item to be process
        :param obj: The value of the dictionnary item to be process
        :return:
        """
        t_start = time.time()
        log_memory_usage(
            "{:.2f} s. ".format(time.time() - self.topology.t_start)
            + "Launch work "
            + work.name.replace(" ", "_")
            + " ({}). mem usage".format(key)
        )
        ret = await trio.run_sync_in_worker_thread(work.func_or_cls, obj)
        if work.output_queue is not None:
            self.push(key, obj, work.output_queue.queue)
            work.output_queue.queue[key] = ret
        logger.info(
            "work {} ({}) done in {:.3f} s".format(
                work.name.replace(" ", "_"), key, time.time() - t_start
            )
        )
        self.nb_working_worker -= 1
        return

    async def workerIO(self, work, key, obj):
        """
        In case IO tasks need to have a specific worker. At this moment, this worker is unused.
        :param work: A work from the topology
        :param key: The key of the dictionnary item to be process
        :param obj: The value of the dictionnary item to be process
        :return:
        """
        t_start = time.time()
        log_memory_usage(
            "{:.2f} s. ".format(time.time() - self.topology.t_start)
            + "Launch work "
            + work.name.replace(" ", "_")
            + " ({}). mem usage".format(key)
        )
        ret = await trio.run_sync_in_worker_thread(work.func_or_cls, obj)
        if work.output_queue is not None:
            self.push(key, obj, work.output_queue.queue)
            work.output_queue.queue[key] = ret
        logger.info(
            "work {} {} done in {:.3f} s".format(
                work.name.replace(" ", "_"), key, time.time() - t_start
            )
        )
        self.nb_working_worker -= 1
        return

    def get_async_works(self):
        """
        Picks up async works and stores them in self.works
        :return:
        """
        for w in self.topology.works:
            if w.kind is None or "one shot" not in w.kind:
                self.works.append(w)

    def has_to_stop(self):
        """
        Work has to stop flag. Check if all works has been done.
        :return: True if there are no workers in working and if there is no items in all queues.
        :type boolean
        """
        return (
            not any([len(q.queue) != 0 for q in self.topology.queues])
        ) and self.nb_working_worker == 0

    def pull(self, input_queue):
        """
        Get an item from the input_queue. If et gets the flag item "in_working",
        get another item et put again the item "in working"
        :param input_queue:
        :return:
        """
        key, obj = input_queue.popitem()
        if key is "in_working":
            key, obj = input_queue.popitem()
            input_queue["in_working"] = True
        return key, obj

    def push(self, key, obj, output_queue):
        """
        Add an item (key, obj) in the output_queue
        :param key: A dictionnary key
        :param obj: A dictionnary value
        :param output_queue: a dictionnary
        :return:
        """
        output_queue[key] = obj
        return

    def print_queues(self):
        """
        For all queues in "self.topology", print lenght and name
        :return:
        """
        for q in self.topology.queues:
            print("{} : {} ".format(len(q.queue), q.name))
        print("\n")


class ExecutorAwaitMultiprocs(ExecutorBase):
    """ Manage the multi-executor mode
     This class is not the one whose really compute the topology. It is used to manage the Multi_executer
     mode(activated with multi_executor=True). With the multi_executor mode, the topology is split and each slice is computed with an Executer_await

    Parameters
    ----------
    Multi_executor: None, boolean

      Activate the multi-executors mode is set at True

    Worker_limit : None, int

      Limits the numbers of workers working in the same time.

    Queue_limit : None, int

      Limits the numbers of items that can be in a output_queue.

    Sleep_time : None, float

      defines the waiting time (from trio.sleep) of a function. Async functions await "trio.sleep(sleep_time)"
      when they have done a work on an item, and when there is nothing in there input_queue.

    """

    def __init__(
        self,
        topology,
        multi_executor=False,
        worker_limit=None,
        queues_limit=4,
        sleep_time=0.1,
    ):
        super().__init__(topology)
        self.multi_executor = multi_executor
        if worker_limit is None:
            self.worker_limit = self.nb_max_workers
        else:
            self.worker_limit = worker_limit
        self.queues_limit = queues_limit
        self.sleep_time = sleep_time

    def compute(self):
        """Compute the topology.
        If self.multi_executor is True, split self.topology work in "self.nb_max_worker" slices.
        There is two ways do do this :
        - If first self.topology has "series" attribute (from seriesOfArray), it creates "self.nb_max_workers"
        topologies and changes "ind_start" and "ind_stop" of topology.series. The split considers series.ind_step.
        - Else, if the first work of the topology has an unique output_queue, it splits that queue in "self.nb_max_worker"
        slices and create as many topologies. On these last, the first work will be removed and the first queue will be filled
        with a partition of the first queue
        Then create as many Executer_await as topologies, give each topology to each executors, and call each Executor_await.compute
        in a process from multiprocessing.

        Else, self.multi_executor is False, simply create an Executor_await, give it the topology and call Executor_await.compute
        :return:
        """
        logger.info(
            "[92m{}: start compute. mem usage: {} Mb[0m".format(
                time_as_str(2), get_memory_usage()
            )
        )
        log_memory_usage(
            "{}:".format(time_as_str(2)) + " start compute. " + "mem usage"
        )
        self.t_start = time.time()
        if self.multi_executor is True:
            self.multi_executor_compute(nb_process=self.nb_max_workers)
        else:
            executor_await = ExecutorAwait(
                self.topology,
                worker_limit=self.worker_limit,
                queues_limit=self.queues_limit,
                sleep_time=self.sleep_time,
            )
            executor_await.do_one_shot_job()
            trio.run(executor_await.start_async_works)
        log_memory_usage(
            "{}:".format(time_as_str(2)) + " end compute. " + "mem usage"
        )
        print("Work all done in {:.5f} s".format(time.time() - self.t_start))

    def multi_executor_compute(self, nb_process):
        # topology doesn't have series
        if not hasattr(self.topology, "series"):
            self.start_mutiprocess_first_queue(nb_process)
        # topology heas series
        else:
            self.start_multiprocess_series(nb_process)

    def start_mutiprocess_first_queue(self, nb_process):
        # get the first queue
        for w in self.topology.works:
            if w.input_queue == None:
                first_queue = w.output_queue
                work_first_queue = w
                break
        # fill the first queue
        if isinstance(work_first_queue.output_queue, tuple):
            raise NotImplementedError("First work have two or more output_queues")
        work_first_queue.func_or_cls(input_queue=None, output_queue=first_queue)
        # split it
        dict_list = []
        for item in self.partition_dict(first_queue.queue, nb_process):
            dict_list.append(item)
        i = 0
        nb_dict = len(dict_list)
        for item in first_queue.queue.items():
            if not self.is_in_dict_list(item, dict_list):
                dict_list[i % nb_dict][item[0]] = item[1]
                i += 1
        # change topology
        new_topology = copy.copy(self.topology)
        del new_topology.works[0]
        process = []
        for i in range(nb_process):
            print("process ", i)
            # Create an executor and start it in a process
            executor = ExecutorAwait(
                new_topology, worker_limit=1, dict_list=dict_list[i]
            )
            p = Process(target=executor.compute)
            process.append(p)
            p.start()
        for p in process:
            p.join()

    def start_multiprocess_series(self, nb_process):

        process = []
        ind_stop_limit = self.topology.series.ind_stop
        # Defining split values
        ind_start = self.topology.series.ind_start
        nb_image_computed = math.floor(
            (self.topology.series.ind_stop - self.topology.series.ind_start)
            / self.topology.series.ind_step
        )
        remainder = nb_image_computed % nb_process
        step_process = math.floor(nb_image_computed / nb_process)
        # change topology
        for i in range(nb_process):
            new_topology = copy.copy(self.topology)
            new_topology.series.ind_start = ind_start
            add_rest = 0
            # To add forgotten images
            if remainder > 0:
                add_rest = 1
                remainder -= 1
            # defining ind_stop
            ind_stop = (
                self.topology.series.ind_start
                + step_process * self.topology.series.ind_step
                + add_rest * self.topology.series.ind_step
            )
            # To make sure images exist
            if ind_stop > ind_stop_limit:
                new_topology.series.ind_stop = ind_stop_limit
            else:
                new_topology.series.ind_stop = ind_stop
            ind_start = self.topology.series.ind_stop
            # Create an executor and launch it in a process
            executor = ExecutorAwait(new_topology, worker_limit=1)
            p = Process(target=executor.compute)
            process.append(p)
            p.start()
        # wait until end of all processes
        for p in process:
            p.join()

    @staticmethod
    def partition_dict(dict, num):
        slice = int(len(dict) / num)
        it = iter(dict)
        for i in range(0, len(dict), slice):
            yield {k: dict[k] for k in islice(it, slice)}

    def is_in_dict_list(self, item, dict_list):
        for dict in dict_list:
            if item in dict.items():
                return True
        return False
