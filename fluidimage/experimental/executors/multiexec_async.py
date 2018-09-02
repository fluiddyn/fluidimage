"""
Multi executors async (:mod:`fluidimage.experimental.executors.multiexec_async`)
================================================================================

.. autoclass:: MultiExecutorAsync
   :members:
   :private-members:

.. autoclass:: ExecutorAsyncForMulti
   :members:
   :private-members:

"""

from multiprocessing import Process
import copy
import math

import trio

from .base import ExecutorBase
from .exec_async import ExecutorAsync


class ExecutorAsyncForMulti(ExecutorAsync):
    """Slightly modified ExecutorAsync"""
    def compute(self):
        self.exec_one_shot_job()
        trio.run(self.start_async_works)

class MultiExecutorAsync(ExecutorBase):
    """Manage the multi-executor mode

     This class is not the one whose really compute the topology. The topology is
     split and each slice is computed with an ExecutorAsync

    Parameters
    ----------

    nb_max_workers : None, int

      Limits the numbers of workers working in the same time.

    nb_items_queue_max : None, int

      Limits the numbers of items that can be in a output_queue.

    sleep_time : None, float

      defines the waiting time (from trio.sleep) of a function. Async functions
      await "trio.sleep(sleep_time)" when they have done a work on an item, and
      when there is nothing in there input_queue.

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

        self.sleep_time = sleep_time
        self.nb_processes = self.nb_max_workers

    def compute(self):
        """Compute the topology.

        There are two ways to split self.topology work:

        - If first self.topology has "series" attribute (from seriesOfArray), it
          creates "self.nb_max_workers" topologies and changes "ind_start" and
          "ind_stop" of topology.series. The split considers series.ind_step.

        - Else, if the first work of the topology has an unique output_queue, it
          splits that queue in "self.nb_max_worker" slices and create as many
          topologies. On these last, the first work will be removed and the first
          queue will be filled with a partition of the first queue Then create as
          many Executer_await as topologies, give each topology to each executors,
          and call each Executor_await.compute in a process from multiprocessing.

        """
        self._init_compute()

        # topology doesn't have series
        if not hasattr(self.topology, "series"):
            self.start_mutiprocess_first_queue()
        # topology heas series
        else:
            self.start_multiprocess_series()
        self._finalize_compute()

    def start_mutiprocess_first_queue(self):
        first_work = self.topology.works[0]
        if first_work.input_queue is not None:
            raise NotImplementedError
        if isinstance(first_work.output_queue, tuple):
            raise NotImplementedError

        # fill the first queue
        first_work.func_or_cls(
            input_queue=None, output_queue=first_work.output_queue
        )

        first_queue = copy.copy(first_work.output_queue)

        # split the first queue
        keys = list(first_queue.keys())

        nb_keys_per_process = max(1, int(len(keys) / self.nb_processes))

        keys_for_processes = [
            keys[iproc : iproc + nb_keys_per_process]
            for iproc in range(self.nb_processes)
        ]

        # change topology
        self.topology.first_queue = self.topology.works[0].output_queue
        topology = copy.copy(self.topology)
        topology.first_queue.clear()
        del topology.works[0]
        old_queue = topology.first_queue

        processes = []
        for iproc, keys_proc in enumerate(keys_for_processes):
            topology_this_process = copy.copy(self.topology)
            new_queue = copy.copy(topology.first_queue)
            topology_this_process.first_queue = new_queue

            for iq, queue in enumerate(topology_this_process.queues):
                if queue is old_queue:
                    topology_this_process.queues[iq] = new_queue

            for work in topology_this_process.works:
                if work.output_queue is old_queue:
                    work.output_queue = new_queue

                if work.input_queue is old_queue:
                    work.input_queue = new_queue

                if isinstance(work.input_queue, (tuple, list)):
                    work.input_queue = list(work.input_queue)
                    for iq, queue in enumerate(work.input_queue):
                        if queue is old_queue:
                            work.input_queue[iq] = new_queue
                if isinstance(work.output_queue, (tuple, list)):
                    work.output_queue = list(work.output_queue)
                    for iq, queue in enumerate(work.output_queue):
                        if queue is old_queue:
                            work.output_queue[iq] = new_queue

            for key in keys_proc:
                new_queue[key] = first_queue[key]

            old_queue = new_queue

            # Create an executor and start it in a process
            executor = ExecutorAsyncForMulti(
                topology_this_process,
                self.path_dir_result,
                nb_max_workers=1,
                nb_items_queue_max=self.nb_items_queue_max,
                sleep_time=self.sleep_time,
            )
            executor.t_start = self.t_start
            p = Process(target=executor.compute)
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

    def start_multiprocess_series(self):

        process = []
        ind_stop_limit = self.topology.series.ind_stop
        # Defining split values
        ind_start = self.topology.series.ind_start
        nb_image_computed = math.floor(
            (self.topology.series.ind_stop - self.topology.series.ind_start)
            / self.topology.series.ind_step
        )
        remainder = nb_image_computed % self.nb_processes
        step_process = math.floor(nb_image_computed / self.nb_processes)
        # change topology
        for i in range(self.nb_processes):
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
            executor = ExecutorAsyncForMulti(
                new_topology,
                self.path_dir_result,
                nb_max_workers=1,
                nb_items_queue_max=self.nb_items_queue_max,
                sleep_time=self.sleep_time,
            )
            executor.t_start = self.t_start
            p = Process(target=executor.compute)
            process.append(p)
            p.start()
        # wait until end of all processes
        for p in process:
            p.join()
