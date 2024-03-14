"""
Multi executors async
=====================

.. autoclass:: MultiExecutorAsync
   :members:
   :private-members:

.. autoclass:: ExecutorAsyncForMulti
   :members:
   :private-members:

"""

import copy
import math
from multiprocessing import Pipe, Process

from .base import MultiExecutorBase
from .exec_async_seq_for_multi import ExecutorAsyncSeqForMulti


class ExecutorAsyncForMulti(ExecutorAsyncSeqForMulti):
    """Slightly modified ExecutorAsync"""


class MultiExecutorAsync(MultiExecutorBase):
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

      Defines the waiting time (from trio.sleep) of a function. Async functions
      await `trio.sleep(sleep_time)` when they have done a work on an item, and
      when there is nothing in their input_queue.

    """

    def _start_processes(self):
        """
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
        if hasattr(self.topology, "series"):
            self._start_multiprocess_series()
        else:
            self._start_multiprocess_first_queue()

    def _start_multiprocess_first_queue(self):
        """Start the processes spitting the work with the first queue"""
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
        self.num_expected_results = len(keys)

        nb_keys_per_process = max(1, int(len(keys) / self.nb_processes))

        keys_for_processes = []
        for iproc in range(self.nb_processes):
            istart = iproc * nb_keys_per_process
            keys_for_processes.append(keys[istart : istart + nb_keys_per_process])

        # change topology
        self.topology.first_queue = self.topology.works[0].output_queue
        topology = copy.copy(self.topology)
        topology.first_queue.clear()
        del topology.works[0]
        old_queue = topology.first_queue

        for idx_process, keys_proc in enumerate(keys_for_processes):
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

            self.launch_process(topology_this_process, idx_process)

    def _start_multiprocess_series(self):
        """Start the processes spitting the work with the series object"""
        ind_stop_limit = self.topology.series.ind_stop
        # Defining split values
        ind_start = self.topology.series.ind_start
        nb_image_computed = math.floor(
            (self.topology.series.ind_stop - self.topology.series.ind_start)
            / self.topology.series.ind_step
        )

        self.num_expected_results = nb_image_computed

        remainder = nb_image_computed % self.nb_processes
        step_process = math.floor(nb_image_computed / self.nb_processes)
        # change topology
        for idx_process in range(self.nb_processes):
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

            if range(
                ind_start,
                new_topology.series.ind_stop,
                new_topology.series.ind_step,
            ):
                self.launch_process(new_topology, idx_process)

            ind_start = self.topology.series.ind_stop

    def init_and_compute(
        self, topology_this_process, log_path, child_conn, idx_process
    ):
        """Create an executor and start it in a process"""
        executor = ExecutorAsyncForMulti(
            topology_this_process,
            self.path_dir_result,
            sleep_time=self.sleep_time,
            path_log=log_path,
            logging_level=self.logging_level,
            t_start=self.t_start,
            index_process=idx_process,
        )
        executor.compute()

        # send the results
        if hasattr(topology_this_process, "results"):
            results = topology_this_process.results
        else:
            results = None

        child_conn.send(results)

    def launch_process(self, topology, idx_process):
        """Launch one process"""

        log_path = self._log_path.parent / f"process_{idx_process:03d}.txt"
        self.log_paths.append(log_path)

        parent_conn, child_conn = Pipe()

        process = Process(
            target=self.init_and_compute,
            args=(topology, log_path, child_conn, idx_process),
        )
        process.connection = parent_conn
        process.daemon = True
        process.start()
        self.processes.append(process)

    def _poll_return_code(self, process):
        return process.exitcode

    def _join_processes(self):
        """Join the processes"""
        for process in self.processes:
            process.join()


Executor = MultiExecutorAsync
