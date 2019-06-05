"""
Multi executors async (:mod:`fluidimage.executors.multi_exec_async`)
====================================================================

.. autoclass:: MultiExecutorAsync
   :members:
   :private-members:

.. autoclass:: ExecutorAsyncForMulti
   :members:
   :private-members:

"""

import copy
import math
import os
import signal
import sys
from multiprocessing import Pipe, Process
from pathlib import Path
from time import time

from fluiddyn import time_as_str
from fluidimage.util import logger

from .base import ExecutorBase
from .exec_async_sequential import ExecutorAsyncSequential


class ExecutorAsyncForMulti(ExecutorAsyncSequential):
    """Slightly modified ExecutorAsync"""

    def __init__(
        self,
        topology,
        path_dir_result,
        log_path,
        sleep_time=0.01,
        logging_level="info",
        stop_if_error=False,
    ):
        if stop_if_error:
            raise NotImplementedError

        self._log_path = log_path
        super().__init__(
            topology,
            path_dir_result,
            nb_max_workers=1,
            nb_items_queue_max=3,
            sleep_time=sleep_time,
            logging_level=logging_level,
        )

    def _init_log_path(self):
        self.path_dir_exceptions = self._log_path.parent

    def _init_compute(self):
        self._init_compute_log()

    def _finalize_compute(self):
        self._reset_std_as_default()

        txt = self.topology.make_text_at_exit(time() - self.t_start)
        with open(self._log_file.name, "a") as file:
            file.write(txt)


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
        sleep_time=0.01,
        logging_level="info",
        stop_if_error=False,
    ):
        if stop_if_error:
            raise NotImplementedError

        super().__init__(
            topology,
            path_dir_result,
            nb_max_workers,
            nb_items_queue_max,
            logging_level=logging_level,
        )

        self.sleep_time = sleep_time
        self.nb_processes = self.nb_max_workers
        self.processes = []

        # to avoid a pylint warning
        self.log_paths = None

    def _init_log_path(self):
        name = "_".join(("log", time_as_str(), str(os.getpid())))
        path_dir_log = self.path_dir_exceptions = self.path_dir_result / name
        path_dir_log.mkdir(exist_ok=True)
        self._log_path = path_dir_log / (name + ".txt")

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
        self.log_paths = []

        if sys.platform != "win32":

            def handler_signals(signal_number, stack):
                del stack
                print(
                    f"signal {signal_number} received: set _has_to_stop to True "
                    f"({type(self).__name__})."
                )
                self._has_to_stop = True
                for process in self.processes:
                    os.kill(process.pid, signal_number)

            signal.signal(12, handler_signals)

        if hasattr(self.topology, "series"):
            self.start_multiprocess_series()
        else:
            self.start_mutiprocess_first_queue()

        self._finalize_compute()

    def start_mutiprocess_first_queue(self):
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

        for ind_process, keys_proc in enumerate(keys_for_processes):
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

            self.launch_process(topology_this_process, ind_process)

        self.wait_for_all_processes()

    def start_multiprocess_series(self):
        """Start the processes spitting the work with the series object"""
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
        for ind_process in range(self.nb_processes):
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

            self.launch_process(new_topology, ind_process)

        self.wait_for_all_processes()

    def launch_process(self, topology, ind_process):
        """Launch one process"""

        def init_and_compute(topology_this_process, log_path, child_conn):
            """Create an executor and start it in a process"""
            executor = ExecutorAsyncForMulti(
                topology_this_process,
                self.path_dir_result,
                sleep_time=self.sleep_time,
                log_path=log_path,
                logging_level=self.logging_level,
            )
            executor.t_start = self.t_start
            executor.compute()

            # send the results
            if hasattr(topology_this_process, "results"):
                results = topology_this_process.results
            else:
                results = None

            child_conn.send(results)

        log_path = Path(
            str(self._log_path).split(".txt")[0] + f"_multi{ind_process:03}.txt"
        )

        self.log_paths.append(log_path)

        parent_conn, child_conn = Pipe()

        process = Process(
            target=init_and_compute, args=(topology, log_path, child_conn)
        )
        process.connection = parent_conn
        process.daemon = True
        process.start()
        self.processes.append(process)

    def wait_for_all_processes(self):
        """logging + wait for all processes to finish"""
        logger.info(
            f"logging files: {[log_path.name for log_path in self.log_paths]}"
        )

        # wait until end of all processes

        self.topology.results = results_all = []
        for process in self.processes:
            results = process.connection.recv()

            if results is not None:
                results_all.extend(results)

        for process in self.processes:
            process.join()
