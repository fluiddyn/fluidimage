"""Executor async/await using servers (:mod:`fluidimage.executors.exec_async_servers`)
======================================================================================

A executor using async for IO and servers for CPU-bounded tasks.

.. autoclass:: ExecutorAsyncServers
   :members:
   :private-members:

"""

import os
import signal
from pathlib import Path

import numpy as np
import trio

from fluiddyn import time_as_str
from fluidimage.util import log_debug, logger

from .exec_async import ExecutorAsync
from .servers import launch_server

max_items_in_server = 4


class ExecutorAsyncServers(ExecutorAsync):
    """Executor async/await using servers.

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

    _type_server = "multiprocessing"

    def _init_log_path(self):
        name = "_".join(("log", time_as_str(), str(os.getpid())))
        path_dir_log = self.path_dir_exceptions = self.path_dir_result / name
        path_dir_log.mkdir(exist_ok=True)
        self._log_path = path_dir_log / (name + ".txt")

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
            sleep_time=sleep_time,
            logging_level=logging_level,
        )

        # create nb_max_workers servers
        self.workers = []

        for ind_worker in range(self.nb_max_workers):
            log_path = Path(
                str(self._log_path).split(".txt")[0]
                + f"_multi{ind_worker:03}.txt"
            )
            self.workers.append(
                launch_server(
                    topology,
                    log_path,
                    self._type_server,
                    sleep_time,
                    logging_level,
                )
            )

        def signal_handler(sig, frame):
            del sig, frame
            logger.info("Ctrl+C signal received...")

            for worker in self.workers:
                worker.terminate()

            self._has_to_stop = True
            self.nursery.cancel_scope.cancel()
            # we need to raise the exception
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, signal_handler)

    def compute(self):
        """Compute the whole topology.

        Begin by executing one shot jobs, then execute multiple shots jobs
        implemented as async functions.  Warning, one shot jobs must be ancestors
        of multiple shots jobs in the topology.

        """
        self._init_compute()
        for worker in self.workers:
            worker.send(("__t_start__", self.t_start))
        self.exec_one_shot_works()
        trio.run(self.start_async_works)
        self._finalize_compute()

    async def start_async_works(self):
        """Create a trio nursery and start all async functions.

        """
        async with trio.open_nursery() as self.nursery:
            for af in reversed(self.async_funcs.values()):
                self.nursery.start_soon(af)

            self.nursery.start_soon(self.update_has_to_stop)

        logger.info("terminate the servers")
        for worker in self.workers:
            worker.terminate()

    async def update_has_to_stop(self):
        """Work has to stop flag. Check if all works has been done.

        Return True if there are no workers in working and if there is no items in
        all queues.

        """

        while not self._has_to_stop:

            result = (
                (not any([bool(queue) for queue in self.topology.queues]))
                and all(worker.is_unoccupied for worker in self.workers)
                and self.nb_working_workers_io == 0
            )

            if result:
                self._has_to_stop = True
                log_debug(f"has_to_stop!")

            if self.logging_level == "debug":
                log_debug(f"self.topology.queues: {self.topology.queues}")
                log_debug(
                    "[worker.is_unoccupied for worker in self.workers]: "
                    f"{[worker.is_unoccupied for worker in self.workers]}"
                )
                log_debug(
                    "[worker.is_available for worker in self.workers]: "
                    f"{[worker.is_available for worker in self.workers]}"
                )
                log_debug(
                    "[worker.nb_items_to_process for worker in self.workers] "
                    f"{[worker.nb_items_to_process for worker in self.workers]}"
                )
                log_debug(
                    f"self.nb_working_workers_io: {self.nb_working_workers_io}"
                )

            await trio.sleep(self.sleep_time)

    async def async_run_work_cpu(self, work, worker):
        """Is destined to be started with a "trio.start_soon".

        Executes the work on an item (key, obj), and add the result on
        work.output_queue.

        Parameters
        ----------

        work :

          A work from the topology

        worker : .servers.WorkerMultiprocessing

          A client to communicate with the server worker.

        """
        try:
            key, obj = work.input_queue.pop_first_item()
        except KeyError:
            worker.is_available = True
            return

        if work.check_exception(key, obj):
            worker.is_available = True
            return

        def run_process():
            # create a communication channel
            parent_conn, child_conn = worker.new_pipe()
            # send (work, key, obj, comm) to the server
            worker.send_job((work.name, key, obj, child_conn))
            worker.is_available = True
            # wait for the end of the computation
            work_name_received, key_received, result = parent_conn.recv()
            assert work.name == work_name_received
            assert key == key_received
            return result

        ret = await trio.run_sync_in_worker_thread(run_process)
        if work.output_queue is not None:
            work.output_queue[key] = ret
        worker.well_done_thanks()

    def def_async_func_work_cpu(self, work):
        async def func(work=work):

            while True:
                while not work.input_queue or (
                    work.output_queue is not None
                    and len(work.output_queue) >= self.nb_items_queue_max
                ):
                    if self._has_to_stop:
                        return
                    await trio.sleep(self.sleep_time)

                available_worker = False
                while not available_worker:
                    if self._has_to_stop:
                        return
                    available_worker = self.get_available_worker()
                    await trio.sleep(self.sleep_time)

                self.nursery.start_soon(
                    self.async_run_work_cpu, work, available_worker
                )
                await trio.sleep(self.sleep_time)

        return func

    def get_available_worker(self):
        """Get a worker available to receive a new job"""
        available_workers = [
            worker
            for worker in self.workers
            if worker.is_available
            and worker.nb_items_to_process < max_items_in_server
        ]

        if not available_workers:
            return False

        nb_items_to_process_workers = [
            worker.nb_items_to_process for worker in available_workers
        ]

        index = np.argmin(nb_items_to_process_workers)
        worker = available_workers[index]
        worker.is_available = False
        return worker


class ExecutorAsyncServersThreading(ExecutorAsyncServers):
    """Just used to get a better coverage"""

    _type_server = "threading"
