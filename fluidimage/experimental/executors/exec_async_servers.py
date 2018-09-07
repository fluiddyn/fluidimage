"""

A executor using async for IO and servers for CPU-bounded tasks.

Not implemented!

"""

import time
import signal

import trio
import numpy as np

from fluidimage.util import logger, log_memory_usage

# from fluidimage.topologies.nb_cpu_cores import nb_cores

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

    def __init__(
        self,
        topology,
        path_dir_result,
        nb_max_workers=None,
        nb_items_queue_max=None,
        sleep_time=0.01,
        logging_level="info",
    ):

        # if nb_max_workers is None:
        #     nb_max_workers = nb_cores

        super().__init__(
            topology,
            path_dir_result,
            nb_max_workers,
            nb_items_queue_max,
            sleep_time=sleep_time,
            logging_level=logging_level,
        )

        # create nb_max_workers servers
        self.workers = [
            launch_server(topology, self._type_server, sleep_time)
            for _ in range(self.nb_max_workers)
        ]

        def signal_handler(sig, frame):
            logger.info("Ctrl+C signal received...")

            for worker in self.workers:
                worker.terminate()

            self._has_to_stop = True
            self.nursery.cancel_scope.cancel()
            # we need to raise the exception
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, signal_handler)

    async def start_async_works(self):
        """Create a trio nursery and start all async functions.

        """
        async with trio.open_nursery() as self.nursery:
            for af in reversed(self.async_funcs.values()):
                self.nursery.start_soon(af)

            self.nursery.start_soon(self.update_has_to_stop)

        print("terminate the servers")
        for worker in self.workers:
            worker.terminate()

    async def update_has_to_stop(self):
        """Work has to stop flag. Check if all works has been done.

        Return True if there are no workers in working and if there is no items in
        all queues.

        """

        while not self._has_to_stop:

            result = (
                (not any([len(queue) != 0 for queue in self.topology.queues]))
                and all(worker.is_unoccupied for worker in self.workers)
                and self.nb_working_workers_io == 0
            )

            if result:
                self._has_to_stop = True
                logger.debug(f"has_to_stop!")

            if self.logging_level == "debug":
                logger.debug(f"self.topology.queues: {self.topology.queues}")
                logger.debug(
                    "[worker.is_unoccupied for worker in self.workers]: "
                    f"{[worker.is_unoccupied for worker in self.workers]}"
                )
                logger.debug(
                    "[worker.nb_items_to_process for worker in self.workers] "
                    f"{[worker.nb_items_to_process for worker in self.workers]}"
                )
                logger.debug(
                    f"self.nb_working_workers_io: {self.nb_working_workers_io}"
                )

            await trio.sleep(self.sleep_time)

    async def async_run_work_cpu(self, work, key, obj, worker):
        """Is destined to be started with a "trio.start_soon".

        Executes the work on an item (key, obj), and add the result on
        work.output_queue.

        Parameters
        ----------

        work :

          A work from the topology

        key : hashable

          The key of the dictionnary item to be process

        obj : object

          The value of the dictionnary item to be process

        """
        t_start = time.time()

        def run_process():

            # create a communication channel
            parent_conn, child_conn = worker.new_pipe()
            # send (work, key, obj, comm) to the server
            worker.send_job((work.name, key, obj, child_conn))
            worker.is_available = True

            # wait for the signal of the start of the computation
            signal = parent_conn.recv()

            # check the value of signal
            if not signal == "computation started":
                raise Exception

            # log launch work
            log_memory_usage(
                "{:.2f} s. ".format(time.time() - self.t_start)
                + "Launch work "
                + work.name.replace(" ", "_")
                + " ({}). mem usage".format(key)
            )

            # wait for the end of the computation
            work_name_received, key_received, result = parent_conn.recv()
            assert work.name == work_name_received
            assert key == key_received

            worker.well_done_thanks()

            return result

        ret = await trio.run_sync_in_worker_thread(run_process)
        if work.output_queue is not None:
            work.output_queue[key] = ret

        logger.info(
            "work {} ({}) done in {:.3f} s".format(
                work.name.replace(" ", "_"), key, time.time() - t_start
            )
        )

    def def_async_func_work_cpu_with_output_queue(self, work):
        async def func(work=work):
            while True:
                while (
                    not work.input_queue
                    or len(work.output_queue) >= self.nb_items_queue_max
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

                key, obj = work.input_queue.popitem()
                self.nursery.start_soon(
                    self.async_run_work_cpu, work, key, obj, available_worker
                )
                await trio.sleep(self.sleep_time)

        return func

    def def_async_func_work_cpu_without_output_queue(self, work):
        async def func(work=work):
            while True:
                while not work.input_queue:
                    if self._has_to_stop:
                        return
                    await trio.sleep(self.sleep_time)

                available_worker = False
                while not available_worker:
                    if self._has_to_stop:
                        return
                    available_worker = self.get_available_worker()
                    await trio.sleep(self.sleep_time)

                key, obj = work.input_queue.popitem()
                self.nursery.start_soon(
                    self.async_run_work_cpu, work, key, obj, available_worker
                )
                await trio.sleep(self.sleep_time)

        return func

    def get_available_worker(self):
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
