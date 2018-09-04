"""

A executor using async for IO and servers for CPU-bounded tasks.

Not implemented!

"""

import time

import trio

from fluidimage.util import logger, log_memory_usage

from .exec_async import ExecutorAsync


def popitem(input_queue):
    """Get an item from the input_queue."""
    key, obj = input_queue.popitem()
    return key, obj


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

    def __init__(
        self,
        topology,
        path_dir_result,
        nb_max_workers=None,
        nb_items_queue_max=None,
        sleep_time=0.1,
        logging_level="info",
    ):
        super().__init__(
            topology,
            path_dir_result,
            nb_max_workers,
            nb_items_queue_max,
            logging_level=logging_level,
        )

        nb_max_workers = self.nb_max_workers

        # create nb_max_workers servers

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

            self.nursery.start_soon(self.update_has_to_stop)
        return

    async def async_run_work_cpu(self, work, key, obj, client):
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
            parent_conn, child_conn = client.new_pipe()
            # send (work, key, obj, comm) to the server
            client.send((work, key, obj, child_conn))

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
            result = parent_conn.recv()

            return result

        self.nb_working_workers_cpu += 1
        ret = await trio.run_sync_in_worker_thread(run_process)
        if work.output_queue is not None:
            work.output_queue[key] = ret
        self.nb_working_workers_cpu -= 1

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
                    await trio.sleep(self.sleep_time)
                    available_worker = self.get_available_worker()

                key, obj = popitem(work.input_queue)
                self.nursery.start_soon(
                    self.async_run_work_cpu, work, key, obj, available_worker
                )
                await trio.sleep(self.sleep_time)

        return func

    def def_async_func_work_cpu_without_output_queue(self, work):
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
                    await trio.sleep(self.sleep_time)
                    available_worker = self.get_available_worker()

                key, obj = popitem(work.input_queue)
                self.nursery.start_soon(
                    self.async_run_work_cpu, work, key, obj
                )
                await trio.sleep(self.sleep_time)

        return func

    def get_available_worker(self):

        ...
        return False
