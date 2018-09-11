"""

A executor using async for IO and multiprocessing for CPU bounded tasks.

.. autoclass:: ExecutorAsyncMultiproc
   :members:
   :private-members:

"""

import time
from multiprocessing import Process, Pipe

import trio

from fluidimage.util import logger, log_memory_usage, cstring

from .exec_async import ExecutorAsync


class ExecutorAsyncMultiproc(ExecutorAsync):
    """Async executor using multiprocessing to launch CPU-bounded tasks"""

    async def async_run_work_cpu(self, work):
        """Is destined to be started with a "trio.start_soon".

        Executes the work on an item (key, obj), and add the result on
        work.output_queue.

        Parameters
        ----------

        work :

          A work from the topology

        """
        self.nb_working_workers_cpu += 1

        try:
            key, obj = work.input_queue.pop_first_item()
        except KeyError:
            self.nb_working_workers_cpu -= 1

        if work.check_exception(key, obj):
            self.nb_working_workers_cpu -= 1
            return

        t_start = time.time()
        log_memory_usage(
            f"{time.time() - self.t_start:.2f} s. Launch work "
            + work.name_no_space
            + f" ({key}). mem usage"
        )

        parent_conn, child_conn = Pipe()

        def exec_work_and_comm(func, obj, child_conn):
            try:
                result = func(obj)
            except Exception as error:
                result = error

            child_conn.send(result)
            child_conn.close()

        def run_process():
            process = Process(
                target=exec_work_and_comm,
                args=(work.func_or_cls, obj, child_conn),
            )
            process.daemon = True
            process.start()
            result = parent_conn.recv()

            process.join(10 * self.sleep_time)
            if process.exitcode != 0:
                logger.info(f"process.exitcode: {process.exitcode}")
                process.terminate()

            return result

        ret = await trio.run_sync_in_worker_thread(run_process)

        if isinstance(ret, Exception):
            logger.error(
                cstring(
                    "error during work " f"{work.name_no_space} ({key})",
                    color="FAIL",
                )
            )
            if self.stop_if_error:
                raise (ret)
        else:
            logger.info(
                f"work {work.name_no_space} ({key}) "
                f"done in {time.time() - t_start:.3f} s"
            )

        if work.output_queue is not None:
            work.output_queue[key] = ret
        self.nb_working_workers_cpu -= 1
