"""

A executor using async for IO and multiprocessing for CPU bounded tasks.

.. autoclass:: ExecutorAsyncMultiproc
   :members:
   :private-members:

"""

import time
from multiprocessing import Process, Pipe

import trio

from fluidimage.util import logger, log_memory_usage

from .exec_async import ExecutorAsync


class ExecutorAsyncMultiproc(ExecutorAsync):
    """Async executor using multiprocessing to launch CPU-bounded tasks"""

    async def async_run_work_cpu(self, work, key, obj):
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
        log_memory_usage(
            "{:.2f} s. ".format(time.time() - self.t_start)
            + "Launch work "
            + work.name.replace(" ", "_")
            + " ({}). mem usage".format(key)
        )

        parent_conn, child_conn = Pipe()

        def exec_work_and_comm(func, obj, child_conn):
            result = func(obj)
            child_conn.send(result)
            child_conn.close()

        def run_process():
            process = Process(
                target=exec_work_and_comm,
                args=(work.func_or_cls, obj, child_conn),
            )
            process.start()
            result = parent_conn.recv()
            process.join(10 * self.sleep_time)
            if process.exitcode != 0:
                logger.info(f"process.exitcode: {process.exitcode}")
                process.terminate()

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
