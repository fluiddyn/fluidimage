"""Executor async/await + multiprocessing (:mod:`fluidimage.executors.exec_async_multiproc`)
============================================================================================

A executor using async for IO and multiprocessing for CPU bounded tasks.

.. autoclass:: ExecutorAsyncMultiproc
   :members:
   :private-members:

"""

import time
from multiprocessing import Event, Pipe, Process

import trio

from fluidimage.util import cstring, log_debug, log_memory_usage, logger

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
            return

        if work.check_exception(key, obj):
            self.nb_working_workers_cpu -= 1
            return

        t_start = time.time()
        log_memory_usage(
            f"{time.time() - self.t_start:.2f} s. Launch work "
            + work.name_no_space
            + f" ({key}). mem usage"
        )

        def exec_work_and_comm(func, obj, child_conn, event):
            # log_debug(f"process ({key}) started")
            event.set()
            # pylint: disable=W0703
            try:
                result = func(obj)
            except Exception as error:
                result = error

            # log_debug(f"in process, send result ({key}): {result}")
            child_conn.send(result)

        parent_conn, child_conn = Pipe()
        event = Event()

        def run_process():

            # we do this complicate thing because there may be a strange bug

            def start_process_and_check(index_attempt):
                process = Process(
                    target=exec_work_and_comm,
                    args=(work.func_or_cls, obj, child_conn, event),
                )
                process.daemon = True
                process.start()
                # check whether the process has really started (possible bug!)
                if not event.wait(1):
                    log_debug(
                        f"problem: process {work.name_no_space} ({key}) "
                        f"has not really started... (attempt {index_attempt})"
                    )
                    process.terminate()
                    return False
                return process

            really_started = False
            for index_attempt in range(10):
                process = start_process_and_check(index_attempt)
                if process:
                    really_started = True
                    break

            if not really_started:
                raise Exception(
                    f"A process {work.name_no_space} ({key}) "
                    "has not started after 10 attempts"
                )

            # todo: use parent_conn.poll to implement a timeout

            # log_debug(f"waiting for result ({key})")
            result = parent_conn.recv()
            # log_debug(f"result ({key}) received")

            process.join(10 * self.sleep_time)
            if process.exitcode != 0:
                logger.info(f"process.exitcode: {process.exitcode}")
                process.terminate()

            return result

        ret = await trio.run_sync_in_worker_thread(run_process)

        if isinstance(ret, Exception):
            self.log_exception(ret, work.name_no_space, key)
            if self.stop_if_error:
                raise ret
        else:
            logger.info(
                f"work {work.name_no_space} ({key}) "
                f"done in {time.time() - t_start:.3f} s"
            )

        if work.output_queue is not None:
            work.output_queue[key] = ret
        self.nb_working_workers_cpu -= 1
