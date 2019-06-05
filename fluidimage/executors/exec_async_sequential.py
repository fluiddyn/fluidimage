"""Executor async/await sequential (:mod:`fluidimage.executors.exec_async_sequential`)
======================================================================================

A executor using async for IO but launching CPU-bounded tasks sequentially.

.. autoclass:: ExecutorAsyncSequential
   :members:
   :private-members:

"""

import time

from fluidimage.util import cstring, log_memory_usage, logger

from .exec_async import ExecutorAsync


class ExecutorAsyncSequential(ExecutorAsync):
    """Async executor launching CPU-bounded tasks sequentially"""

    async def async_run_work_cpu(self, work):
        """Executes the work on an item (key, obj), and add the result on
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
        # pylint: disable=W0703
        try:
            # here we do something very bad from the async point of view:
            # we launch a potentially long blocking function:
            ret = work.func_or_cls(obj)
        except Exception as error:
            self.log_exception(error, work.name_no_space, key)
            if self.stop_if_error:
                raise
            ret = error
        else:
            logger.info(
                f"work {work.name_no_space} ({key}) "
                f"done in {time.time() - t_start:.3f} s"
            )

        if work.output_queue is not None:
            work.output_queue[key] = ret
        self.nb_working_workers_cpu -= 1
