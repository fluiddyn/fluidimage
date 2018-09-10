"""

A executor using async for IO and multiprocessing for CPU bounded tasks.

.. autoclass:: ExecutorAsyncSequential
   :members:
   :private-members:

"""

import time

from fluidimage.util import logger, log_memory_usage, cstring

from .exec_async import ExecutorAsync


class ExecutorAsyncSequential(ExecutorAsync):
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
        if work.check_exception(key, obj):
            return

        t_start = time.time()
        log_memory_usage(
            f"{time.time() - self.t_start:.2f} s. Launch work "
            + work.name_no_space
            + f" ({key}). mem usage"
        )
        self.nb_working_workers_cpu += 1
        try:
            # here we do something very bad from the async point of view:
            # we launch a potentially long blocking function:
            ret = work.func_or_cls(obj)
        except Exception as error:
            logger.error(
                cstring(
                    "error during work " f"{work.name_no_space} ({key})",
                    color="FAIL",
                )
            )
            if self.stop_if_error:
                raise
            ret = error
        else:
            logger.info(
                f"work {work.name_no_space} ({key}) "
                f"done in {time.time() - t_start:.3f} s"
            )

        self.nb_working_workers_cpu -= 1
        if work.output_queue is not None:
            work.output_queue[key] = ret
