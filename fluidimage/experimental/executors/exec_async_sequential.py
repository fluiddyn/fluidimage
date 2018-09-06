"""

A executor using async for IO and multiprocessing for CPU bounded tasks.

.. autoclass:: ExecutorAsyncSequential
   :members:
   :private-members:

"""

import time

from fluidimage.util import logger, log_memory_usage

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
        t_start = time.time()
        log_memory_usage(
            "{:.2f} s. ".format(time.time() - self.t_start)
            + "Launch work "
            + work.name.replace(" ", "_")
            + " ({}). mem usage".format(key)
        )
        self.nb_working_workers_cpu += 1
        # here we do something very bad from the async point of view:
        # we launch a potentially long blocking function:
        ret = work.func_or_cls(obj)

        if work.output_queue is not None:
            work.output_queue[key] = ret
        self.nb_working_workers_cpu -= 1
        logger.info(
            "work {} ({}) done in {:.3f} s".format(
                work.name.replace(" ", "_"), key, time.time() - t_start
            )
        )
