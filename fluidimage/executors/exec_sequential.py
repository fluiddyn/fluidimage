"""Execute a topology sequentially (:mod:`fluidimage.executors.exec_async_sequential`)
======================================================================================

.. autoclass:: ExecutorBase
   :members:
   :private-members:

"""

import time

from fluidimage.util import cstring, log_memory_usage, logger

from .base import ExecutorBase


class ExecutorSequential(ExecutorBase):
    """Execute a topology sequentially"""

    def compute(self):
        """Compute the whole topology.

        """
        self._init_compute()
        self.exec_one_shot_works()
        self._run_works()
        self._finalize_compute()

    def _run_works(self):

        while not all([len(queue) == 0 for queue in self.topology.queues]):

            for work in self.works:

                # global functions
                if work.kind is not None and "global" in work.kind:
                    if len(work.output_queue) > self.nb_items_queue_max:
                        continue

                    work.func_or_cls(work.input_queue, work.output_queue)

                else:
                    if not work.input_queue:
                        continue

                    key, obj = work.input_queue.pop_first_item()

                    if work.check_exception(key, obj):
                        continue

                    t_start = time.time()
                    log_memory_usage(
                        f"{time.time() - self.t_start:.2f} s. Launch work "
                        + work.name_no_space
                        + f" ({key}). mem usage"
                    )
                    # pylint: disable=W0703
                    try:
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
