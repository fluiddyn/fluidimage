"""Execute a topology sequentially
==================================

.. autoclass:: ExecutorBase
   :members:
   :private-members:

"""

import time

from .base import ExecutorBase


class ExecutorSequential(ExecutorBase):
    """Execute a topology sequentially"""

    def compute(self):
        """Compute the whole topology."""
        self._init_compute()
        self.exec_one_shot_works()
        self._run_works()
        self._finalize_compute()
        if hasattr(self.topology, "results"):
            self._save_results_names()

    def _run_works(self):
        while not all([len(queue) == 0 for queue in self.topology.queues]):
            for work in self.works:
                # global functions
                if work.kind is not None and "global" in work.kind:
                    if (
                        work.output_queue is not None
                        and len(work.output_queue) > self.nb_items_queue_max
                    ):
                        continue

                    work.func_or_cls(work.input_queue, work.output_queue)

                else:
                    if not work.input_queue:
                        continue

                    key, obj = work.input_queue.pop_first_item()

                    if work.check_exception(key, obj):
                        continue

                    t_start = time.time()
                    self.log_in_file_memory_usage(
                        f"{time.time() - self.t_start:.2f} s. Launch work "
                        + work.name_no_space
                        + f" ({key}). mem usage"
                    )

                    arg = work.prepare_argument(key, obj)

                    # pylint: disable=W0703
                    try:
                        ret = work.func_or_cls(arg)
                    except Exception as error:
                        self.log_exception(error, work.name_no_space, key)
                        if self.stop_if_error:
                            raise
                        ret = error
                    else:
                        self.log_in_file(
                            f"work {work.name_no_space} ({key}) "
                            f"done in {time.time() - t_start:.3f} s"
                        )

                    if work.output_queue is not None:
                        work.output_queue[key] = ret

            if hasattr(self.topology, "results"):
                self._save_results_names()

    def _init_compute_log(self):
        self.nb_max_workers = 1
        super()._init_compute_log()

    def _save_job_data(self):
        super()._save_job_data()
        if hasattr(self.topology, "results"):
            self._init_results_log(self.path_job_data)


Executor = ExecutorSequential
