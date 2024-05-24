"""Sequential executor for multi executor

.. autoclass:: ExecutorAsyncSeqForMulti
   :members:
   :private-members:

"""

import sys
from pathlib import Path
from time import time

import trio

from fluiddyn import time_as_str

from .exec_async_sequential import ExecutorAsyncSequential


class ExecutorAsyncSeqForMulti(ExecutorAsyncSequential):
    """Slightly modified ExecutorAsyncSequential"""

    def __init__(
        self,
        topology,
        path_dir_result,
        nb_max_workers=1,
        sleep_time=0.01,
        logging_level="info",
        stop_if_error=False,
        path_log=None,
        t_start=None,
        index_process=None,
    ):
        if stop_if_error:
            raise NotImplementedError(
                "stop_if_error not implemented for ExecutorAsyncForMulti"
            )

        self._log_path = path_log
        topology.executor = self
        super().__init__(
            topology,
            path_dir_result,
            nb_max_workers=nb_max_workers,
            nb_items_queue_max=8,
            sleep_time=sleep_time,
            logging_level=logging_level,
            path_log=path_log,
        )

        self.t_start = t_start
        self.index_process = index_process

        # No need to correctly set num_expected_results for this class
        self.num_expected_results = None

        if hasattr(self.topology, "results"):
            self.async_funcs["_save_topology_results"] = (
                self._save_topology_results
            )
            path_log_dir = Path(self._log_path).parent
            self.path_job_data = path_log_dir.with_name(
                "job" + path_log_dir.name[3:]
            )
            self._init_results_log(self.path_job_data)

            sys.stdout = self._log_file

    def _get_file_object_for_logger(self):
        return self._log_file

    def _init_log_path(self):
        self.path_dir_exceptions = Path(self._log_path).parent

    def _init_compute(self):
        self.time_start_str = time_as_str()
        self._init_compute_log()
        if hasattr(self, "_path_results"):
            self._path_results.touch()
            with open(self._path_num_results, "w", encoding="utf-8") as file:
                file.write("0\n")

    def _init_num_expected_results(self):
        pass

    def _finalize_compute(self):
        self._reset_std_as_default()

        txt = self.topology.make_text_at_exit(time() - self.t_start)
        with open(self._log_path, "a", encoding="utf-8") as file:
            file.write(txt)

        if hasattr(self.topology, "results"):
            self._save_results_names()

    async def _save_topology_results(self):
        while not self._has_to_stop:
            self._save_results_names()
            await trio.sleep(1.0)


Executor = ExecutorAsyncSeqForMulti
