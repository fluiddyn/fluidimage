"""Sequential executor for multi executor

.. autoclass:: ExecutorAsyncSeqForMulti
   :members:
   :private-members:

"""

import sys
from pathlib import Path
from time import time

import trio

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

        if hasattr(self.topology, "results"):
            self.async_funcs["_save_topology_results"] = (
                self._save_topology_results
            )
            self._path_results = (
                Path(self._log_path).parent
                / f"results_{self.index_process:03}.txt"
            )
            self._path_results.touch()

            self._path_num_results = (
                self._path_results.parent
                / f"len_results_{self.index_process:03}.txt"
            )
            with open(self._path_num_results, "w", encoding="utf-8") as file:
                file.write("0\n")

            self._len_saved_results = 0

            sys.stdout = self._log_file

    def _get_file_object_for_logger(self):
        return self._log_file

    def _init_log_path(self):
        self.path_dir_exceptions = Path(self._log_path).parent

    def _init_compute(self):
        self._init_compute_log()

    def _finalize_compute(self):
        self._reset_std_as_default()

        txt = self.topology.make_text_at_exit(time() - self.t_start)
        with open(self._log_path, "a", encoding="utf-8") as file:
            file.write(txt)

        if hasattr(self.topology, "results"):
            self._save_results_names()

    def _save_results_names(self):

        new_results = self.topology.results[self._len_saved_results :]
        self._len_saved_results = len(self.topology.results)

        with open(self._path_num_results, "w", encoding="utf-8") as file:
            file.write(f"{self._len_saved_results}\n")

        if new_results:
            if isinstance(new_results[0], str):
                new_results = [Path(path).name for path in new_results]
            elif hasattr(new_results[0], "name"):
                new_results = [_r.name for _r in new_results]
            else:
                new_results = [str(_r) for _r in new_results]
            new_results = "\n".join(new_results) + "\n"

            with open(self._path_results, "a", encoding="utf-8") as file:
                file.write(new_results)

    async def _save_topology_results(self):
        while not self._has_to_stop:
            self._save_results_names()
            await trio.sleep(1.0)


Executor = ExecutorAsyncSeqForMulti
