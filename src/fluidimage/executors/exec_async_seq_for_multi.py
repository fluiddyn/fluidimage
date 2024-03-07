from pathlib import Path
from time import time

from .exec_async_sequential import ExecutorAsyncSequential


class ExecutorAsyncSeqForMulti(ExecutorAsyncSequential):
    """Slightly modified ExecutorAsync"""

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
            path_results = (
                Path(self._log_path).parent
                / f"results_{self.index_process:03}.txt"
            )
            results = (
                "\n".join([Path(path).name for path in self.topology.results])
                + "\n"
            )
            with open(path_results, "w", encoding="utf-8") as file:
                file.write(results)


Executor = ExecutorAsyncSeqForMulti
