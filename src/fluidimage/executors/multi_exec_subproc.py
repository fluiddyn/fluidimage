"""Multi executor based on subprocesses

.. autoclass:: MultiExecutorSubproc
   :members:
   :private-members:

"""

import subprocess
import sys
from copy import deepcopy
from time import sleep

from rich.console import Console
from rich.progress import Progress

from fluiddyn import time_as_str
from fluidimage.util import logger

from .base import MultiExecutorBase


class MultiExecutorSubproc(MultiExecutorBase):
    """Multi executor based on subprocesses and splitters"""

    def _start_processes(self):

        try:
            splitter_cls = self.topology.Splitter
        except AttributeError as error:
            raise ValueError(
                "MultiExecutorSubproc can only execute "
                "topologies with a Splitter."
            ) from error

        params = deepcopy(self.topology.params)

        try:
            params._set_child(
                "compute_kwargs",
                attribs={
                    "executor": "exec_async_seq_for_multi",
                    "nb_max_workers": 1,
                },
            )
        except ValueError:
            params.compute_kwargs.executor = "exec_async_seq_for_multi"
            params.compute_kwargs.nb_max_workers = 1

        try:
            params.compute_kwargs._set_child(
                "kwargs_executor",
                attribs={
                    "path_log": None,
                    "t_start": self.t_start,
                    "index_process": None,
                },
            )
        except ValueError:
            params.compute_kwargs.kwargs_executor.t_start = self.t_start

        if hasattr(self.topology, "how_saving"):
            params.saving.how = self.topology.how_saving

        if hasattr(self.topology, ""):
            params.saving.path = self.topology.path_dir_result

        splitter = splitter_cls(params, self.nb_processes, self.topology)
        self.num_expected_results = splitter.num_expected_results

        path_dir_params = (
            self.path_dir_result / f"params_files_{self._unique_postfix}"
        )
        path_dir_params.mkdir(exist_ok=True)

        if (
            hasattr(self.topology, "how_saving")
            and self.topology.how_saving == "complete"
            and hasattr(splitter, "save_indices_files")
        ):
            splitter.save_indices_files(path_dir_params)

        for index_process, params_split in enumerate(
            splitter.iter_over_new_params()
        ):
            kwargs_executor = params_split.compute_kwargs.kwargs_executor
            kwargs_executor.path_log = (
                self._log_path.parent / f"process_{index_process:03d}.txt"
            )
            kwargs_executor.index_process = index_process

            path_params = path_dir_params / f"params{index_process:00d}.xml"
            params_split._save_as_xml(path_params)

            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "fluidimage.run_from_xml",
                    str(path_params),
                ],
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.processes.append(process)
            # shift a bit the imports
            sleep(0.2)

        logger.info(
            "%s: %s sequential executors launched in parallel",
            time_as_str(2),
            len(self.processes),
        )

    def _wait_for_all_processes(self):

        running_processes = {
            idx: process for idx, process in enumerate(self.processes)
        }
        running_processes_updated = {}
        return_codes = {}
        errors = {}

        num_results_vs_idx_process = [0 for idx in range(len(self.processes))]
        paths_len_results = [
            self._log_path.parent / f"len_results_{idx:03}.txt"
            for idx in range(len(self.processes))
        ]
        num_results = num_results_previous = 0

        console = Console(file=sys.__stdout__)

        with Progress(console=console) as progress:

            progress_task = progress.add_task(
                "[green]Computation", total=self.num_expected_results
            )

            while running_processes:
                sleep(0.2)
                for idx, process in running_processes.items():
                    ret_code = process.poll()
                    if ret_code is None:
                        running_processes_updated[idx] = process
                        if paths_len_results[idx].exists():
                            with open(
                                paths_len_results[idx], encoding="utf-8"
                            ) as file:
                                content = file.readline()
                                if content:
                                    num_results_vs_idx_process[idx] = int(content)
                    else:
                        return_codes[idx] = ret_code
                        if ret_code != 0:
                            error = process.stderr.read()
                            errors[idx] = error
                            logger.error(error)

                num_results = sum(num_results_vs_idx_process)
                if num_results != num_results_previous:
                    if num_results_previous == 0:
                        print(f"{time_as_str(2)}: first result detected")
                    num_results_previous = num_results
                    progress.update(progress_task, completed=num_results)
                running_processes, running_processes_updated = (
                    running_processes_updated,
                    running_processes,
                )
                running_processes_updated.clear()

        if errors:
            raise RuntimeError(
                f"{len(errors)} sub-executors failed (over {len(self.processes)} processes)."
            )

    def _finalize_compute(self):
        self.topology.results = results = []
        for path in self._log_path.parent.glob("results_*.txt"):
            with open(path, encoding="utf-8") as file:
                results.extend(file.readlines())

        super()._finalize_compute()


Executor = MultiExecutorSubproc
