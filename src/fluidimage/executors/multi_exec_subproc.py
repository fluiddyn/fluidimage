"""Multi executor based on subprocesses

.. autoclass:: MultiExecutorSubproc
   :members:
   :private-members:

"""

import subprocess
import sys
from time import sleep

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

        splitter = splitter_cls(self.topology.params, self.nb_processes)

        path_dir_params = (
            self.path_dir_result / f"params_files_{self._unique_postfix}"
        )
        path_dir_params.mkdir(exist_ok=True)

        for index_process, params_split in enumerate(
            splitter.iter_over_new_params()
        ):
            p_series = splitter.get_params_series(params_split)
            if (
                len(
                    range(
                        p_series.ind_start, p_series.ind_stop, p_series.ind_step
                    )
                )
                == 0
            ):
                continue

            params_split._set_child(
                "compute_kwargs",
                attribs={
                    "executor": "exec_async_seq_for_multi",
                    "nb_max_workers": 1,
                },
            )
            params_split.compute_kwargs._set_child(
                "kwargs_executor",
                attribs={
                    "path_log": self._log_path.parent
                    / f"process_{index_process:03d}.txt",
                    "t_start": self.t_start,
                    "index_process": index_process,
                },
            )

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

    def _wait_for_all_processes(self):

        running_processes = {
            idx: process for idx, process in enumerate(self.processes)
        }
        return_codes = {}

        while running_processes:
            running_processes_updated = {}
            for idx, process in running_processes.items():
                ret_code = process.poll()
                if ret_code is None:
                    running_processes_updated[idx] = process
                else:
                    return_codes[idx] = ret_code
                    if ret_code != 0:
                        logger.error(process.stderr.read())
            running_processes = running_processes_updated
            sleep(0.1)


Executor = MultiExecutorSubproc
