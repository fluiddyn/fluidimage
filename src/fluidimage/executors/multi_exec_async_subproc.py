"""
Multi executors async using subprocesses
========================================

.. autoclass:: ExecutorAsyncForMultiSubproc
   :members:
   :private-members:

"""

import sys
from pathlib import Path
from subprocess import Popen

from fluidimage.util import logger

from .multi_exec_async import ExecutorAsyncForMulti, MultiExecutorAsync


def main():
    """Create an executor and start it in a process"""

    # topology_this_process, path_input, path_output, log_path

    executor = ExecutorAsyncForMulti(
        topology_this_process,
        path_dir_result,
        sleep_time=sleep_time,
        log_path=log_path,
        logging_level=logging_level,
    )
    executor.t_start = t_start
    executor.compute()


class MultiExecutorAsyncSubproc(MultiExecutorAsync):

    def launch_process(self, topology, ind_process):
        """Launch one process"""

        log_path = Path(
            str(self._log_path).split(".txt")[0] + f"_multi{ind_process:03}.txt"
        )

        self.log_paths.append(log_path)

        process = Popen(
            [
                sys.executable,
                "-m",
                "fluidimage.executors.multi_exec_async_subproc",
            ],
        )

        self.processes.append(process)

    def wait_for_all_processes(self):
        """logging + wait for all processes to finish"""
        logger.info(
            f"logging files: {[log_path.name for log_path in self.log_paths]}"
        )

        # wait until end of all processes

        self.topology.results = results_all = []
        for index, process in enumerate(self.processes):
            try:
                results = process.connection.recv()
            except EOFError:
                logger.error(f"EOFError for process {index} ({process})")
                results = None

            if results is not None:
                results_all.extend(results)

        for process in self.processes:
            process.join()


Executor = MultiExecutorAsync


if __name__ == "__main__":
    main()
