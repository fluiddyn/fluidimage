"""Base class for executors
===========================

.. autoclass:: ExecutorBase
   :members:
   :private-members:

.. autoclass:: MultiExecutorBase
   :members:
   :private-members:

"""

import atexit
import copy
import os
import signal
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from time import sleep, time

from rich.console import Console
from rich.progress import Progress

from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile
from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage.config import get_config
from fluidimage.topologies.nb_cpu_cores import nb_cores
from fluidimage.util import (
    get_txt_memory_usage,
    log_error,
    log_memory_usage,
    logger,
    reset_logger,
    safe_eval,
    str_short,
)

config = get_config()

_omp_num_threads_equal_1_at_import = os.environ.get("OMP_NUM_THREADS") == "1"


class ExecutorBase(ABC):
    """Base class for executors.

    Parameters
    ----------

    topology : fluidimage.topology.base.TopologyBase

      A computational topology.

    path_dir_result : str or pathlib.Path

      The path of the directory where the results have to be saved.

    nb_max_workers : int, optional (None)

    nb_items_queue_max : int, optional (None),

    logging_level : str, optional {"info"},

    sleep_time : number, optional {None},

    stop_if_error : bool, optional {False}

    """

    info_job: dict
    path_job_data: Path
    _path_lockfile: Path
    num_expected_results: int
    time_start: str

    def _init_log_path(self):
        self.time_start_str = time_as_str()
        unique_postfix = f"{self.time_start_str}_{os.getpid()}"
        path_job_data = self.path_dir_result / f"job_{unique_postfix}"

        if path_job_data.exists():
            index = 0
            unique_postfix0 = unique_postfix
            while path_job_data.exists():
                index += 1
                unique_postfix = unique_postfix0 + str(index)
                path_job_data = self.path_dir_result / f"log_{unique_postfix}"

        self._unique_postfix = unique_postfix
        self.path_job_data = path_job_data
        self._path_lockfile = self.path_job_data / "is_running.lock"
        self.path_dir_exceptions = self.path_dir_result / f"log_{unique_postfix}"
        self._log_path = self.path_dir_result / f"log_{unique_postfix}.txt"

    def __init__(
        self,
        topology,
        path_dir_result,
        nb_max_workers=None,
        nb_items_queue_max=None,
        logging_level="info",
        sleep_time=None,
        stop_if_error=False,
        path_log=None,
    ):
        if not _omp_num_threads_equal_1_at_import:
            raise SystemError(
                "For performance reason,"
                'the environment variable OMP_NUM_THREADS has to be set to "1" '
                "before executing a Fluidimage topology."
            )

        del sleep_time
        self.topology = topology
        self.logging_level = logging_level
        self.stop_if_error = stop_if_error

        path_dir_result = Path(path_dir_result).resolve()
        path_dir_result.mkdir(exist_ok=True)
        self.path_dir_result = path_dir_result

        self._init_log_path()
        if path_log is not None:
            self._log_path = path_log
        self._log_file = open(self._log_path, "w", encoding="utf-8")

        stdout = sys.stdout
        if isinstance(stdout, MultiFile):
            stdout = sys.__stdout__

        stderr = sys.stderr
        if isinstance(stderr, MultiFile):
            stderr = sys.__stderr__

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

        sys.stdout = MultiFile([stdout, self._log_file])
        sys.stderr = MultiFile([stderr, self._log_file])

        if logging_level:
            for handler in logger.handlers:
                logger.removeHandler(handler)

            from fluidimage import config_logging

            config_logging(logging_level, file=self._get_file_object_for_logger())

        if nb_max_workers is None:
            if config is not None:
                try:
                    nb_max_workers = safe_eval(
                        config["topology"]["nb_max_workers"]
                    )
                except KeyError:
                    pass

        # default nb_max_workers
        # Difficult: trade off between overloading and limitation due to input
        # output.  The user can do much better for a specific case.
        if nb_max_workers is None:
            if nb_cores < 16:
                nb_max_workers = nb_cores + 2
            else:
                nb_max_workers = nb_cores

        self.nb_max_workers = nb_max_workers

        if nb_items_queue_max is None:
            nb_items_queue_max = max(4 * nb_max_workers, 8)
        self.nb_items_queue_max = nb_items_queue_max

        self._has_to_stop = False
        if sys.platform != "win32":

            def handler_signals(signal_number, stack):
                del stack
                print(
                    f"signal {signal_number} received: set _has_to_stop to True "
                    f"({type(self).__name__})."
                )
                self._has_to_stop = True

            signal.signal(12, handler_signals)

        # Picks up async works
        self.works = [
            work
            for work in self.topology.works
            if work.kind is None or "one shot" not in work.kind
        ]

        self._indices_to_be_computed = None

        # to avoid a pylint warning
        self.t_start = None

    def _get_file_object_for_logger(self):
        return sys.stdout

    def _init_compute(self):
        self.t_start = time()
        self._init_num_expected_results()
        if self.num_expected_results == 0:
            return
        self._init_compute_log()
        self._save_job_data()

    def _init_compute_log(self):
        log_memory_usage(time_as_str(2) + ": starting execution. mem usage")

        topology_name = str_short(type(self.topology))
        executor_name = str_short(type(self))
        print("  topology:", topology_name)
        print("  executor:", executor_name)
        print("  nb_cpus_allowed =", nb_cores)
        print("  nb_max_workers =", self.nb_max_workers)
        print("  num_expected_results", self.num_expected_results)
        print("  path_dir_result =", self.path_dir_result)
        print(
            "Monitoring app can be launched with:\n"
            f"fluidimage-monitor {self.path_dir_result}"
        )
        self.info_job = {
            "topology": topology_name,
            "executor": executor_name,
            "nb_cpus_allowed": nb_cores,
            "nb_max_workers": self.nb_max_workers,
            "path_dir_result": self.path_dir_result,
            "num_expected_results": self.num_expected_results,
            "time_start": self.time_start_str,
        }

    def _save_lock_file(self):
        if self._path_lockfile.exists():
            raise RuntimeError(
                f"File {self._path_lockfile} already exists. It usually "
                "means that this directory is already being used by "
                "another process. Alternatively it might be that an "
                "old lockfile has not been deleted (which is a bug). "
                "If no process uses this directory, the lockfile "
                "can safely be removed."
            )
        else:
            with open(self._path_lockfile, "w", encoding="utf-8") as file:
                file.write(time_as_str() + f"\n{os.getpid()}\n")

        atexit.register(self._release_lock)

    def _release_lock(self):
        if self._path_lockfile.exists():
            self._path_lockfile.unlink(missing_ok=True)
            t_as_str = time_as_str()
            path_end_time = self._path_lockfile.with_name("time_end.txt")
            path_end_time.write_text(f"{t_as_str}\n")

    def _save_job_data(self):
        self.path_job_data = self.path_dir_result / f"job_{self._unique_postfix}"
        self.path_job_data.mkdir(exist_ok=False)
        self._save_lock_file()

        params = self.topology.params
        if isinstance(params, dict):
            params = ParamContainer("params", attribs=params)
        params._save_as_xml(self.path_job_data / "params.xml")

        info_job = ParamContainer("info", attribs=self.info_job)
        info_job._save_as_xml(self.path_job_data / "info.xml")

    def _reset_std_as_default(self):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        reset_logger()
        self._log_file.close()

    def _finalize_compute(self):
        log_memory_usage(time_as_str(2) + ": end of `compute`. mem usage")
        self.topology.print_at_exit(time() - self.t_start)
        self._reset_std_as_default()
        self._release_lock()

    def log_in_file(self, *args, sep=" ", end="\n"):
        """Simple write in the log file (without print)"""
        self._log_file.write(sep.join(str(arg) for arg in args) + end)

    def log_in_file_memory_usage(self, txt, color="OKGREEN", end="\n"):
        """Write the memory usage in the log file"""
        self._log_file.write(get_txt_memory_usage(txt, color) + end)

    def exec_one_shot_works(self):
        """
        Execute all "one shot" functions.

        """
        for work in self.topology.works:
            if work.kind is not None and "one shot" in work.kind:
                pretty = str_short(work.func_or_cls.__func__)
                print(f'Running "one_shot" job "{work.name}" ({pretty})')
                work.func_or_cls(work.input_queue, work.output_queue)

    def log_exception(self, exception, work_name, key):
        """Log an exception in a file."""
        path_log = self.path_dir_exceptions / f"exception_{work_name}_{key}.txt"

        log_error(
            "error during work " f"{work_name} ({key}) (logged in {path_log})"
        )

        self.path_dir_exceptions.mkdir(exist_ok=True)

        try:
            parts = traceback.format_exception(
                etype=type(exception), value=exception, tb=exception.__traceback__
            )
        except TypeError:
            # Python >= 3.10
            parts = traceback.format_exception(exception)
        formatted_exception = "".join(parts)

        with open(path_log, "w", encoding="utf-8") as file:
            file.write(
                f"Exception for work {work_name}, key {key}:\n"
                + formatted_exception
            )

    def _init_num_expected_results(self):
        if hasattr(self.topology, "series"):
            self._init_num_expected_results_series()
        else:
            self._init_num_expected_results_first_queue()

    def _init_num_expected_results_series(self):
        if self.topology.how_saving == "complete" and hasattr(
            self.topology, "compute_indices_to_be_computed"
        ):
            self._indices_to_be_computed = (
                self.topology.compute_indices_to_be_computed()
            )
            self.num_expected_results = len(self._indices_to_be_computed)
        else:
            series = self.topology.series
            self.num_expected_results = len(
                range(series.ind_start, series.ind_stop, series.ind_step)
            )

    def _init_num_expected_results_first_queue(self):
        first_work = self.topology.works[0]
        if first_work.input_queue is not None:
            raise NotImplementedError
        if isinstance(first_work.output_queue, tuple):
            raise NotImplementedError
        # fill the first queue
        first_work.func_or_cls(
            input_queue=None, output_queue=first_work.output_queue
        )
        self._first_queue = copy.copy(first_work.output_queue)
        # split the first queue
        self._keys_first_queue = list(self._first_queue.keys())
        self.num_expected_results = len(self._keys_first_queue)


class MultiExecutorBase(ExecutorBase):
    """Manage the multi-executor mode

     This class is not the one whose really compute the topology. The topology is
     split and each slice is computed with an ExecutorAsync

    Parameters
    ----------

    nb_max_workers : None, int

      Limits the numbers of workers working in the same time.

    nb_items_queue_max : None, int

      Limits the numbers of items that can be in a output_queue.

    sleep_time : None, float

      Defines the waiting time (from trio.sleep) of a function. Async functions
      await `trio.sleep(sleep_time)` when they have done a work on an item, and
      when there is nothing in their input_queue.

    """

    errors: dict

    def __init__(
        self,
        topology,
        path_dir_result,
        nb_max_workers=None,
        nb_items_queue_max=None,
        sleep_time=0.01,
        logging_level="info",
        stop_if_error=False,
    ):
        if stop_if_error:
            raise NotImplementedError

        super().__init__(
            topology,
            path_dir_result,
            nb_max_workers,
            nb_items_queue_max,
            logging_level=logging_level,
        )

        self.sleep_time = sleep_time
        self.nb_processes = self.nb_max_workers
        self.processes = []

        # to avoid a pylint warning
        self.log_paths = None

    def _init_log_path(self):
        super()._init_log_path()
        path_dir_log = self.path_dir_exceptions
        path_dir_log.mkdir(exist_ok=True)
        self._log_path = path_dir_log / (path_dir_log.name + ".txt")

    @abstractmethod
    def _start_processes(self):
        """Start the processes doing the hard work"""

    def compute(self):
        """Compute the topology."""

        self._init_compute()

        if self.num_expected_results == 0:
            print("Nothing to be computed")
            return

        self.log_paths = []

        if sys.platform != "win32":

            def handler_signals(signal_number, stack):
                del stack
                print(
                    f"signal {signal_number} received: set _has_to_stop to True "
                    f"({type(self).__name__})."
                )
                self._has_to_stop = True
                for process in self.processes:
                    os.kill(process.pid, signal_number)

            signal.signal(12, handler_signals)

        self._start_processes()
        self._wait_for_all_processes()
        self._finalize_compute()

    def _poll_return_code(self, process):
        return process.poll()

    def _join_processes(self):
        """Join the processes"""

    def _wait_for_all_processes(self):

        running_processes = {
            idx: process for idx, process in enumerate(self.processes)
        }
        running_processes_updated = {}
        return_codes = {}
        self.errors = {}

        num_results_vs_idx_process = [0 for idx in range(len(self.processes))]
        paths_len_results = [
            self.path_job_data / f"len_results_{idx:03}.txt"
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
                    ret_code = self._poll_return_code(process)
                    if ret_code is None:
                        running_processes_updated[idx] = process
                    else:
                        return_codes[idx] = ret_code
                        if ret_code != 0:
                            try:
                                error = process.stderr.read()
                            except AttributeError:
                                error = f"{ret_code = }"
                            self.errors[idx] = error
                            logger.error(error)

                    if paths_len_results[idx].exists():
                        with open(
                            paths_len_results[idx], encoding="utf-8"
                        ) as file:
                            content = file.read()
                            if content:
                                num_results_vs_idx_process[idx] = int(content)

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

        self._join_processes()

    def _finalize_compute(self):
        self.topology.results = results = []
        for path in self.path_job_data.glob("results_*.txt"):
            with open(path, encoding="utf-8") as file:
                results.extend(line.strip() for line in file.readlines())

        if self.errors:
            text_error = (
                f"{len(self.errors)} sub-executors failed "
                f"(over {len(self.processes)} processes)."
            )
            for idx_process, error in self.errors.items():
                text_error += f"\n{idx_process = }\nerror:\n{error}"
            print(text_error, file=sys.stderr)

        super()._finalize_compute()

        if self.errors:
            raise RuntimeError(text_error)
