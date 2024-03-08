"""Base class for executors
===========================

.. autoclass:: ExecutorBase
   :members:
   :private-members:

.. autoclass:: MultiExecutorBase
   :members:
   :private-members:

"""

import os
import signal
import sys
import traceback
from pathlib import Path
from time import time

from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile
from fluidimage.config import get_config
from fluidimage.topologies.nb_cpu_cores import nb_cores
from fluidimage.util import (
    log_error,
    log_memory_usage,
    logger,
    reset_logger,
    safe_eval,
    str_short,
)

config = get_config()

_omp_num_threads_equal_1_at_import = os.environ.get("OMP_NUM_THREADS") == "1"


class ExecutorBase:
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

    def _init_log_path(self):
        name = f"log_{self._unique_postfix}"
        self.path_dir_exceptions = self.path_dir_result / name
        self._log_path = self.path_dir_result / (name + ".txt")

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

        path_dir_result = Path(path_dir_result)
        path_dir_result.mkdir(exist_ok=True)
        self.path_dir_result = path_dir_result

        self._unique_postfix = f"{time_as_str()}_{os.getpid()}"

        self._init_log_path()
        if path_log is not None:
            self._log_path = path_log
        self._log_file = open(self._log_path, "w")

        stdout = sys.stdout
        if isinstance(stdout, MultiFile):
            stdout = sys.__stdout__

        stderr = sys.stderr
        if isinstance(stderr, MultiFile):
            stderr = sys.__stderr__

        sys.stdout = MultiFile([stdout, self._log_file])
        sys.stderr = MultiFile([stderr, self._log_file])

        if logging_level:
            for handler in logger.handlers:
                logger.removeHandler(handler)

            from fluidimage import config_logging

            config_logging(logging_level, file=sys.stdout)

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

        # to avoid a pylint warning
        self.t_start = None

    def _init_compute(self):
        self.t_start = time()
        self._init_compute_log()

    def _init_compute_log(self):
        log_memory_usage(time_as_str(2) + ": starting execution. mem usage")
        logger.info("  topology: %s", str_short(type(self.topology)))
        logger.info("  executor: %s", str_short(type(self)))
        logger.info("  nb_cpus_allowed = %s", nb_cores)
        logger.info("  nb_max_workers = %s", self.nb_max_workers)
        logger.info("  path_dir_result = %s", self.path_dir_result)

    def _reset_std_as_default(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        reset_logger()
        self._log_file.close()

    def _finalize_compute(self):
        log_memory_usage(time_as_str(2) + ": end of `compute`. mem usage")
        self.topology.print_at_exit(time() - self.t_start)
        self._reset_std_as_default()

    def exec_one_shot_works(self):
        """
        Execute all "one shot" functions.

        """
        for work in self.topology.works:
            if work.kind is not None and "one shot" in work.kind:
                pretty = str_short(work.func_or_cls.__func__)
                logger.info(f'Running "one_shot" job "{work.name}" ({pretty})')
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
        formated_exception = "".join(parts)

        with open(path_log, "w") as file:
            file.write(
                f"Exception for work {work_name}, key {key}:\n"
                + formated_exception
            )


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
        name = f"log_{self._unique_postfix}"
        path_dir_log = self.path_dir_exceptions = self.path_dir_result / name
        path_dir_log.mkdir(exist_ok=True)
        self._log_path = path_dir_log / (name + ".txt")

    def compute(self):
        """Compute the topology."""

        self._init_compute()
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
