"""Base class for executors
===========================

.. autoclass:: ExecutorBase
   :members:
   :private-members:

"""

import os
import sys
from time import time
import signal
from pathlib import Path

from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile

from fluidimage.config import get_config
from fluidimage.topologies.nb_cpu_cores import nb_cores
from fluidimage.util.log import logger, reset_logger

from fluidimage import config_logging

config = get_config()


class ExecutorBase:
    """Base class for executors.

    Parameters
    ----------

    topology : fluidimage.topology

      A Topology from fluidimage.topology.

    """

    def __init__(
        self,
        topology,
        path_dir_result,
        nb_max_workers,
        nb_items_queue_max,
        logging_level="info",
    ):
        self.topology = topology

        if path_dir_result is not None:
            path_dir_result = Path(path_dir_result)
            path_dir_result.mkdir(exist_ok=True)
            self.path_dir_result = path_dir_result
            log = path_dir_result / (
                "log_" + time_as_str() + "_" + str(os.getpid()) + ".txt"
            )
            self._log_file = open(log, "w")

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

            config_logging(logging_level, file=sys.stdout)

        if nb_max_workers is None:
            if config is not None:
                try:
                    nb_max_workers = eval(config["topology"]["nb_max_workers"])
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
            nb_items_queue_max = max(2 * nb_max_workers, 2)
        self.nb_items_queue_max = nb_items_queue_max

        self._has_to_stop = False
        if sys.platform != "win32":

            def handler_signals(signal_number, stack):
                print(
                    "signal {} received: set _has_to_stop to True".format(
                        signal_number
                    )
                )
                self._has_to_stop = True

            signal.signal(12, handler_signals)

    def _init_compute(self):
        self.t_start = time()

    def _reset_std_as_default(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        reset_logger()
        self._log_file.close()

    def _finalize_compute(self):
        self.topology._print_at_exit(time() - self.t_start)
        self._reset_std_as_default()
