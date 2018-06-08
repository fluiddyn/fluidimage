"""Topology base (:mod:`fluidimage.topologies.experimental.base`)
=================================================================

.. autoclass:: TopologyBase
   :members:
   :private-members:

"""

import signal
import sys
import os

from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile

from ..util.util import logger
from .. import config_logging
from .nb_workers import nb_max_workers as _nb_max_workers, nb_cores

_stdout_at_import = sys.stdout
_stderr_at_import = sys.stderr


class TopologyBase(object):
    """Base class for topologies of processing.

    Parameters
    ----------

    queues : list

    path_output : None

    logging_level : 'info'

    nb_max_workers : None

    """

    def __init__(self,
                 path_output=None,
                 logging_level="info",
                 nb_max_workers=None,
    ):

        if path_output is not None:
            if not os.path.exists(path_output):
                os.makedirs(path_output)
            self.path_output = path_output
            log = os.path.join(
                path_output,
                "log_" + time_as_str() + "_" + str(os.getpid()) + ".txt",
            )
            self._log_file = open(log, "w")

            stdout = sys.stdout
            if isinstance(stdout, MultiFile):
                stdout = _stdout_at_import

            stderr = sys.stderr
            if isinstance(stderr, MultiFile):
                stderr = _stderr_at_import

            sys.stdout = MultiFile([stdout, self._log_file])
            sys.stderr = MultiFile([stderr, self._log_file])

        if logging_level is not None:
            for handler in logger.handlers:
                logger.removeHandler(handler)

            config_logging(logging_level, file=sys.stdout)

        if nb_max_workers is None:
            nb_max_workers = _nb_max_workers

        self.nb_max_workers_io = max(int(nb_max_workers * 0.8), 2)
        self.nb_max_launch = max(int(self.nb_max_workers_io), 1)

        if nb_max_workers < 1:
            raise ValueError("nb_max_workers < 1")

        print("nb_cpus_allowed = {}".format(nb_cores))
        print("nb_max_workers = ", nb_max_workers)
        print("nb_max_workers_io = ", self.nb_max_workers_io)

        self.nb_max_workers = nb_max_workers
        self.nb_cores = nb_cores
        self.nb_items_lim = max(2 * nb_max_workers, 2)

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

    def compute(self, sequential=None, has_to_exit=True, executer=None):
        """Compute (run all works to be done).

        Parameters
        ----------

        sequential : None

          If bool(sequential) is True, the computations are run in sequential
          (useful for debugging).

        has_to_exit : True

          If bool(has_to_exit) is True and if the computation has to stop
          because of a signal 12 (cluster), a signal 99 is sent at exit.

        """
        if executer is None:
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.executer = executer

        self.executer.compute(sequential=sequential, has_to_exit=has_to_exit)
        self._reset_std_as_default()

    def _reset_std_as_default(self):
        sys.stdout = _stdout_at_import
        sys.stderr = _stderr_at_import
        self._log_file.close()

    def _print_at_exit(self, time_since_start):
        """Print information before exit."""
        txt = "Stop compute after t = {:.2f} s".format(time_since_start)
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += (
                " ({} results, {:.2f} s/result).".format(
                    nb_results, time_since_start / nb_results
                )
            )
        else:
            txt += "."

        if hasattr(self, "path_dir_result"):
            txt += "\npath results:\n" + self.path_dir_result

        print(txt)

    def make_code_graphviz(self, name_file):
        """Generate the graphviz / dot code."""

        code = "digraph {\nrankdir = LR\ncompound=true\n"
        # waiting queues
        code += '\nnode [shape="record"]\n'

        txt_queue = '"{name}"\t[label="<f0> {name}|' + "|".join(
            ["<f{}>".format(i) for i in range(1, 5)]
        ) + '"]\n'

        for q in self.queues:
            code += txt_queue.format(name=q.name)

        # works and links
        code += '\nnode [shape="ellipse"]\n'

        txt_work = '"{name}"\t[label="{name}"]'
        for q in self.queues:
            name_work = q.work_name or str(q.work)
            code += txt_work.format(name=name_work)
            code += '"{}" -> "{}"'.format(q.name, name_work)
            if hasattr(q.destination, "name"):
                code += '"{}" -> "{}"'.format(name_work, q.destination.name)

        code += "}\n"

        with open(name_file, "w") as f:
            f.write(code)

        print(
            "A graph can be produced with one of these commands:\n"
            "dot topo.dot -Tpng -o topo.png\n"
            "dot topo.dot -Tx11"
        )
