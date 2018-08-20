"""Topology base (:mod:`fluidimage.topologies.experimental.base`)
=================================================================

.. autoclass:: TopologyBase
   :members:
   :private-members:

"""

from time import time
import signal
import sys
import os
from warnings import warn

from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile

from fluidimage.util.util import logger
from fluidimage import config_logging
from fluidimage.experimental.executors.nb_workers import (
    nb_max_workers as _nb_max_workers,
    nb_cores,
)

_stdout_at_import = sys.stdout
_stderr_at_import = sys.stderr


class MyObj:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.__dict__.update(kwargs)

    def __repr__(self):
        return super().__repr__() + "\n" + self._kwargs.__repr__()


class Queue(MyObj):
    """Represent a queue"""

    def __init__(self, **kwargs):
        super(Queue, self).__init__(**kwargs)
        self.queue = None


class Work(MyObj):
    """Represent a work"""

    def have_to_work(self):
        print("{} have to work ?".format(self.name))
        if isinstance(self.input_queue, tuple):
            for q in self.input_queue:
                if not q.queue:  # if a queue is empty
                    return False
        else:
            if not self.input_queue.queue:
                return False
        return True


class TopologyBase:
    """Base class for topologies of processing.

    Parameters
    ----------

    queues : list

    path_output : None

    logging_level : 'info'

    nb_max_workers : None

    """

    def __init__(
        self, path_output=None, logging_level="info", nb_max_workers=None
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

        self.queues = []
        self.works = []

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
        self.t_start = time()

    def add_queue(self, name: str, kind: str = None):
        """Create a new queue."""
        queue = Queue(name=name, kind=kind)
        self.queues.append(queue)
        return queue

    def add_work(
        self,
        name: str,
        func_or_cls=None,
        params_cls=None,
        input_queue=None,
        output_queue=None,
        kind: str = None,
    ):
        """Create a new work relating queues."""
        if func_or_cls is None:
            warn(f'func_or_cls is None for work "{name}"')

        work = Work(
            name=name,
            input_queue=input_queue,
            func_or_cls=func_or_cls,
            params_cls=params_cls,
            output_queue=output_queue,
            kind=kind,
        )
        self.works.append(work)

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
            txt += " ({} results, {:.2f} s/result).".format(
                nb_results, time_since_start / nb_results
            )
        else:
            txt += "."

        if hasattr(self, "path_dir_result"):
            txt += "\npath results:\n" + self.path_dir_result

        print(txt)

    def make_code_graphviz(self, name_file):
        """Generate the graphviz / dot code."""

        if name_file.endswith(".dot"):
            name_file = name_file[:-4]

        code = "digraph {\nrankdir = LR\ncompound=true\n"
        # waiting queues
        code += '\nnode [shape="record"]\n'
        txt_queue = (
            '{name_quoted:40s} [label="<f0> {name}|'
            + "|".join(["<f{}>".format(i) for i in range(1, 5)])
            + '"]\n'
        )

        for q in self.queues:
            name_quoted = '"{}"'.format(q.name)
            code += txt_queue.format(name=q.name, name_quoted=name_quoted)

        # works and links
        code += '\nnode [shape="ellipse"]\n'

        txt_work = '{:40s} [label="{}",color = "{}"]\n'

        for work in self.works:
            name_work = work.name
            color = "Black"
            if work.kind is not None:
                if "io" in work.kind:
                    color = "Green"
            code += txt_work.format(
                '"{}"'.format(name_work, color), name_work, color
            )

        code += "\n"

        str_link = (
            '{:40s} -> "{}" [arrowhead = "{}", style = "{}", color = "{}"]\n'
        )

        for work in self.works:
            name_work = work.name
            arrowhead = "normal"
            style = "dashed"
            color = "Black"
            if work.kind is not None:
                if "one shot" in work.kind:
                    style = "filled"
                if "global" in work.kind:
                    arrowhead = "odiamond"
                if "io" in work.kind:
                    color = "Green"
            if work.input_queue is not None:
                queues = work.input_queue
                if isinstance(queues, Queue):
                    queues = (queues,)
                for queue in queues:
                    code += str_link.format(
                        '"' + queue.name + '"', name_work, arrowhead, style, color
                    )
            if work.output_queue is not None:
                queues = work.output_queue
                if isinstance(queues, Queue):
                    queues = (queues,)
                for queue in queues:
                    code += str_link.format(
                        '"' + name_work + '"', queue.name, arrowhead, style, color
                    )

        # Legend
        code += "\n subgraph cluster_01 {"
        code += '\n node [height="0px", width="0px",shape=none,];'
        code += "\n edge [ minlen = 1,];"
        code += '\n label = "Legend";'
        code += '\n key [label=<<table border="0" cellpadding="2" cellspacing="0" cellborder="0">'
        code += '\n <tr><td align="right" port="i1">Global</td></tr>'
        code += '\n <tr><td align="right" port="i2">One Shot</td></tr>'
        code += '\n <tr><td align="right" port="i3">Multiple Shot</td></tr>'
        code += '\n <tr><td align="right" port="i4">I/O</td></tr>'
        code += "\n </table>>]"
        code += '\n key2 [label=<<table border="0" cellpadding="2" cellspacing="0" cellborder="0">'
        code += '\n<tr><td port="i1">&nbsp;</td></tr>'
        code += '\n<tr><td port="i2">&nbsp;</td></tr>'
        code += '\n<tr><td port="i3">&nbsp;</td></tr>'
        code += '\n<tr><td port="i4">&nbsp;</td></tr>'
        code += "\n </table>>]"
        code += '\n  key:i1:e -> key2:i1:w [arrowhead = "odiamond"]'
        code += '\n  key:i2:e -> key2:i2:w [arrowhead = "none"]'
        code += '\n  key:i3:e -> key2:i3:w [style = "dashed", arrowhead = "none"]'
        code += '\n  key:i4:e -> key2:i4:w [arrowhead = "none", color="Green"]'
        code += "\n } \n"

        code += "}\n"

        with open(name_file + ".dot", "w") as file:
            file.write(code)

        print(
            "A graph can be produced with graphviz with one of these commands:\n"
            f"dot {name_file}.dot -Tpng -o {name_file}.png && eog {name_file}.png\n"
            f"dot {name_file}.dot -Tx11"
        )


if __name__ == "__main__":
    topo = TopologyBase()

    queue_names_piv = topo.add_queue("names piv")
    queue_names_couples = topo.add_queue("names couples")
    queue_paths = topo.add_queue("paths")
    queue_arrays = topo.add_queue("arrays")
    queue_couples = topo.add_queue("couples of arrays")
    queue_piv = topo.add_queue("piv")

    topo.add_work(
        "fill names piv",
        output_queue=queue_names_piv,
        kind=("global", "one shot"),
    )
    topo.add_work(
        "fill (names couples, paths)",
        input_queue=queue_names_piv,
        output_queue=(queue_names_couples, queue_paths),
        kind=("global", "one shot"),
    )
    topo.add_work(
        "path -> arrays",
        input_queue=queue_paths,
        output_queue=queue_arrays,
        kind="io",
    )
    topo.add_work(
        "make couples arrays",
        input_queue=(queue_arrays, queue_names_couples),
        output_queue=queue_couples,
        kind="global",
    )

    topo.add_work(
        "couples -> piv", input_queue=queue_couples, output_queue=queue_piv
    )

    topo.add_work("save piv", input_queue=queue_piv, kind="io")

    topo.make_code_graphviz("tmp.dot")
