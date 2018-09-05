"""Topology base (:mod:`fluidimage.topologies.experimental.base`)
=================================================================

.. autoclass:: TopologyBase
   :members:
   :private-members:

"""

from warnings import warn

from ..executors import executors, ExecutorBase


class MyObj:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.__dict__.update(kwargs)

    def __repr__(self):
        return super().__repr__() + "\n" + self._kwargs.__repr__()


class Queue(dict):
    """Represent a queue"""

    def __init__(self, name, kind=None):
        self.name = name
        self.kind = kind


class Work(MyObj):
    """Represent a work"""


class TopologyBase:
    """Base class for topologies of processing.

    This class is meant to be subclassed, not instantiated directly.

    Parameters
    ----------

    path_dir_result : None, str

    logging_level : None,  {'warning', 'info', 'debug', ...}

    nb_max_workers : None, int

    """

    def __init__(
        self, path_dir_result=None, logging_level="info", nb_max_workers=None
    ):

        self.path_dir_result = path_dir_result
        self.logging_level = logging_level
        self.nb_max_workers = nb_max_workers

        self.queues = []
        self.works = []
        self.works_dict = {}

    def add_queue(self, name: str, kind: str = None):
        """Create a new queue."""
        queue = Queue(name=name, kind=kind)
        self.queues.append(queue)
        return queue

    def add_work(
        self,
        name: str,
        func_or_cls,
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

        if name in self.works_dict:
            raise ValueError(f"The name {name} is already used.")
        self.works_dict[name] = work

    def compute(self, executor="exec_async", nb_max_workers=None):
        """Compute (run all works to be done). """
        if executor is None:
            executor = "exec_async"

        if not isinstance(executor, ExecutorBase):
            if executor not in executors:
                raise NotImplementedError

            exec_class = executors[executor]
            self.executor = exec_class(
                self,
                path_dir_result=self.path_dir_result,
                nb_max_workers=nb_max_workers,
                logging_level=self.logging_level,
            )

        self.executor.compute()

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
            txt += "\npath results:\n" + str(self.path_dir_result)

        print(txt)

    def make_code_graphviz(self, name_file):
        """Generate the graphviz / dot code."""
        name_file = str(name_file)

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

    def print_queues(self):
        """
        Print the length of all queues
        """
        for queue in self.topology.queues:
            print(f"queue {queue.name}, length: {len(queue)}")
        print("\n")
