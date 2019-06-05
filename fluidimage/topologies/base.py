"""Topology base (:mod:`fluidimage.topologies.base`)
====================================================

.. autoclass:: Work
   :members:
   :private-members:

.. autoclass:: Queue
   :members:
   :private-members:

.. autoclass:: TopologyBase
   :members:
   :private-members:

"""

from collections import OrderedDict
from warnings import warn

from fluidimage.util import cstring, logger

from ..executors import ExecutorBase, executors


class Work:
    """Represent a work"""

    def __init__(
        self,
        name: str,
        func_or_cls,
        params_cls=None,
        input_queue=None,
        output_queue=None,
        kind: str = None,
    ):
        self._kwargs = dict(
            name=name,
            func_or_cls=func_or_cls,
            params_cls=params_cls,
            input_queue=input_queue,
            output_queue=output_queue,
            kind=kind,
        )
        # to avoid a pylint warning
        self.name = None

        self.__dict__.update(self._kwargs)
        self.name_no_space = self.name.replace(" ", "_")

    def __repr__(self):
        return super().__repr__() + f"\n{self._kwargs}"

    def check_exception(self, key, obj):
        """Check if `obj` is an exception"""
        if isinstance(obj, Exception):
            if self.output_queue is not None:
                self.output_queue[key] = obj
            else:
                logger.error(
                    cstring(
                        f"work {self.name_no_space} ({key}) "
                        "can not be done because of a previously "
                        "raised exception.",
                        color="FAIL",
                    )
                )
            return True
        return False


class Queue(OrderedDict):
    """Represent a queue"""

    def __init__(self, name, kind=None):
        self.name = name
        self.kind = kind
        super().__init__()

    def __repr__(self):
        return f'\nqueue "{self.name}": ' + super().__repr__()

    def __copy__(self):
        newone = type(self)(self.name, kind=self.kind)
        newone.__dict__.update(self.__dict__)

        for key, values in self.items():
            newone[key] = values

        return newone

    def pop_first_item(self):
        return self.popitem(last=False)


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
        self.executor = None

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

    def compute(
        self,
        executor=None,
        nb_max_workers=None,
        sleep_time=0.01,
        sequential=False,
        stop_if_error=False,
    ):
        """Compute (run the works until all queues are empty).

        Parameters
        ----------

        executor : str or fluidimage.executors.base.ExecutorBase, optional

          If None, ``executor="multi_exec_async"``

        nb_max_workers : int, optional

        sleep_time : number, optional {0.01}

        sequential : bool, optional {False}

        stop_if_error : bool, optional {False}

        """

        if sequential:
            if executor is not None and executor != "exec_sequential":
                raise ValueError(
                    "Incompatible arguments sequential=True and "
                    f"executor={executor}"
                )
            executor = "exec_sequential"

        if executor is None:
            # fastest and safest executor for most cases
            executor = "multi_exec_async"

        if not isinstance(executor, ExecutorBase):
            if executor not in executors:
                raise NotImplementedError(f"executor {executor} does not exist")

            exec_class = executors[executor]
            self.executor = exec_class(
                self,
                path_dir_result=self.path_dir_result,
                nb_max_workers=nb_max_workers,
                sleep_time=sleep_time,
                logging_level=self.logging_level,
                stop_if_error=stop_if_error,
            )

        self.executor.compute()

    def make_text_at_exit(self, time_since_start):
        """Make a text printed before exit."""
        txt = f"Stop compute after t = {time_since_start:.2f} s"
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

        return txt

    def print_at_exit(self, time_since_start):
        """Print information before exit."""
        print(self.make_text_at_exit(time_since_start))

    def make_code_graphviz(self, name_file="tmp.dot"):
        """Generate the graphviz / dot code.

        This method only generates a graphviz code. The graph can be visualized
        with for example::

          dot {name_file}.dot -Tpng -o {name_file}.png && eog {name_file}.png

        """
        name_file = str(name_file)

        if name_file.endswith(".dot"):
            name_file = name_file[:-4]

        code = "digraph {\nrankdir = LR\ncompound=true\n"
        # waiting queues
        code += '\nnode [shape="record"]\n'
        txt_queue = (
            '{name_quoted:40s} [label="<f0> {name}|'
            + "|".join([f"<f{i}>" for i in range(1, 5)])
            + '"]\n'
        )

        for queue in self.queues:
            name_quoted = f'"{queue.name}"'
            code += txt_queue.format(name=queue.name, name_quoted=name_quoted)

        # works and links
        code += '\nnode [shape="ellipse"]\n'

        txt_work = '{:40s} [label="{}",color = "{}"]\n'

        for work in self.works:
            name_work = work.name
            color = "Black"
            if work.kind is not None:
                if "io" in work.kind:
                    color = "Green"
            code += txt_work.format(f'"{name_work}"', name_work, color)

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
