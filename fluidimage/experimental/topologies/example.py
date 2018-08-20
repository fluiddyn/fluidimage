"""Topology example for testing (:mod:`fluidimage.experimental.topologies.example`)
===================================================================================

This topology has two pythran cpu bounded tasks. It helps see executors behavior with C fonctions.

.. autoclass:: TopologyExample
   :members:
   :private-members:

"""
import os
import sys
import time
import numpy as np
import scipy.io

from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile
from fluidimage import config_logging

from fluidimage.experimental.cpu_bounded_task_examples_pythran import cpu1, cpu2
from .base import TopologyBase
from ...util.util import logger, imread

_stdout_at_import = sys.stdout
_stderr_at_import = sys.stderr


class TopologyExample(TopologyBase):
    """Topology example for testing.

    Parameters
    ----------

    params : None

      A ParamContainer containing the parameters for the computation.

    logging_level : str, {'warning', 'info', 'debug', ...}

      Logging level.

    nb_max_workers : None, int

      Maximum numbers of "workers". If None, a number is computed from the
      number of cores detected. If there are memory errors, you can try to
      decrease the number of workers.

    """

    def __init__(
        self,
        path_dir=None,
        path_output=None,
        logging_level="info",
        nb_max_workers=None,
    ):

        super().__init__(
            logging_level=logging_level, nb_max_workers=nb_max_workers
        )

        if path_dir is None:
            self.path_dir = "../../../image_samples/Karman/Images2"
        else:
            self.path_dir = path_dir

        if path_output is not None:
            if not os.path.exists(path_output):
                os.makedirs(path_output)
            self.path_output = path_output
        log = os.path.join(
            path_output, "log_" + time_as_str() + "_" + str(os.getpid()) + ".txt"
        )

        stdout = sys.stdout
        if isinstance(stdout, MultiFile):
            stdout = _stdout_at_import

        stderr = sys.stderr
        if isinstance(stderr, MultiFile):
            stderr = _stderr_at_import

        self._log_file = open(log, "w")
        sys.stdout = MultiFile([stdout, self._log_file])
        sys.stderr = MultiFile([stderr, self._log_file])

        if logging_level is not None:
            for handler in logger.handlers:
                logger.removeHandler(handler)

        config_logging(logging_level, file=sys.stdout)

        if hasattr(self, "path_output"):
            logger.info("path results:\n" + self.path_output)

        self.img_counter = 0

        queue_names_img1 = self.add_queue("names img 1")
        queue_names_img2 = self.add_queue("names img 2")
        queue_array_couple = self.add_queue("array couples")
        queue_cpu1 = self.add_queue("queue_cpu1")
        queue_cpu2 = self.add_queue("queue_cpu2")

        self.add_work(
            "fill names",
            func_or_cls=self.fill_names,
            output_queue=(queue_names_img1, queue_names_img2),
            kind=("global", "one shot"),
        )
        self.add_work(
            "make couple",
            func_or_cls=self.make_couple,
            input_queue=(queue_names_img1, queue_names_img2),
            output_queue=queue_array_couple,
            kind=("global", "io"),
        )
        self.add_work(
            "cpu1",
            func_or_cls=self.cpu1,
            input_queue=queue_array_couple,
            output_queue=queue_cpu1,
            kind="server",
        )

        self.add_work(
            "cpu2",
            func_or_cls=self.cpu2,
            params_cls=None,
            input_queue=queue_cpu1,
            output_queue=queue_cpu2,
            kind="server",
        )

        self.add_work(
            "save", func_or_cls=self.save, params_cls=None, input_queue=queue_cpu2
        )

    def fill_names(self, input_queue, output_queues):

        list_dir = os.listdir(self.path_dir)
        for dir in list_dir:
            output_queues[0].queue[dir] = dir
            output_queues[1].queue[dir] = dir
        return

    def make_couple(self, input_queues, output_queue):

        if not input_queues[0].queue or not input_queues[1].queue:
            return False
        key1, obj1 = input_queues[0].queue.popitem()
        start = time.time()
        key2, obj2 = input_queues[1].queue.popitem()
        print(self.path_dir + str(obj1))
        img1 = np.array(imread(self.path_dir + "/" + str(obj1)))
        img2 = np.array(imread(self.path_dir + "/" + str(obj2)))
        output_queue.queue[str(key1 + "" + key2)] = [img1, img2]
        return True

    def save(self, array):
        self.img_counter += 1
        scipy.io.savemat(
            self.path_dir + "/../test/array_" + str(self.img_counter),
            mdict={"array": array},
        )
        print("SAVED !!")

    def cpu1(self, arrays):
        return cpu1(arrays[0], arrays[1])

    def cpu2(self, arrays):
        return cpu2(arrays[0], arrays[1])

    def _print_at_exit(self, time_since_start):

        txt = "Stop compute after t = {:.2f} s".format(time_since_start)
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += " ({} piv fields, {:.2f} s/field).".format(
                nb_results, time_since_start / nb_results
            )
        else:
            txt += "."

        txt += "\npath results:\n" + self.path_dir_result

        print(txt)


if __name__ == "__main__":
    topo = TopologyExample(logging_level="info")
    # topo.make_code_graphviz("tmp.dot")
    topo.compute()
