"""Topology example for testing (:mod:`fluidimage.experimental.topologies.example`)
===================================================================================

This topology has two pythran cpu bounded tasks. It helps see executors behavior
with C functions.

.. autoclass:: TopologyExample
   :members:
   :private-members:

"""

import os

import numpy as np
import scipy.io


from fluidimage.experimental.cpu_bounded_task_examples_pythran import cpu1, cpu2

from .base import TopologyBase

from ...util.util import imread


class TopologyExample(TopologyBase):
    """Topology example for testing.

    Parameters
    ----------

    logging_level : str, {'warning', 'info', 'debug', ...}

      Logging level.

    nb_max_workers : None, int

      Maximum numbers of "workers". If None, a number is computed from the
      number of cores detected. If there are memory errors, you can try to
      decrease the number of workers.

    """

    def __init__(
        self, path_input=None, logging_level="info", nb_max_workers=None, nloops=1
    ):
        def func1(arrays):
            return cpu1(arrays[0], arrays[1], nloops)

        def func2(arrays):
            return cpu2(arrays[0], arrays[1], nloops)

        self.path_input = path_input

        self.path_dir_result = path_input.parent / "Images.example_new"

        super().__init__(
            path_output=self.path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        if not self.path_dir_result.exists():
            self.path_dir_result.mkdir()

        self.img_counter = 0

        queue_names_img0 = self.add_queue("names img 0")
        queue_names_img1 = self.add_queue("names img 1")
        queue_names_img2 = self.add_queue("names img 2")
        queue_array_couple = self.add_queue("array couples")
        queue_cpu1 = self.add_queue("queue_cpu1")
        queue_cpu2 = self.add_queue("queue_cpu2")

        self.add_work(
            "fill names0",
            func_or_cls=self.fill_names0,
            output_queue=(queue_names_img0),
            kind=("global", "one shot"),
        )

        self.add_work(
            "fill names",
            func_or_cls=self.fill_names,
            input_queue=queue_names_img0,
            output_queue=(queue_names_img1, queue_names_img2),
            kind=("global", "one shot"),
        )
        self.add_work(
            "make couple",
            func_or_cls=self.make_couple,
            input_queue=(queue_names_img1, queue_names_img2),
            output_queue=queue_array_couple,
            kind=("global", "one shot"),
        )
        self.add_work(
            "cpu1",
            func_or_cls=func1,
            input_queue=queue_array_couple,
            output_queue=queue_cpu1,
            kind="server",
        )

        self.add_work(
            "cpu2",
            func_or_cls=func2,
            params_cls=None,
            input_queue=queue_cpu1,
            output_queue=queue_cpu2,
            kind="server",
        )

        self.add_work(
            "save", func_or_cls=self.save, params_cls=None, input_queue=queue_cpu2
        )

    def fill_names0(self, input_queue, output_queue):
        for name in os.listdir(self.path_input):
            output_queue[name] = name

    def fill_names(self, input_queue, output_queues):
        for name in list(input_queue.keys()):
            input_queue.pop(name)
            output_queues[0][name] = name
            output_queues[1][name] = name

    def make_couple(self, input_queues, output_queue):

        if len(input_queues[0]) != len(input_queues[1]):
            raise ValueError()

        for _ in range(len(input_queues[0])):
            key1, obj1 = input_queues[0].popitem()
            key2, obj2 = input_queues[1].popitem()
            img1 = np.array(imread(self.path_input / obj1))
            img2 = np.array(imread(self.path_input / obj2))
            output_queue[str(key1 + "-" + key2)] = [img1, img2]

    def save(self, arr):
        self.img_counter += 1
        scipy.io.savemat(
            self.path_dir_result / f"array_{self.img_counter}",
            mdict={"array": arr},
        )

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

        txt += "\npath results:\n" + str(self.path_dir_result)

        print(txt)
