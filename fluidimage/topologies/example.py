"""Topology example (:mod:`fluidimage.topologies.example`)
==========================================================

This topology has two pythran cpu bounded tasks. It helps see executors behavior
with C functions.

.. autoclass:: TopologyExample
   :members:
   :private-members:

"""

import os

import numpy as np
import scipy.io

from transonic import boost

from . import TopologyBase
from ..util import imread

A = "uint8[:,:]"


@boost
def cpu1(array1: A, array2: A, nloops: int = 10):

    a = np.arange(10000000 // nloops)
    result = a
    for i in range(nloops):
        result += a ** 3 + a ** 2 + 2

    for i in range(nloops):
        array1 = array1 * array2
    return (array1, array1)


@boost
def cpu2(array1: A, array2: A, nloops: int = 10):

    a = np.arange(10000000 // nloops)
    result = a
    for i in range(nloops):
        result += a ** 3 + a ** 2 + 2

    for i in range(nloops):
        array1 = np.multiply(array1, array2)
    return array1


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

    @classmethod
    def create_default_params(cls):
        params = dict(
            path_input=None,
            path_dir_result=None,
            nloops=1,
            multiplicator_nb_images=1,
        )
        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):
        self.params = params

        path_input = params["path_input"]
        path_dir_result = params["path_dir_result"]
        nloops = params["nloops"]
        self.multiplicator_nb_images = params["multiplicator_nb_images"]

        def func1(arrays):
            key = arrays[0]
            if key == "Karman_02_00":
                raise ValueError("For testing")
            arr0, arr1 = cpu1(arrays[1], arrays[2], nloops)
            return key, arr0, arr1

        def func2(arrays):
            key = arrays[0]
            result = cpu2(arrays[1], arrays[2], nloops)
            return key, result

        self.path_input = path_input

        super().__init__(
            path_dir_result=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        if not self.path_dir_result.exists():
            self.path_dir_result.mkdir()

        self.img_counter = 0

        queue_names = self.add_queue("names images")
        queue_couples_names = self.add_queue("names couples")
        queue_arrays = self.add_queue("arrays")
        queue_couples_arrays = self.add_queue("couples arrays")
        queue_cpu1 = self.add_queue("queue cpu1")
        queue_cpu2 = self.add_queue("queue cpu2")

        self.add_work(
            "fill names",
            func_or_cls=self.fill_names,
            output_queue=queue_names,
            kind=("global", "one shot"),
        )
        self.add_work(
            "fill names couples",
            func_or_cls=self.fill_couples_names,
            input_queue=queue_names,
            output_queue=queue_couples_names,
            kind=("global", "one shot"),
        )

        self.add_work(
            "read array",
            func_or_cls=self.read_array,
            input_queue=queue_names,
            output_queue=queue_arrays,
            kind="io",
        )
        self.add_work(
            "fill couples arrays",
            func_or_cls=self.fill_couples_arrays,
            input_queue=(queue_couples_names, queue_arrays),
            output_queue=queue_couples_arrays,
            kind=("global"),
        )
        self.add_work(
            "cpu1",
            func_or_cls=func1,
            input_queue=queue_couples_arrays,
            output_queue=queue_cpu1,
            kind="cpu",
        )
        self.add_work(
            "cpu2",
            func_or_cls=func2,
            input_queue=queue_cpu1,
            output_queue=queue_cpu2,
            kind="cpu",
        )
        self.add_work(
            "save", func_or_cls=self.save, input_queue=queue_cpu2, kind="io"
        )

    def fill_names(self, input_queue, output_queue):
        for ind in range(self.multiplicator_nb_images):
            for name in sorted(os.listdir(self.path_input)):
                key = name.split(".bmp")[0] + f"_{ind:02}"
                output_queue[key] = name

    def fill_couples_names(self, input_queue, output_queue):
        for key, name in list(input_queue.items()):
            output_queue[key] = [key, (name, name)]

    def read_array(self, name):
        if name == "Karman_03.bmp":
            raise ValueError("For testing")

        image = imread(self.path_input / name)
        return image

    def fill_couples_arrays(self, input_queues, output_queue):
        queue_couples_names, queue_arrays = input_queues
        queue_couples_arrays = output_queue

        for key, array in list(queue_arrays.items()):
            if key not in queue_couples_names:
                continue
            queue_arrays.pop(key)
            queue_couples_names.pop(key)
            # propagating possible exceptions...
            if isinstance(array, Exception):
                queue_couples_arrays[key] = array
            else:
                queue_couples_arrays[key] = [key, array, array]

    def save(self, inputs):
        key = inputs[0]
        arr = inputs[1]
        name = key + ".h5"
        scipy.io.savemat(self.path_dir_result / name, mdict={"array": arr})
