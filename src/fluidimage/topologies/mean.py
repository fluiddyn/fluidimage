"""Mean images topology

.. autoclass:: TopologyMeanImage
   :members:
   :private-members:

"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

import fluidimage
from fluiddyn.io.image import imsave
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage import ParamContainer
from fluidimage.topologies.base import TopologyBaseFromImages
from fluidimage.topologies.splitters import SplitterFromImages
from fluidimage.util import imread
from fluidimage.works import BaseWorkFromImage

# from transonic import boost, Array


# A2d = Array[np.uint32, "2d", "C"]


# x2 speedup of this operation but this is clearly not the bottleneck yet...
# @boost
# def sum_4_2darrays(arr0: A2d, arr1: A2d, arr2: A2d, arr3: A2d):
#     """Sum 4 2d arrays"""
#     n0, n1 = arr0.shape
#     for i0 in range(n0):
#         for i1 in range(n1):
#             arr0[i0, i1] = (
#                 arr0[i0, i1] + arr1[i0, i1] + arr2[i0, i1] + arr3[i0, i1]
#             )
#     return arr0


class TopologyMeanImage(TopologyBaseFromImages):
    """Compute in parallel the mean image."""

    _short_name = "mean"
    Splitter = SplitterFromImages
    result: np.ndarray
    path_result: Path

    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag="params")
        super()._add_default_params_saving(params)
        BaseWorkFromImage._complete_params_with_default(params)
        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):

        p_images = params.images
        self.serie = SerieOfArraysFromFiles(p_images.path, p_images.str_subset)

        super().__init__(
            params=params,
            path_dir_src=self.serie.path_dir,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        queue_paths = self.add_queue("paths")
        queue_arrays = self.add_queue("arrays")
        queue_tmp_arrays = self.add_queue("tmp_arrays", "list")

        self.add_work(
            "fill_path",
            self.fill_queue_paths,
            output_queue=queue_paths,
            kind="one shot",
        )

        self.add_work(
            "read_array",
            imread,
            input_queue=queue_paths,
            output_queue=queue_arrays,
            kind="io",
        )

        self.add_work(
            "main",
            self.main,
            input_queue=(queue_paths, queue_arrays, queue_tmp_arrays),
            kind="global",
        )

        self.results = []

    def reduce_queue_tmp_arrays4(self, queue_tmp_arrays):
        while len(queue_tmp_arrays) >= 4:
            arr0, n0 = queue_tmp_arrays.pop()
            arr1, n1 = queue_tmp_arrays.pop()
            arr2, n2 = queue_tmp_arrays.pop()
            arr3, n3 = queue_tmp_arrays.pop()
            arr_sum = arr0 + arr1 + arr2 + arr3
            # arr_sum = sum_4_2darrays(arr0, arr1, arr2, arr3)
            n_sum = n0 + n1 + n2 + n3
            # print("reduce_queue_tmp_arrays4", n_sum)
            queue_tmp_arrays.insert(0, (arr_sum, n_sum))
            self.results.extend([n_sum] * 3)

    def reduce_queue_tmp_arrays2(self, queue_tmp_arrays):
        while len(queue_tmp_arrays) >= 2:
            arr0, n0 = queue_tmp_arrays.pop()
            arr1, n1 = queue_tmp_arrays.pop()
            arr_sum = arr0 + arr1
            n_sum = n0 + n1
            # print("reduce_queue_tmp_arrays2", n_sum)
            queue_tmp_arrays.insert(0, (arr_sum, n_sum))
            self.results.extend([n_sum])

    def main(self, input_queues, output_queue):
        del output_queue
        queue_paths, queue_arrays, queue_tmp_arrays = input_queues
        assert isinstance(queue_tmp_arrays, list)

        while queue_arrays:
            name, arr = queue_arrays.pop_first_item()
            queue_tmp_arrays.append((arr.astype(np.uint32), 1))

        self.reduce_queue_tmp_arrays4(queue_tmp_arrays)

        if (
            not queue_paths
            and not queue_arrays
            and (
                not hasattr(self.executor, "nb_working_workers_io")
                or self.executor.nb_working_workers_io == 0
            )
        ):
            self.reduce_queue_tmp_arrays2(queue_tmp_arrays)
            if not queue_tmp_arrays:
                return
            assert len(queue_tmp_arrays) == 1, queue_tmp_arrays
            arr_result, n_result = queue_tmp_arrays.pop()

            executor = self.executor

            try:
                index_process = executor.index_process
            except AttributeError:
                index_process = 0

            path = executor.path_job_data / f"tmp_sum{index_process:03d}.h5"
            with h5py.File(path, "w") as file:
                file.create_dataset("arr", data=arr_result)
                file.attrs["num_images"] = n_result

    def final_seq_work(self):
        path_tmp_files = sorted(self.executor.path_job_data.glob("tmp_sum*.h5"))

        queue_tmp_arrays = []
        for path_tmp_file in path_tmp_files:
            with h5py.File(path_tmp_file, "r") as file:
                arr = file["arr"][...]
                num_images = file.attrs["num_images"]
                queue_tmp_arrays.append((arr, num_images))

        self.reduce_queue_tmp_arrays4(queue_tmp_arrays)
        self.reduce_queue_tmp_arrays2(queue_tmp_arrays)

        assert len(queue_tmp_arrays) == 1
        arr, num_images = queue_tmp_arrays[0]
        self.result = (arr / num_images).astype(np.uint8)
        self.results.append(num_images)

        self.path_result = self.path_dir_result.with_name(
            self.path_dir_result.name + ".png"
        )
        imsave(self.path_result, self.result)


def parse_args():
    """Parse the arguments of the command line"""

    parser = argparse.ArgumentParser(
        description=TopologyMeanImage.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="Path file or directory.",
        type=str,
        nargs="?",
        default=os.getcwd(),
    )
    parser.add_argument("-v", "--verbose", help="verbose mode", action="count")
    parser.add_argument(
        "-V",
        "--version",
        help="Print fluidimage version and exit",
        action="count",
    )

    parser.add_argument(
        "--executor",
        help="Name of the executor.",
        type=str,
        default="exec_sequential",
    )

    parser.add_argument(
        "-np",
        "--nb-max-workers",
        help="Maximum number of workers/processes.",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--subset",
        help="Subset of images.",
        type=str,
        default=None,
    )

    return parser.parse_args()


Topology = TopologyMeanImage


def main():
    """Main function for fluidimage-mean"""
    args = parse_args()

    if args.version:
        print(f"fluidimage {fluidimage.__version__}")
        return

    params = Topology.create_default_params()
    params.images.path = str(args.path)
    params.images.str_subset = args.subset
    topology = Topology(params)
    topology.compute(args.executor, nb_max_workers=args.nb_max_workers)
