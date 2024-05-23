"""Mean images topology

.. autoclass:: TopologyMeanImage
   :members:
   :private-members:

"""

from pathlib import Path

import h5py
import numpy as np

from fluiddyn.io.image import imsave
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage import ParamContainer
from fluidimage.topologies import prepare_path_dir_result
from fluidimage.topologies.base import TopologyBaseFromImages
from fluidimage.topologies.splitters import SplitterFromImages
from fluidimage.util import imread
from fluidimage.works import BaseWorkFromImage


def reduce_queue_tmp_arrays4(queue_tmp_arrays):
    while len(queue_tmp_arrays) >= 4:
        arr0, n0 = queue_tmp_arrays.pop()
        arr1, n1 = queue_tmp_arrays.pop()
        arr2, n2 = queue_tmp_arrays.pop()
        arr3, n3 = queue_tmp_arrays.pop()
        arr_sum = arr0 + arr1 + arr2 + arr3
        n_sum = n0 + n1 + n2 + n3
        # print("reduce_queue_tmp_arrays4", n_sum)
        queue_tmp_arrays.insert(0, (arr_sum, n_sum))


def reduce_queue_tmp_arrays2(queue_tmp_arrays):
    while len(queue_tmp_arrays) >= 2:
        arr0, n0 = queue_tmp_arrays.pop()
        arr1, n1 = queue_tmp_arrays.pop()
        arr_sum = arr0 + arr1
        n_sum = n0 + n1
        # print("reduce_queue_tmp_arrays2", n_sum)
        queue_tmp_arrays.insert(0, (arr_sum, n_sum))


class TopologyMeanImage(TopologyBaseFromImages):
    """Compute in parallel the mean image."""

    _short_name = "mean_image"
    Splitter = SplitterFromImages

    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag="params")
        super()._add_default_params_saving(params)
        BaseWorkFromImage._complete_params_with_default(params)
        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):
        self.params = params

        p_images = self.params.images
        self.serie = SerieOfArraysFromFiles(p_images.path, p_images.str_subset)

        path_dir = self.serie.path_dir
        path_dir_result, self.how_saving = prepare_path_dir_result(
            path_dir, params.saving.path, params.saving.postfix, params.saving.how
        )

        self.path_dir_result = path_dir_result
        self.path_dir_src = Path(path_dir)

        super().__init__(
            path_dir_result=path_dir_result,
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

    def main(self, input_queues, output_queue):
        del output_queue
        queue_paths, queue_arrays, queue_tmp_arrays = input_queues
        assert isinstance(queue_tmp_arrays, list)

        while queue_arrays:
            name, arr = queue_arrays.pop_first_item()
            print(name)
            queue_tmp_arrays.append((arr.astype(np.uint32), 1))

        reduce_queue_tmp_arrays4(queue_tmp_arrays)

        if (
            not queue_paths
            and not queue_arrays
            and self.executor.nb_working_workers_io == 0
        ):
            reduce_queue_tmp_arrays2(queue_tmp_arrays)
            if not queue_tmp_arrays:
                return
            assert len(queue_tmp_arrays) == 1, queue_tmp_arrays
            print(f"{queue_tmp_arrays = }")
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

    def finalize_compute_seq(self):
        path_tmp_files = sorted(self.executor.path_job_data.glob("tmp_sum*.h5"))

        queue_tmp_arrays = []
        for path_tmp_file in path_tmp_files:
            with h5py.File(path_tmp_file, "r") as file:
                arr = file["arr"][...]
                num_images = file.attrs["num_images"]
                queue_tmp_arrays.append((arr, num_images))

        reduce_queue_tmp_arrays4(queue_tmp_arrays)
        reduce_queue_tmp_arrays2(queue_tmp_arrays)

        assert len(queue_tmp_arrays) == 1
        arr, num_images = queue_tmp_arrays[0]
        self.result = arr / num_images
