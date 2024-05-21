"""Mean images topology

.. autoclass:: TopologyMeanImage
   :members:
   :private-members:

"""

from pathlib import Path

from fluiddyn.io.image import imsave
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage import ParamContainer
from fluidimage.topologies import prepare_path_dir_result
from fluidimage.topologies.base import TopologyBaseFromImages
from fluidimage.topologies.splitters import SplitterFromImages
from fluidimage.util import imread
from fluidimage.works import BaseWorkFromImage


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

    def main(self, input_queues, output_queue):
        del output_queue
        queue_paths, queue_arrays, queue_tmp_arrays = input_queues
        assert isinstance(queue_tmp_arrays, list)

        while queue_arrays:
            name, arr = queue_arrays.pop_first_item()
            print(name)

        # check arrays queue

        # fill tmp_arrays with (arr, num_arrays_used)

        # get 4 arrays (or less if queue_paths is empty)

        # compute the mean over the 4 arrays

        # insert the sum and remove the arrays

        # when done, save tmp_mean
