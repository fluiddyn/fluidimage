"""Topology for image2image preprocessing
=========================================

.. autoclass:: TopologyImage2Image
   :members:
   :private-members:

"""

import sys
from pathlib import Path

from fluiddyn.io.image import imsave
from fluidimage import ParamContainer
from fluidimage.topologies import prepare_path_dir_result
from fluidimage.topologies.splitters import SplitterFromImages
from fluidimage.util import imread
from fluidimage.works.image2image import WorkImage2Image

from .base import TopologyBaseFromImages


class TopologyImage2Image(TopologyBaseFromImages):
    """Topology for images processing with a user-defined function

    The most useful methods for the user (in particular :func:`compute`) are
    defined in the base class :class:`fluidimage.topologies.base.TopologyBase`.

    Parameters
    ----------

    params : None

      A ParamContainer (created with the class method
      :func:`create_default_params`) containing the parameters for the
      computation.

    logging_level : str, {'warning', 'info', 'debug', ...}

      Logging level.

    nb_max_workers : None, int

      Maximum numbers of "workers". If None, a number is computed from the
      number of cores detected. If there are memory errors, you can try to
      decrease the number of workers.

    """

    _short_name = "im2im"
    Splitter = SplitterFromImages

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters.

        Typical usage::

           params = TopologyImage2Image.create_default_params()
           # modify parameters here
           ...

           topo = TopologyImage2Image(params)

        """
        params = ParamContainer(tag="params")

        super()._add_default_params_saving(params)
        WorkImage2Image._complete_params_with_default(params)

        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):
        self.params = params

        if params.im2im is None:
            raise ValueError("params.im2im has to be set.")

        self.work = WorkImage2Image(params)
        self.serie = self.work.serie
        im2im_func = self.work.im2im_func

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

        self.queue_paths = self.add_queue("paths")
        self.queue_arrays = self.add_queue("arrays")
        self.queue_results = self.add_queue("results")

        self.add_work(
            "fill_path",
            self.fill_queue_paths,
            output_queue=self.queue_paths,
            kind="one shot",
        )

        self.add_work(
            "read_array",
            imread,
            input_queue=self.queue_paths,
            output_queue=self.queue_arrays,
            kind="io",
        )

        self.add_work(
            "im2im",
            im2im_func,
            input_queue=self.queue_arrays,
            output_queue=self.queue_results,
            kind="eat key value",
        )

        self.add_work(
            "save",
            self.save_image,
            input_queue=self.queue_results,
            kind=("io", "eat key value"),
        )
        self.results = []

    def save_image(self, tuple_path_image):
        """Save an image"""
        path, image = tuple_path_image
        name_file = Path(path).name
        path_out = self.path_dir_result / name_file
        imsave(path_out, image)
        self.results.append(name_file)


Topology = TopologyImage2Image


if "sphinx" in sys.modules:
    _params = TopologyImage2Image.create_default_params()

    __doc__ += _params._get_formatted_docs()
