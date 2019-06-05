"""Topology for image2image preprocessing (:mod:`fluidimage.topologies.image2image`)
====================================================================================

.. autoclass:: TopologyImage2Image
   :members:
   :private-members:

"""

import json
import sys
from pathlib import Path

from fluiddyn.io.image import imsave
from fluidimage import ParamContainer, SerieOfArraysFromFiles
from fluidimage.preproc.image2image import (
    complete_im2im_params_with_default,
    init_im2im_function,
)
from fluidimage.topologies import prepare_path_dir_result
from fluidimage.util import imread, logger

from .base import TopologyBase


class TopologyImage2Image(TopologyBase):
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
        complete_im2im_params_with_default(params)

        params._set_child("images", attribs={"path": "", "str_slice": None})

        params.images._set_doc(
            """
Parameters indicating the input image set.

path : str, {''}

    String indicating the input images (can be a full path towards an image
    file or a string given to `glob`).

str_slice : None

    String indicating as a Python slicing how to select images from the serie of
    images on the disk. If None, no selection so all images are going to be
    processed.

"""
        )

        params._set_child(
            "saving", attribs={"path": None, "how": "ask", "postfix": "pre"}
        )

        params.saving._set_doc(
            """Saving of the results.

path : None or str

    Path of the directory where the data will be saved. If None, the path is
    obtained from the input path and the parameter `postfix`.

how : str {'ask'}

    'ask', 'new_dir', 'complete' or 'recompute'.

postfix : str

    Postfix from which the output file is computed.
"""
        )

        params._set_internal_attr(
            "_value_text",
            json.dumps(
                {
                    "program": "fluidimage",
                    "module": "fluidimage.topologies.image2image",
                    "class": "TopologyImage2Image",
                }
            ),
        )

        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):

        self.params = params

        if params.im2im is None:
            raise ValueError("params.im2im has to be set.")

        self.serie = SerieOfArraysFromFiles(
            params.images.path, params.images.str_slice
        )

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
            self.imread,
            input_queue=self.queue_paths,
            output_queue=self.queue_arrays,
            kind="io",
        )

        im2im_func = self.init_im2im(params)

        self.add_work(
            "im2im",
            im2im_func,
            input_queue=self.queue_arrays,
            output_queue=self.queue_results,
        )

        self.add_work(
            "save", self.save_image, input_queue=self.queue_results, kind="io"
        )

    def init_im2im(self, params_im2im):
        """Initialize the im2im function"""
        self.im2im_obj, self.im2im_func = init_im2im_function(
            im2im=params_im2im.im2im, args_init=params_im2im.args_init
        )
        return self.im2im_func

    def imread(self, path):
        """Read an image"""
        array = imread(path)
        return (array, path)

    def save_image(self, tuple_image_path):
        """Save an image"""
        image, path = tuple_image_path
        name_file = Path(path).name
        path_out = self.path_dir_result / name_file
        imsave(path_out, image)

    def fill_queue_paths(self, input_queue, output_queue):
        """Fill the first queue (paths)"""
        assert input_queue is None

        serie = self.serie
        if not serie:
            logger.warning("add 0 image. No image to process.")
            return

        names = serie.get_name_arrays()

        for name in names:
            path_im_output = self.path_dir_result / name
            path_im_input = str(self.path_dir_src / name)
            if self.how_saving == "complete":
                if not path_im_output.exists():
                    output_queue[name] = path_im_input
            else:
                output_queue[name] = path_im_input

        if not names:
            if self.how_saving == "complete":
                logger.warning(
                    'topology in mode "complete" and work already done.'
                )
            else:
                logger.warning("Nothing to do")
            return

        nb_names = len(names)

        logger.info(f"Add {nb_names} images to compute.")
        logger.info(f"First files to process: {names[:4]}")

        logger.debug(f"All files: {names}")


if "sphinx" in sys.modules:
    params = TopologyImage2Image.create_default_params()

    __doc__ += params._get_formatted_docs()
