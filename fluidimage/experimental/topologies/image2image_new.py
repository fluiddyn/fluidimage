"""Topology for image processing (:mod:`fluidimage.topologies.image2image`)
===========================================================================

.. autoclass:: TopologyImage2Image
   :members:
   :private-members:

"""
import os
import json


from fluidimage.topologies import prepare_path_dir_result
from .base import TopologyBase

from fluidimage import ParamContainer, SeriesOfArrays
from fluidimage.util.util import logger, imread

from fluidimage.preproc.image2image import (
    complete_im2im_params_with_default,
    init_im2im_function,
)

from fluiddyn.io.image import imsave


class TopologyImage2Image(TopologyBase):
    """Topology for images processing with a user-defined function

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

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters.

        For developers: cf. fluidsim.base.params

        """
        params = ParamContainer(tag="params")
        complete_im2im_params_with_default(params)

        params._set_child(
            "series",
            attribs={
                "path": "",
                "strslice": None,
                "ind_start": 0,
                "ind_stop": None,
                "ind_step": 1,
            },
        )

        params.series._set_doc(
            """
Parameters indicating the input series of images.

path : str, {''}

    String indicating the input images (can be a full path towards an image
    file or a string given to `glob`).

strslice : None

    String indicating as a Python slicing how series of images are formed.
    See the parameters the PIV topology.

ind_start : int, {0}

ind_step : int, {1}

int_stop : None

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

    def __init__(self, params=None, logging_level="info", nb_max_workers=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params

        if params.im2im is None:
            raise ValueError("params.im2im has to be set.")

        self.series = SeriesOfArrays(
            params.series.path,
            params.series.strslice,
            ind_start=params.series.ind_start,
            ind_stop=params.series.ind_stop,
            ind_step=params.series.ind_step,
        )

        path_dir = self.series.serie.path_dir
        path_dir_result, self.how_saving = prepare_path_dir_result(
            path_dir, params.saving.path, params.saving.postfix, params.saving.how
        )

        self.path_dir_result = path_dir_result
        self.path_dir_src = params.series.path

        super(TopologyImage2Image, self).__init__(
            path_output=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        self.queue_path = self.add_queue("queue_names")
        self.queue_array_path = self.add_queue("queue_array_path")
        self.queue_out = self.add_queue("queue_out")

        self.add_work(
            "fill_path",
            self.add_series,
            output_queue=self.queue_path,
            kind="one shot",
        )

        self.im2im_func = self.init_im2im(params)

        self.add_work(
            "get_array",
            self.imread,
            input_queue=self.queue_path,
            output_queue=self.queue_array_path,
        )

        self.add_work(
            "im2im",
            self.im2im_func,
            input_queue=self.queue_array_path,
            output_queue=self.queue_out,
        )

        self.add_work("save", self.save_image, input_queue=self.queue_out)

    def init_im2im(self, params_im2im):
        self.im2im_obj, self.im2im_func = init_im2im_function(
            im2im=params_im2im.im2im, args_init=params_im2im.args_init
        )
        return self.im2im_func

    def imread(self, path):
        array = imread(path)
        return (array, path)

    def save_image(self, tuple_image_path):
        image, path = tuple_image_path
        nfile = os.path.split(path)[-1]
        path_out = os.path.join(self.path_dir_result, nfile)
        imsave(path_out, image)

    def add_series(self, input_queue, output_queue):

        series = self.series
        if len(series) == 0:
            logger.warning("add 0 image. No image to process.")
            return

        names = series.get_name_all_arrays()

        for name in names:
            if self.how_saving == "complete":
                if not os.path.exists(os.path.join(self.path_dir_result, name)):
                    output_queue[name] = os.path.join(self.path_dir_src, name)
            else:
                output_queue.queue[name] = os.path.join(self.path_dir_src, name)

        if len(names) == 0:
            logger.warning('topology in mode "complete" and work already done.')
            return

        nb_names = len(names)
        print("Add {} images to compute.".format(nb_names))

        logger.debug(repr(names))

        print("First files to process:", names[:4])
