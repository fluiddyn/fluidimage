"""Topology for surface tracking (:mod:`fluidimage.topologies.surface_tracking`)
================================================================================

.. autoclass:: TopologySurfaceTracking
   :members:
   :private-members:

.. todo::

   Make this code usable:

   - good unittests
   - tutorial with ipynb
   - example integrated in the documentation

"""
import json
import math
import numpy as np
from pathlib import Path

from fluidimage.topologies import prepare_path_dir_result
from fluidimage import ParamContainer, SerieOfArraysFromFiles
from fluidimage.util import logger, imread
from fluiddyn.io.image import imsave
from fluidimage.works.surface_tracking import WorkSurfaceTracking

from .base import TopologyBase


class TopologySurfaceTracking(TopologyBase):
    """Topology for SurfaceTracking.

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
 #       complete_surftrack_params_with_default(params)

        params._set_child(
            "film",
            attribs={
                "fileName": "",
                "path": "",
                "path_ref": "",
                "ind_start": 0,
                "ind_stop": None,
                "ind_step": 1,
            },
        )

        WorkSurfaceTracking._complete_params_with_default(params)

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
                    "module": "fluidimage.topologies.surface_tracking",
                    "class": "TopologySurfaceTracking",
                }
            ),
        )

        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):

        self.params = params

        if params.surface_tracking is None:
            raise ValueError("params.surface_tracking has to be set.")

        self.serie = SerieOfArraysFromFiles(
            params.images.path, params.images.str_slice
        )

        path_dir = self.serie.path_dir
        path_dir_result, self.how_saving = prepare_path_dir_result(
            path_dir, params.saving.path, params.saving.postfix, params.saving.how
        )

        self.path_dir_result = path_dir_result
        self.path_dir_src = Path(path_dir)

        self.surface_tracking_work = WorkSurfaceTracking(params)

        super().__init__(
            path_dir_result=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        self.queue_paths = self.add_queue("paths")
        self.queue_arrays = self.add_queue("arrays")
        self.queue_angles = self.add_queue("angles")
        self.queuemod_angles = self.add_queue("angles mod")
        self.queue_heights = self.add_queue("heights")

        self.init_correctphase_function()

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

        self.add_work(
            "process_frame",
            self.processframe_func,
            input_queue=self.queue_arrays,
            output_queue=self.queue_angles,
        )

        self.add_work(
            "correct_phase",
            self.correctphase_func,
            input_queue=self.queue_angles,
            output_queue=self.queuemod_angles,
            kind="global"
        )

        self.add_work(
            "calcul_height",
            self.calculheight_func,
            input_queue=self.queuemod_angles,
            output_queue=self.queue_heights,
        )

        self.add_work(
            "save", self.save_image, input_queue=self.queue_heights, kind="io"
        )

    def processframe_func(self, input_queue):
        return (self.surface_tracking_work.process_frame1(input_queue[0]),
                input_queue[1])

    def calculheight_func(self, input_queue):
        return (self.surface_tracking_work.calculheight(input_queue[0]),
                input_queue[1])

    def init_correctphase_function(self):
        self.angleid = 0
        self.tmp = None

    def correctphase_func(self, input_queue, output_queue):
        print("in correctphase_function")
        fix_y = int(np.fix(self.surface_tracking_work.l_y / 2 / self.surface_tracking_work.red_factor))
        fix_x = int(np.fix(self.surface_tracking_work.l_x / 2 / self.surface_tracking_work.red_factor))
        for key in input_queue:
            (angle, path_angle,) = input_queue.pop(key)
            correct_angle = angle
            if self.tmp is None:
                self.tmp = angle
            jump = angle[fix_y, fix_x] - self.tmp[fix_y, fix_x]
            while abs(jump) > math.pi:
                correct_angle = angle - np.sign(jump) * 2 * math.pi
                jump = correct_angle[fix_y, fix_x] - self.tmp[fix_y, fix_x]
            self.tmp = angle
            output_queue[key] = (correct_angle, path_angle,)
            self.angleid = self.angleid + 1

    def imread(self, path):
        array = imread(path)
        return (array, path)

    def save_image(self, tuple_image_path):
        image, path = tuple_image_path
        name_file = Path(path).name
        path_out = self.path_dir_result / name_file
        imsave(path_out, image)

    def fill_queue_paths(self, input_queue, output_queue):

        assert input_queue is None

        serie = self.serie
        if len(serie) == 0:
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

        if len(names) == 0:
            if self.how_saving == "complete":
                logger.warning(
                    'topology in mode "complete" and work already done.'
                )
            else:
                logger.warning("Nothing to do")
            return

        nb_names = len(names)
        logger.info(f"Add {nb_names} images to compute.")
        logger.info("First files to process: " + str(names[:4]))

        logger.debug("All files: " + str(names))
