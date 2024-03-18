"""Topology for surface tracking(:mod:`fluidimage.topologies.surface_tracking`)
===============================================================================

.. autoclass:: TopologySurfaceTracking
   :members:
   :private-members:

.. todo::

   Make this code usable:

   - good unittests
   - tutorial with ipynb
   - example integrated in the documentation

"""

import sys
from pathlib import Path

from fluiddyn.io.image import imsave_h5
from fluidimage import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays
from fluidimage.topologies import TopologyBase, prepare_path_dir_result
from fluidimage.util import imread, logger
from fluidimage.works import image2image
from fluidimage.works.surface_tracking import WorkSurfaceTracking


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

    _short_name = "surf"

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters."""
        params = ParamContainer(tag="params")

        WorkSurfaceTracking._complete_params_with_default(params)

        params._set_child(
            "images",
            attribs={
                "path": "",
                "path_ref": "",
                "str_subset_ref": None,
                "str_subset": None,
            },
        )

        params.images._set_doc(
            """
Parameters indicating the input image set.

path : str, {''}

    String indicating the input images (can be a full path towards an image
    file or a string given to `glob`).

path_ref : str, {''}

    String indicating the reference input images (can be a full path towards
    an image file or a string given to `glob`).

str_subset_ref : None

    String indicating as a Python slicing how to select reference images
    from the serie of reference images on the disk (in order to compute
    k_x value necessary for gain and filter).
    If None, no selection so all images are going to be processed.

str_subset : None

    String indicating as a Python slicing how to select images from the
    serie of images on the disk. If None, no selection so all images
    are going to be processed.

"""
        )

        super()._add_default_params_saving(params)

        params._set_child("preproc")
        image2image.complete_im2im_params_with_default(params.preproc)

        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):
        self.params = params

        if params.surface_tracking is None:
            raise ValueError("params.surface_tracking has to be set.")

        self.serie = SerieOfArraysFromFiles(
            params.images.path, params.images.str_subset
        )
        self.series = SeriesOfArrays(
            params.images.path,
            "i:i+"
            + str(self.serie.get_slicing_tuples()[0][2] + 1)
            + ":"
            + str(self.serie.get_slicing_tuples()[0][2]),
            ind_start=self.serie.get_slicing_tuples()[0][0],
            ind_stop=self.serie.get_slicing_tuples()[0][1] - 1,
            ind_step=self.serie.get_slicing_tuples()[0][2],
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

        queue_paths = self.add_queue("paths")
        queue_couples_of_names = self.add_queue("couples of names")
        queue_arrays = queue_arrays1 = self.add_queue("arrays")
        queue_angles = self.add_queue("angles")
        queue_couples_of_arrays = self.add_queue(
            "couples of corrected angles and angles"
        )
        queue_mod0_angles = self.add_queue("corrected angles copy")
        queue_mod_angles = self.add_queue("corrected angles")
        queue_heights_and_shapes = self.add_queue("heights and shapes")
        queue_heights = self.add_queue("heights")

        if params.preproc.im2im is not None:
            queue_arrays1 = self.add_queue("arrays1")

        self.add_work(
            "fill_path",
            self.fill_queue_paths,
            output_queue=(queue_paths, queue_couples_of_names),
            kind="one shot",
        )

        self.add_work(
            "read_array",
            imread,
            input_queue=queue_paths,
            output_queue=queue_arrays,
            kind="io",
        )

        if params.preproc.im2im is not None:
            im2im_func = image2image.get_im2im_function_from_params(
                params.preproc
            )

            self.add_work(
                "image2image",
                func_or_cls=im2im_func,
                input_queue=queue_arrays,
                output_queue=queue_arrays1,
                kind="eat key value",
            )

        self.add_work(
            "process_frame",
            self.surface_tracking_work.process_frame_func,
            input_queue=queue_arrays1,
            output_queue=queue_angles,
        )

        self.add_work(
            "create_couple",
            self.make_couples,
            input_queue=(queue_mod0_angles, queue_angles, queue_couples_of_names),
            output_queue=(queue_mod_angles, queue_couples_of_arrays),
            kind="global",
        )

        self.add_work(
            "correct_couple_of_phases",
            self.surface_tracking_work.correctcouple,
            input_queue=queue_couples_of_arrays,
            output_queue=queue_mod0_angles,
        )

        self.add_work(
            "calcul_height",
            self.surface_tracking_work.calculheight_func,
            input_queue=queue_mod_angles,
            output_queue=queue_heights_and_shapes,
        )

        self.add_work(
            "set_borders_zero",
            self.surface_tracking_work.set_borders_zero_func,
            input_queue=queue_heights_and_shapes,
            output_queue=queue_heights,
        )

        self.add_work(
            "save",
            self.save_image,
            input_queue=queue_heights,
            kind=("io", "eat key value"),
        )

    def make_couples(self, input_queues, output_queues):
        """correctphase"""
        queue_couples_of_names = input_queues[2]
        queue_angles = input_queues[1]
        queue_mod0_angles = input_queues[0]
        queue_couple = output_queues[1]
        if not (queue_couples_of_names):
            for key in tuple(queue_mod0_angles):
                output_queues[0][key] = queue_mod0_angles[key]
                del queue_mod0_angles[key]
        if not (queue_angles):
            print("no queue")
            return
        for key, couple in tuple(queue_couples_of_names.items()):
            # if correspondant arrays are available, make an array couple
            if couple[0] is couple[1]:
                if couple[0] in queue_angles.keys():
                    array1 = queue_angles[couple[0]]
                    array2 = queue_angles[couple[0]]

                    queue_couple[couple[0]] = (array1, array2)
                    del queue_angles[couple[0]]
                    del queue_couples_of_names[key]
            elif (
                couple[0] in queue_mod0_angles.keys()
                and couple[1] in queue_angles.keys()
            ):
                array1 = queue_mod0_angles[couple[0]]
                array2 = queue_angles[couple[1]]

                queue_couple[couple[1]] = (array1, array2)
                del queue_angles[couple[1]]
                del queue_couples_of_names[key]
                output_queues[0][couple[0]] = queue_mod0_angles[couple[0]]
                del queue_mod0_angles[couple[0]]

    def save_image(self, tuple_path_image):
        path, image = tuple_path_image
        name_file = Path(path).name
        path_out = self.path_dir_result / name_file
        imsave_h5(path_out, image, splitext=False)

    def fill_queue_paths(self, input_queue, output_queues):
        assert input_queue is None
        queue_paths = output_queues[0]
        queue_couples_of_names = output_queues[1]

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
                    queue_paths[name] = path_im_input
            else:
                queue_paths[name] = path_im_input

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

        logger.debug("All files: %s", names)

        series = self.series
        if not series:
            logger.warning("add 0 couple. No phase to correct.")
            return

        nb_series = len(series)
        logger.info(f"Add {nb_series} phase to correct.")

        for iserie, serie in enumerate(series):
            if iserie > 1:
                break
            logger.info(
                "Files of serie {}: {}".format(iserie, serie.get_name_arrays())
            )
        # for the first corrected angle : corrected_angle = angle
        ind_serie, serie = next(series.items())
        name = serie.get_name_arrays()[0]
        queue_couples_of_names[ind_serie - 1] = (name, name)
        for ind_serie, serie in series.items():
            queue_couples_of_names[ind_serie] = serie.get_name_arrays()


Topology = TopologySurfaceTracking

if "sphinx" in sys.modules:
    params = TopologySurfaceTracking.create_default_params()

    __doc__ += params._get_formatted_docs()
