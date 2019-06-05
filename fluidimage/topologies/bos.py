"""Topology for BOS computation (:mod:`fluidimage.topologies.bos`)
==================================================================

.. autoclass:: TopologyBOS
   :members:
   :private-members:

"""
import json
import os
import sys
from pathlib import Path

from fluidimage import ParamContainer, SerieOfArraysFromFiles
from fluidimage.data_objects.piv import ArrayCoupleBOS
from fluidimage.topologies import TopologyBase, prepare_path_dir_result
from fluidimage.util import imread, logger
from fluidimage.works.piv import WorkPIV

from . import image2image


class TopologyBOS(TopologyBase):
    """Topology for BOS computation.

    See https://en.wikipedia.org/wiki/Background-oriented_schlieren_technique

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

      Maximum numbers of "workers". If None, a number is computed from the number of
      cores detected. If there are memory errors, you can try to decrease the number
      of workers.

    """

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters.

        Typical usage::

          params = TopologyPIV.create_default_params()
          # modify parameters here
          ...

          topo = TopologyPIV(params)

        """
        params = ParamContainer(tag="params")

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

        params._set_attrib("reference", 0)

        params._set_doc(
            """
reference : str or int, {0}

    Reference file (from which the displacements will be computed). Can be an
    absolute file path, a file name or the index in the list of files found
    from the parameters in ``params.images``.

"""
        )

        params._set_child(
            "saving", attribs={"path": None, "how": "ask", "postfix": "bos"}
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

        WorkPIV._complete_params_with_default(params)

        params._set_internal_attr(
            "_value_text",
            json.dumps(
                {
                    "program": "fluidimage",
                    "module": "fluidimage.topologies.bos",
                    "class": "TopologyBOS",
                }
            ),
        )

        params._set_child("preproc")
        image2image.complete_im2im_params_with_default(params.preproc)

        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):

        self.params = params

        self.serie = SerieOfArraysFromFiles(
            params.images.path, params.images.str_slice
        )

        path_dir = Path(self.serie.path_dir)
        path_dir_result, self.how_saving = prepare_path_dir_result(
            path_dir, params.saving.path, params.saving.postfix, params.saving.how
        )

        self.path_dir_result = path_dir_result
        self.path_dir_src = Path(path_dir)

        if not isinstance(params.reference, int):
            reference = Path(params.reference).expanduser()
        else:
            reference = params.reference

        if isinstance(reference, int):
            names = self.serie.get_name_arrays()
            names = sorted(names)
            path_reference = path_dir / names[reference]

        else:
            reference = Path(reference)
            if reference.is_file():
                path_reference = reference
            else:
                path_reference = path_dir_result / reference
                if not path_reference.is_file():
                    raise ValueError(
                        "Bad value of params.reference:" + path_reference
                    )

        self.name_reference = path_reference.name
        self.path_reference = path_reference
        self.image_reference = imread(path_reference)

        super().__init__(
            path_dir_result=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        queue_paths = self.add_queue("paths")
        queue_arrays = queue_arrays1 = self.add_queue("arrays")
        queue_bos = self.add_queue("bos")

        if params.preproc.im2im is not None:
            queue_arrays1 = self.add_queue("arrays1")

        self.add_work(
            "fill paths",
            func_or_cls=self.fill_queue_paths,
            output_queue=queue_paths,
            kind=("global", "one shot"),
        )

        def _imread(path):
            image = imread(path)
            return image, Path(path).name

        self.add_work(
            "read array",
            func_or_cls=_imread,
            input_queue=queue_paths,
            output_queue=queue_arrays,
            kind="io",
        )

        if params.preproc.im2im is not None:
            im2im_func = image2image.TopologyImage2Image.init_im2im(
                self, params.preproc
            )

            self.add_work(
                "image2image",
                func_or_cls=im2im_func,
                input_queue=queue_arrays,
                output_queue=queue_arrays1,
            )

        self.add_work(
            "compute bos",
            func_or_cls=self.calcul,
            params_cls=params,
            input_queue=queue_arrays,
            output_queue=queue_bos,
        )

        self.add_work(
            "save bos",
            func_or_cls=self.save_bos_object,
            input_queue=queue_bos,
            kind="io",
        )

    def save_bos_object(self, obj):
        """Save a BOS object"""
        ret = obj.save(self.path_dir_result, kind="bos")
        return ret

    def calcul(self, tuple_image_path):
        """Compute a BOS field"""
        image, name = tuple_image_path
        array_couple = ArrayCoupleBOS(
            names=(self.name_reference, name),
            arrays=(self.image_reference, image),
            params_mask=self.params.mask,
            serie=self.serie,
            paths=[self.path_reference, self.path_dir_src / name],
        )
        return WorkPIV(self.params).calcul(array_couple)

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

        try:
            del output_queue[self.path_reference]
        except KeyError:
            pass

        nb_names = len(names)
        logger.info(f"Add {nb_names} images to compute.")
        logger.info(f"First files to process: {names[:4]}")

        logger.debug(f"All files: {names}")

    def make_text_at_exit(self, time_since_start):
        """Make a text printed at exit"""
        txt = "Stop compute after t = {time_since_start:.2f} s"
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += f" ({nb_results} bos fields, {time_since_start / nb_results:.2f} s/field)."
        else:
            txt += "."

        txt += f"\npath results:\n{self.path_dir_result}"

        return txt


if "sphinx" in sys.modules:
    params = TopologyBOS.create_default_params()

    __doc__ += params._get_formatted_docs()

if __name__ == "__main__":
    params = TopologyBOS.create_default_params()
    params.series.path = "../../../image_samples/Karman/Images"
    params.series.ind_start = 1
    params.series.ind_step = 2

    params.piv0.shape_crop_im0 = 32
    params.multipass.number = 2
    params.multipass.use_tps = False

    params.mask.strcrop = ":, 50:500"

    # params.saving.how = 'complete'
    params.saving.postfix = "bos_example2"

    topo = TopologyBOS(params, logging_level="info")

    topo.make_code_graphviz("tmp.dot")
