"""Topology for BOS computation (:mod:`fluidimage.topologies.bos`)
==================================================================

.. autoclass:: TopologyBOS
   :members:
   :private-members:

"""

import sys
from pathlib import Path

from fluidimage import ParamContainer
from fluidimage.data_objects.piv import get_name_bos
from fluidimage.topologies import prepare_path_dir_result
from fluidimage.topologies.base import TopologyBaseFromImages
from fluidimage.topologies.splitters import SplitterFromImages
from fluidimage.util import imread
from fluidimage.works import image2image
from fluidimage.works.bos import WorkBOS


class SplitterBos(SplitterFromImages):
    def __init__(self, params, num_processes, topology=None):
        super().__init__(params, num_processes, topology=topology)
        self.params.reference = topology.path_reference


def _imread(path):
    image = imread(path)
    return image, Path(path).name


class TopologyBOS(TopologyBaseFromImages):
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

    _short_name = "bos"
    Splitter = SplitterBos

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

        super()._add_default_params_saving(params)
        WorkBOS._complete_params_with_default(params)

        params._set_child("preproc")
        image2image.complete_im2im_params_with_default(params.preproc)

        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):
        self.params = params

        self.main_work = WorkBOS(params)
        self.serie = self.main_work.serie
        self.path_reference = self.main_work.path_reference

        path_dir = Path(self.serie.path_dir)
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

        self.add_work(
            "read array",
            func_or_cls=_imread,
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
        self.results = []

    def save_bos_object(self, obj):
        """Save a BOS object"""
        ret = obj.save(self.path_dir_result, kind="bos")
        self.results.append(ret)

    def calcul(self, tuple_image_path):
        """Compute a BOS field"""
        return self.main_work.calcul(tuple_image_path)

    def _get_name_result_from_name(self, name):
        return get_name_bos(name, self.serie)

    def _fix_indices_images(self, indices_images):
        """Fix the indices images in fill_queue_paths"""

        if not self.path_reference.is_relative_to(self.serie.path_dir):
            return

        if not self.path_reference.name.startswith(self.serie.base_name):
            return

        try:
            indices_ref = tuple(
                self.serie.compute_indices_from_name(self.path_reference.name)
            )
        except ValueError:
            return

        try:
            indices_images.remove(indices_ref)
        except ValueError:
            pass

    def make_text_at_exit(self, time_since_start):
        """Make a text printed at exit"""
        txt = f"Stop compute after t = {time_since_start:.2f} s"
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += f" ({nb_results} bos fields, {time_since_start / nb_results:.2f} s/field)."
        else:
            txt += "."

        txt += "\npath results:\n" + str(Path(self.path_dir_result).absolute())

        return txt


Topology = TopologyBOS


if "sphinx" in sys.modules:
    _params = TopologyBOS.create_default_params()
    __doc__ += _params._get_formatted_docs()
