"""Topology for BOS computation (:mod:`fluidimage.topologies.bos`)
==================================================================

NotImplementedError!

.. autoclass:: TopologyBOS
   :members:
   :private-members:

"""
import os
import json

from .. import ParamContainer, SeriesOfArrays

from .base import TopologyBase

from .waiting_queues.base import (
    WaitingQueueMultiprocessing,
    WaitingQueueThreading,
    WaitingQueueMakeCoupleBOS,
    WaitingQueueLoadImage,
)

from ..works.piv import WorkPIV

from . import prepare_path_dir_result

from ..data_objects.piv import get_name_bos, ArrayCouple
from ..util.util import logger, imread


class TopologyBOS(TopologyBase):
    """Topology for BOS.

    See https://en.wikipedia.org/wiki/Background-oriented_schlieren_technique

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

        params._set_attrib("reference", 0)

        params._set_doc(
            """
reference : str or int, {0}

    Reference file (from which the displacements will be computed). Can be an
    absolute file path, a file name or the index in the list of files found
    from the parameters in ``params.series``.

"""
        )

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
            "saving", attribs={"path": None, "how": "ask", "postfix": "piv"}
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

        return params

    def __init__(self, params=None, logging_level="info", nb_max_workers=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.piv_work = WorkPIV(params)

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

        reference = os.path.expanduser(params.reference)

        if isinstance(reference, int):
            names = self.series.get_name_all_arrays()
            names.sort()
            path_reference = os.path.join(path_dir, names[reference])
        elif os.path.isfile(reference):
            path_reference = reference
        else:
            path_reference = os.path.join(path_dir_result, reference)
            if not os.path.isfile(path_reference):
                raise ValueError(
                    "Bad value of params.reference:" + path_reference
                )

        self.path_reference = path_reference
        self.image_reference = imread(path_reference)

        self.results = {}

        def save_piv_object(o):
            ret = o.save(path_dir_result, kind="bos")
            return ret

        self.wq_piv = WaitingQueueThreading(
            "delta", save_piv_object, self.results, topology=self
        )
        self.wq_couples = WaitingQueueMultiprocessing(
            "couple", self.piv_work.calcul, self.wq_piv, topology=self
        )
        self.wq_images = WaitingQueueMakeCoupleBOS(
            "array image",
            self.wq_couples,
            topology=self,
            image_reference=self.image_reference,
            path_reference=self.path_reference,
            serie=self.series.serie,
        )
        self.wq0 = WaitingQueueLoadImage(
            destination=self.wq_images, path_dir=path_dir, topology=self
        )

        super(TopologyBOS, self).__init__(
            [self.wq0, self.wq_images, self.wq_couples, self.wq_piv],
            path_output=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        self.add_series(self.series)

    def add_series(self, series):

        if len(series) == 0:
            logger.warning("add 0 image. No BOS to compute.")
            return

        names = series.get_name_all_arrays()

        if self.how_saving == "complete":
            names_to_compute = []
            for name in names:
                name_bos = get_name_bos(name, series.serie)
                if not os.path.exists(
                    os.path.join(self.path_dir_result, name_bos)
                ):
                    names_to_compute.append(name)

            names = names_to_compute
            if len(names) == 0:
                logger.warning(
                    'topology in mode "complete" and work already done.'
                )
                return

        nb_names = len(names)
        print("Add {} BOS fields to compute.".format(nb_names))

        logger.debug(repr(names))

        print("First files to process:", names[:4])

        self.wq0.add_name_files(names)

        # a little bit strange, to apply mask...
        try:
            params_mask = self.params.mask
        except AttributeError:
            params_mask = None

        im = self.image_reference

        couple = ArrayCouple(
            names=("", ""), arrays=(im, im), params_mask=params_mask
        )
        im, _ = couple.get_arrays()

        self.piv_work._prepare_with_image(im)

    def _print_at_exit(self, time_since_start):

        txt = "Stop compute after t = {:.2f} s".format(time_since_start)
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += " ({} bos fields, {:.2f} s/field).".format(
                nb_results, time_since_start / nb_results
            )
        else:
            txt += "."

        txt += "\npath results:\n" + self.path_dir_result

        print(txt)


params = TopologyBOS.create_default_params()

__doc__ += params._get_formatted_docs()
