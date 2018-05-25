"""Topology for PIV computation (:mod:`fluidimage.topologies.piv`)
==================================================================

.. autoclass:: TopologyPIV
   :members:
   :private-members:

"""
import os
import json

from .. import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays

from .base import TopologyBase

from .waiting_queues.base import (
    WaitingQueueMultiprocessing,
    WaitingQueueThreading,
    WaitingQueueMakeCouple,
    WaitingQueueLoadImage,
)

from . import prepare_path_dir_result
from ..works.piv import WorkPIV
from ..data_objects.piv import get_name_piv, ArrayCouple
from ..util.util import logger
from . import image2image


class TopologyPIV(TopologyBase):
    """Topology for PIV.

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

        params._set_child(
            "series",
            attribs={
                "path": "",
                "strcouple": "i:i+2",
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

strcouple : 'i:i+2'

    String indicating as a Python slicing how couples of images are formed.
    There is one couple per value of `i`. The values of `i` are set with the
    other parameters `ind_start`, `ind_step` and `ind_stop` approximately with
    the function range (`range(ind_start, ind_stop, ind_step)`).

    Python slicing is a very powerful notation to define subset from a
    (possibly multidimensional) set of images. For a user, an alternative is to
    understand how Python slicing works. See for example this page:
    http://stackoverflow.com/questions/509211/explain-pythons-slice-notation.

    Another possibility is to follow simple examples:

    For single-frame images (im0, im1, im2, im3, ...), we keep the default
    value 'i:i+2' to form the couples (im0, im1), (im1, im2), ...

    To see what it gives, one can use ipython and range:

    >>> i = 0
    >>> list(range(10))[i:i+2]
    [0, 1]

    >>> list(range(10))[i:i+4:2]
    [0, 2]

    We see that we can also use the value 'i:i+4:2' to form the couples (im0,
    im2), (im1, im3), ...

    For double-frame images (im1a, im1b, im2a, im2b, ...) you can write

    >>> params.series.strcouple = 'i, 0:2'

    In this case, the first couple will be (im1a, im1b).

    To get the first couple (im1a, im1a), we would have to write

    >>> params.series.strcouple = 'i:i+2, 0'

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
                    "module": "fluidimage.topologies.piv",
                    "class": "TopologyPIV",
                }
            ),
        )

        params._set_child("preproc")
        image2image.complete_im2im_params_with_default(params.preproc)

        return params

    def __init__(self, params=None, logging_level="info", nb_max_workers=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.piv_work = WorkPIV(params)

        serie_arrays = SerieOfArraysFromFiles(params.series.path)

        self.series = SeriesOfArrays(
            serie_arrays,
            params.series.strcouple,
            ind_start=params.series.ind_start,
            ind_stop=params.series.ind_stop,
            ind_step=params.series.ind_step,
        )

        path_dir = self.series.serie.path_dir
        path_dir_result, self.how_saving = prepare_path_dir_result(
            path_dir, params.saving.path, params.saving.postfix, params.saving.how
        )

        self.path_dir_result = path_dir_result

        self.results = {}

        def save_piv_object(o):
            ret = o.save(path_dir_result)
            return ret

        self.wq_piv = WaitingQueueThreading(
            "delta", save_piv_object, self.results, topology=self
        )
        self.wq_couples = WaitingQueueMultiprocessing(
            "couple", self.piv_work.calcul, self.wq_piv, topology=self
        )

        self.wq_images = WaitingQueueMakeCouple(
            "array image", self.wq_couples, topology=self
        )

        if params.preproc.im2im is not None:
            self.im2im_func = image2image.TopologyImage2Image.init_im2im(
                self, params.preproc
            )

            self.wq_images0 = WaitingQueueMultiprocessing(
                "image ", self.im2im_func, self.wq_images, topology=self
            )
            wq_after_load = self.wq_images0
        else:
            wq_after_load = self.wq_images

        self.wq0 = WaitingQueueLoadImage(
            destination=wq_after_load, path_dir=path_dir, topology=self
        )

        if params.preproc.im2im is not None:
            waiting_queues = [
                self.wq0,
                self.wq_images0,
                self.wq_images,
                self.wq_couples,
                self.wq_piv,
            ]
        else:
            waiting_queues = [
                self.wq0,
                self.wq_images,
                self.wq_couples,
                self.wq_piv,
            ]

        super(TopologyPIV, self).__init__(
            waiting_queues,
            path_output=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        self.add_series(self.series)

    def add_series(self, series):

        if len(series) == 0:
            logger.warning("add 0 couple. No PIV to compute.")
            return

        if self.how_saving == "complete":
            names = []
            index_series = []
            for i, serie in enumerate(series):
                name_piv = get_name_piv(serie, prefix="piv")
                if os.path.exists(os.path.join(self.path_dir_result, name_piv)):
                    continue

                for name in serie.get_name_arrays():
                    if name not in names:
                        names.append(name)

                index_series.append(i * series.ind_step + series.ind_start)

            if len(index_series) == 0:
                logger.warning(
                    'topology in mode "complete" and work already done.'
                )
                return

            series.set_index_series(index_series)

            logger.debug(repr(names))
            logger.debug(repr([serie.get_name_arrays() for serie in series]))
        else:
            names = series.get_name_all_arrays()

        nb_series = len(series)
        print("Add {} PIV fields to compute.".format(nb_series))

        for i, serie in enumerate(series):
            if i > 1:
                break

            print("Files of serie {}: {}".format(i, serie.get_name_arrays()))

        self.wq0.add_name_files(names)
        self.wq_images.add_series(series)

        k, o = self.wq0.popitem()
        im = self.wq0.work(o)
        self.wq0.fill_destination(k, im)

        # a little bit strange, to apply mask...
        try:
            params_mask = self.params.mask
        except AttributeError:
            params_mask = None

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
            txt += " ({} piv fields, {:.2f} s/field).".format(
                nb_results, time_since_start / nb_results
            )
        else:
            txt += "."

        txt += "\npath results:\n" + self.path_dir_result

        print(txt)


params = TopologyPIV.create_default_params()

__doc__ += params._get_formatted_docs()
