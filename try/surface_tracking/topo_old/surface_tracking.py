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

from ..topologies import prepare_path_dir_result
from .base import TopologyBase
from .waiting_queues.base import (
    WaitingQueueMultiprocessing,
    WaitingQueueThreading,
    WaitingQueueLoadImagePath,
    WaitingQueueMakeCouple,
)

from ..data_objects.piv import ArrayCouple
from .. import ParamContainer
from .. import SeriesOfArrays, SerieOfArraysFromFiles
from ..util import logger
from ..works.old.surface_tracking import WorkSurfaceTracking


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

        params._set_child(
            "saving",
            attribs={
                "plot": False,
                "how_many": 1000,
                "path": None,
                "how": "hdf5",
                "postfix": "surface_tracking",
            },
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

        WorkSurfaceTracking._complete_params_with_default(params)

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

        params._set_child("mask", attribs={"strcrop": None})

        params.mask._set_doc(
            """
            Parameters describing how images are masked.

            strcrop : None, str

                Two-dimensional slice (for example '100:600, :'). If None, the
                whole image is used.  """
        )

        return params

    def __init__(self, params=None, logging_level="info", nb_max_workers=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.path = params.film.path
        self.path_ref = params.film.path_ref
        serie_arrays_ref = SerieOfArraysFromFiles(params.film.path_ref)

        self.surface_tracking_work = WorkSurfaceTracking(params)

        serie_arrays = SerieOfArraysFromFiles(
            params.film.path + "/" + params.film.fileName
        )
        self.series = SeriesOfArrays(
            serie_arrays,
            params.series.strcouple,
            ind_start=params.film.ind_start,
            ind_stop=params.film.ind_stop,
            ind_step=params.film.ind_step,
        )

        path_dir = self.path
        path_dir_result, self.how_saving = prepare_path_dir_result(
            path_dir, params.saving.path, params.saving.postfix, params.saving.how
        )

        self.path_dir_result = params.film.path
        self.results = {}

        def save_surface_tracking_object(o):
            ret = o.save(path_dir_result / o.nameFrame)
            return ret

        self.wq_sf_out = WaitingQueueThreading(
            "save_surface_tracking_object",
            save_surface_tracking_object,
            self.results,
            topology=self,
        )
        self.wq_sf_in = WaitingQueueMultiprocessing(
            "surface_tracking_work",
            self.surface_tracking_work.compute,
            self.wq_sf_out,
            topology=self,
        )

        self.wq_images = WaitingQueueMakeCouple(
            "array image", self.wq_sf_in, topology=self
        )

        self.wq0 = WaitingQueueLoadImagePath(
            destination=self.wq_images, path_dir=path_dir, topology=self
        )

        waiting_queues = [self.wq0, self.wq_images, self.wq_sf_in, self.wq_sf_out]

        super().__init__(
            waiting_queues,
            path_output=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        self.add_series(self.series)

    def add_series(self, series):
        """
        Fill working queues
        :param series: Series representing a film
        :type SeriesOfArrays
        :return:
        """
        if len(series) == 0:
            logger.warning("add 0 couple. No PIV to compute.")
            return

        if self.how_saving == "complete":
            names = []
            index_series = []
            for i, serie in enumerate(series):
                name_sf = serie.get_name_arrays()
                #                if os.path.exists(os.path.join(self.path_dir_result, name_sf[0])):
                #                    print(os.path.join(self.path_dir_result, name_sf[0]))
                #                    continue

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
        print(f"Add {nb_series} surface fields to compute.")

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

    def print_at_exit(self, time_since_start):
        pass


params = TopologySurfaceTracking.create_default_params()
__doc__ += params._get_formatted_docs()
