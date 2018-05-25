#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 09:57:58 2018

@author: blancfat8p
"""

"""Topology for SurfaceTracking computation (:mod:`fluidimage.topologies.surface_tracking`)
==================================================================

.. autoclass:: TopologySurfaceTracking
   :members:
   :private-members:

"""
import json

from . import prepare_path_dir_result
from .base import TopologyBase
from .waiting_queues.base import (
    WaitingQueueMultiprocessing,
    WaitingQueueThreading,
    WaitingQueueLoadImagePath,
)
from .. import ParamContainer
from .. import SeriesOfArrays
from ..data_objects.surfaceTracking import *
from ..util.util import logger
from ..works.surfaceTracking.surface_tracking import WorkSurfaceTracking


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
            "film",
            attribs={
                "fileName": "",
                "path": "",
                "pathRef": "",
                "ind_start": 0,
                "ind_stop": None,
                "ind_step": 1,
            },
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

        return params

    def __init__(self, params=None, logging_level="info", nb_max_workers=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.path = params.film.path
        self.surface_tracking_work = WorkSurfaceTracking(params)

        serie_arrays = SerieOfArraysFromFiles(
            params.film.path + "/" + params.film.fileName
        )
        print(params.film.path + "/" + params.film.fileName)
        self.series = SeriesOfArrays(
            serie_arrays,
            None,
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
            ret = o.save(path_dir_result + "/results/" + o.nameFrame)
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

        self.wq0 = WaitingQueueLoadImagePath(
            destination=self.wq_sf_in, path_dir=path_dir, topology=self
        )

        waiting_queues = [self.wq0, self.wq_sf_in, self.wq_sf_out]

        super(TopologySurfaceTracking, self).__init__(
            waiting_queues,
            path_output=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        self.add_frames(self.series)  # similar to add

    def add_frames(self, series):
        """
        Inspired by Topologies/piv add_Series
        :param frames:
        :return:
        """

        if len(series) == 0:
            logger.warning("add 0 image. No image to process.")
            return

        names = series.get_name_all_arrays()

        if self.how_saving == "complete":
            names_to_compute = []
            for name in names:
                if not os.path.exists(os.path.join(self.path_dir_result, name)):
                    names_to_compute.append(name)

            names = names_to_compute
            if len(names) == 0:
                logger.warning(
                    'topology in mode "complete" and work already done.'
                )
                return

        nb_names = len(names)
        print("Add {} images to compute.".format(nb_names))

        logger.debug(repr(names))

        print("First files to process:", names[:4])

        self.wq0.add_name_files(names)

    def _print_at_exit(self, time_since_start):
        pass

    def get_file(self, path="./", fn=None):
        """read the files with SeriesOfArraysFromFile in path or a specified file if fn-arg is given"""
        path = self.path
        if fn is None:
            # print(path + '/*')
            film = SerieOfArraysFromFiles(path + "/" + params.film.fileName)
        return film

    def get_name_surface_tracking(serie, prefix="piv"):
        index_slices = serie._index_slices
        str_indices = ""
        for i, inds in enumerate(index_slices):
            index = inds[0]
            str_index = get_str_index(serie, i, index)
            if len(inds) > 1:
                str_index += "-" + get_str_index(serie, i, inds[1] - 1)

            if i > 1:
                str_indices += serie._index_separators[i - 1]
            str_indices += str_index

        name = prefix + "_" + str_indices + ".h5"
        return name


class WaitingQueueLoadFrame(WaitingQueueThreading):
    nb_max_workers = 8

    def __init__(self, *args, **kwargs):
        self.path_dir = kwargs.pop("path_dir")
        super().__init__(
            name="SurfaceTracking", work=WorkSurfaceTracking, *args, **kwargs
        )
        self.num_frame_to_compute = []
        self.work_name = __name__ + ".load"

    def add_name_files(self, names):
        self.update(
            {name: os.path.join(self.path_dir, name) for name in names}, names
        )


params = TopologySurfaceTracking.create_default_params()
__doc__ += params._get_formatted_docs()
