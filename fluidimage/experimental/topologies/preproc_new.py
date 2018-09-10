"""Topology for preprocessing (:mod:`fluidimage.experimental.topologies.preproc_new`)
=====================================================================================

New topology for preprocessing images.

.. autoclass:: TopologyPreproc
   :members:
   :private-members:

"""
import os
import json
import copy
import sys
from typing import List, Tuple, Dict, Any
from fluiddyn.util.paramcontainer import ParamContainer

from fluidimage import SeriesOfArrays
from fluidimage.util import imread

from fluidimage.works.preproc import WorkPreproc

from fluidimage.topologies import prepare_path_dir_result
from fluidimage.data_objects.display_pre import DisplayPreProc
from fluidimage.data_objects.preproc import (
    get_name_preproc,
    ArraySerie as ArraySubset,
)

from fluidimage.util.log import logger
from fluidimage.experimental.topologies.base import TopologyBase

from .piv_new import still_is_in_dict


class TopologyPreproc(TopologyBase):
    """Preprocess series of images and provides interface for I/O and
    multiprocessing.

    Parameters
    ----------

    params: None

      A ParamContainer containing the parameters for the computation.

    logging_level: str, {'warning', 'info', 'debug', ...}

      Logging level.

    nb_max_workers: None, int

      Maximum numbers of "workers". If None, a number is computed from the
      number of cores detected. If there are memory errors, you can try to
      decrease the number of workers.

    """

    @classmethod
    def create_default_params(cls, backend="python"):
        """Class method returning the default parameters.

        Parameters
        ----------

        backend : {'python', 'opencv'}

            Specifies which backend to use.

        """
        params = WorkPreproc.create_default_params(backend)
        params.preproc.series._set_attribs(
            {
                "strcouple": "i:i+1",
                "ind_start": 0,
                "ind_stop": None,
                "ind_step": 1,
                "sequential_loading": True,
            }
        )
        params.preproc.series._set_doc(
            """
Parameters indicating the input series of images.

path : str, {''}

    String indicating the input images (can be a full path towards an image
    file or a string given to `glob`).

strcouple : str, {'i:i+1'}

    Determines the subset from the whole series of images that should be loaded
    and preprocessed together. Particularly useful when temporal filtering requires
    loading multiple neighbouring images at a time.

    For example, for a series of images with just one index,

    >>> params.preproc.series.strcouple = 'i:i+1'   # load one image at a time
    >>> params.preproc.series.strcouple = 'i-2:i+3'  # loads 5 images at a time

    Similarly for two indices,

    >>> params.preproc.series.strcouple = 'i:i+1,0'   # load one image at a time, with second index fixed
    >>> params.preproc.series.strcouple = 'i-2:i+3,i'  # loads 5 images at a time, with second index free

    ..todo::

        rename this parameter to strsubset / strslice

ind_start : int, {0}

    Start index for the whole series of images being loaded.
    For more details: see `class SeriesOfArrays`.

ind_stop : int or None

    Stop index for the whole series of images being loaded.
    For more details: see `class SeriesOfArrays`.

ind_step : int, {1}

    Step index for the whole series of images being loaded.
    For more details: see `class SeriesOfArrays`.

sequential_loading : bool, {True}

    When set as `true` the image loading waiting queue `WaitingQueueLoadImageSeries`
    is processed sequentially. i.e. only one subset of the whole series is loaded at a time.
"""
        )

        params.preproc.series._set_doc(
            """
Parameters describing image loading prior to preprocessing.

strcouple : str
    Determines the subset from the whole series of images that should be loaded
    and preprocessed together. Particularly useful when temporal filtering requires
    multiple images.

    For example, for a series of images with just one index,

        >>> strcouple = 'i:i+1'   # load one image at a time
        >>> strcouple = 'i-2:i+3'  # loads 5 images at a time

    Similarly for two indices,

        >>> strcouple = 'i:i+1,0'   # load one image at a time, with second index fixed
        >>> strcouple = 'i-2:i+3,0'  # loads 5 images at a time, with second index fixed

    ..todo::

        rename this parameter to strsubset / strslice

ind_start : int
    Start index for the whole series of images being loaded.
    For more details: see `class SeriesOfArrays`.

ind_stop : int
    Stop index for the whole series of images being loaded.
    For more details: see `class SeriesOfArrays`.

ind_step : int
    Step index for the whole series of images being loaded.
    For more details: see `class SeriesOfArrays`.

sequential_loading : bool
    When set as `true` the image loading waiting queue `WaitingQueueLoadImageSeries`
    is processed sequentially. i.e. only one subset of the whole series is loaded at a time.

"""
        )

        params.preproc._set_child(
            "saving",
            attribs={
                "path": None,
                "strcouple": None,
                "how": "ask",
                "format": "img",
                "postfix": "pre",
            },
        )

        params.preproc.saving._set_doc(
            """
Parameters describing image saving after preprocessing.

path : str or None
    Path to which preprocessed images are saved.

strcouple : str or None
    Determines the sub-subset of images must be saved from subset of images that were
    loaded and preprocessed. When set as None, saves the middle image from every subset.

    ..todo::

        rename this parameter to strsubset / strslice 

how : str {'ask', 'new_dir', 'complete', 'recompute'}
    How preprocessed images must be saved if it already exists or not.

format : str {'img', 'hdf5'}
    Format in which preprocessed image data must be saved.

postfix : str
    A suffix added to the new directory where preprocessed images are saved.

"""
        )

        params._set_internal_attr(
            "_value_text",
            json.dumps(
                {
                    "program": "fluidimage",
                    "module": "fluidimage.topologies.preproc",
                    "class": "TopologyPreproc",
                }
            ),
        )

        return params

    def __init__(
        self, params: ParamContainer, logging_level="info", nb_max_workers=None
    ):
        self.params = params.preproc
        self.preproc_work = WorkPreproc(params)
        serie_arrays = self.preproc_work.serie_arrays
        self.series = SeriesOfArrays(
            serie_arrays,
            params.preproc.series.strcouple,
            ind_start=params.preproc.series.ind_start,
            ind_stop=params.preproc.series.ind_stop,
            ind_step=params.preproc.series.ind_step,
        )

        subset = self.series.get_serie_from_index(0)
        self.nb_items_per_serie = subset.get_nb_arrays()

        if os.path.isdir(params.preproc.series.path):
            path_dir = params.preproc.series.path
        else:
            path_dir = os.path.dirname(params.preproc.series.path)
        self.path_dir_input = path_dir

        path_dir_result, self.how_saving = prepare_path_dir_result(
            path_dir,
            params.preproc.saving.path,
            params.preproc.saving.postfix,
            params.preproc.saving.how,
        )

        super().__init__(
            path_dir_result=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        self.params.saving.path = self.path_dir_result
        self.results = self.preproc_work.results
        self.display = self.preproc_work.display

        # Define Queues
        queue_name_subsets = self.add_queue("subsets of filename series")
        queue_paths = self.add_queue("image paths")
        queue_arrays = self.add_queue("arrays")
        queue_array_subsets = self.add_queue("subsets of array series")
        queue_preproc = self.add_queue("preproc")

        self.add_work(
            "fill (name subsets, paths)",
            func_or_cls=self.fill_name_series_and_paths,
            output_queue=(queue_name_subsets, queue_paths),
            kind=("global", "one shot"),
        )

        self.add_work(
            "path -> arrays",
            func_or_cls=imread,
            input_queue=queue_paths,
            output_queue=queue_arrays,
            kind="io",
        )

        self.add_work(
            "make subsets of array series",
            func_or_cls=self.make_subset,
            input_queue=(queue_name_subsets, queue_arrays),
            output_queue=queue_array_subsets,
            kind="global",
        )

        self.work_preproc = WorkPreproc(params)

        self.add_work(
            "subsets -> preproc",
            func_or_cls=self.work_preproc.calcul,
            params_cls=params,
            input_queue=queue_array_subsets,
            output_queue=queue_preproc,
        )

        self.add_work(
            "save preprocessed arrays",
            func_or_cls=self.save_preproc_results_object,
            input_queue=queue_preproc,
            kind="io",
        )

    def save_preproc_results_object(self, o: ArraySubset):
        return o.save(path=self.path_dir_result)

    def init_series(self) -> List[str]:
        """Initializes the SeriesOfArrays object `self.series` based on input
        parameters."""
        series = self.series
        if len(series) == 0:
            logger.warning("encountered empty series. No images to preprocess.")
            return

        if self.how_saving == "complete":
            names = []
            index_series = []
            for i, subset in enumerate(series):
                names_serie = subset.get_name_arrays()
                name_preproc = get_name_preproc(
                    subset,
                    names_serie,
                    i,
                    series.nb_series,
                    self.params.saving.format,
                )
                if os.path.exists(
                    os.path.join(self.path_dir_result, name_preproc)
                ):
                    continue

                for name in names_serie:
                    if name not in names:
                        names.append(name)

                index_series.append(i + series.ind_start)

            if len(index_series) == 0:
                logger.warning(
                    'topology in mode "complete" and work already done.'
                )
                return

            series.set_index_series(index_series)

            logger.debug(repr(names))
            logger.debug(repr([subset.get_name_arrays() for serie in series]))
        else:
            names = series.get_name_all_arrays()

        logger.info("Add {} image serie(s) to compute.".format(len(series)))
        return names

    def fill_name_series_and_paths(
        self, input_queue: None, output_queues: Tuple[Dict]
    ) -> None:
        queue_name_subsets, queue_paths = output_queues

        names = self.init_series()
        # TODO: See if names have to be used or not
        for i, subset in enumerate(self.series):
            inew = i * self.series.ind_step + self.series.ind_start
            queue_name_subsets[inew] = subset.get_name_arrays()
            for name, path in zip(
                subset.get_name_arrays(), subset.get_path_files()
            ):
                queue_paths[name] = path

    def make_subset(self, input_queues: Tuple[Dict], output_queue: Dict) -> bool:
        # for readablity
        queue_name_subsets, queue_arrays = input_queues

        # for each name subset
        for key, names in queue_name_subsets.items():
            # if correspondant arrays have been loaded from images,
            # make an array subset
            if all([name in queue_arrays for name in names]):
                arrays = (queue_arrays[name] for name in names)
                serie = copy.copy(self.series.get_serie_from_index(key))

                array_subset = ArraySubset(
                    names=names, arrays=arrays, serie=serie
                )
                output_queue[key] = array_subset
                del queue_name_subsets[key]
                # remove the image_array if it not will be used anymore

                key_arrays = list(queue_arrays.keys())
                for key_array in key_arrays:
                    if not still_is_in_dict(key_array, queue_name_subsets):
                        del queue_arrays[key_array]

                return True
        return False
