"""Topology for PIV computation (:mod:`fluidimage.experimental.topologies.piv`)
===============================================================================

New topology for PIV computation.

.. autoclass:: TopologyPIV
   :members:
   :private-members:

"""
import os
import json
import copy
import sys
from typing import List, Tuple

from fluidimage import SeriesOfArrays
from fluidimage.util import imread

from fluidimage.works.preproc import WorkPreproc

from fluidimage.topologies import prepare_path_dir_result
from fluidimage.data_objects.display_pre import DisplayPreProc
from fluidimage.data_objects.preproc import get_name_preproc

from fluidimage.util.log import logger
from fluidimage.experimental.topologies.base import TopologyBase


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

    def __init__(self, params, logging_level="info", nb_max_workers=None):
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

        serie = self.series.get_serie_from_index(0)
        self.nb_items_per_serie = serie.get_nb_arrays()

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
        queue_name_series = self.add_queue("filename series")
        queue_paths = self.add_queue("image paths")
        queue_array = self.add_queue("arrays")
        queue_array_series = self.add_queue("array series")
        queue_preproc = self.add_queue("preproc")

        self.add_work(
            "fill (series name couple, paths)",
            func_or_cls=self.fill_name_series_and_paths,
            output_queue=(queue_name_series, queue_paths),
            kind=("global", "one shot"),
        )

        def save_preproc_results_object(self, o):
            return o.save(path=self.path_dir_result)
    


    