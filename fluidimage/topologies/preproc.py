"""Topology for image preprocessing (:mod:`fluidimage.topologies.preproc`)
==========================================================================

.. autoclass:: TopologyPreproc
   :members:
   :private-members:

"""
import copy
import json
import os
import sys
from typing import Dict, List, Tuple

from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage import SeriesOfArrays
from fluidimage.data_objects.preproc import ArraySerie as ArraySubset
from fluidimage.data_objects.preproc import get_name_preproc
from fluidimage.topologies import TopologyBase, prepare_path_dir_result
from fluidimage.util import DEBUG, imread, logger
from fluidimage.works.preproc import WorkPreproc

from . import image2image
from .piv import is_name_in_queue


class TopologyPreproc(TopologyBase):
    """Preprocess series of images.

    The most useful methods for the user (in particular :func:`compute`) are
    defined in the base class :class:`fluidimage.topologies.base.TopologyBase`.

    Parameters
    ----------

    params: None

      A ParamContainer (created with the class method
      :func:`create_default_params`) containing the parameters for the
      computation.

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

        Typical usage::

          params = TopologyPreproc.create_default_params()
          # modify parameters here
          ...

          topo = TopologyPreproc(params)

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
            }
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

    .. todo::

        Rename this parameter to strsubset / strslice

ind_start : int
    Start index for the whole series of images being loaded.
    For more details: see `class SeriesOfArrays`.

ind_stop : int
    Stop index for the whole series of images being loaded.
    For more details: see `class SeriesOfArrays`.

ind_step : int
    Step index for the whole series of images being loaded.
    For more details: see `class SeriesOfArrays`.

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

str_subset : str or None
    NotImplemented! Determines the sub-subset of images must be saved from subset
    of images that were loaded and preprocessed. When set as None, saves the
    middle image from every subset.

    .. todo::

        Implement the option params.saving.str_subset...

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

        params._set_child("im2im")
        image2image.complete_im2im_params_with_default(params.im2im)

        return params

    def __init__(
        self, params: ParamContainer, logging_level="info", nb_max_workers=None
    ):
        self.params = params.preproc

        self.preproc_work = WorkPreproc(params)
        self.results = []
        self.display = self.preproc_work.display

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

        # Define waiting queues
        queue_subsets_of_names = self.add_queue("subsets of filenames")
        queue_paths = self.add_queue("image paths")
        queue_arrays = queue_arrays1 = self.add_queue("arrays")
        queue_subsets_of_arrays = self.add_queue("subsets of arrays")
        queue_preproc_objects = self.add_queue("preproc results")

        if params.im2im.im2im is not None:
            queue_arrays1 = self.add_queue("arrays1")

        # Define works
        self.add_work(
            "fill (subsets_of_names, paths)",
            func_or_cls=self.fill_subsets_of_names_and_paths,
            output_queue=(queue_subsets_of_names, queue_paths),
            kind=("global", "one shot"),
        )

        self.add_work(
            "imread",
            func_or_cls=imread,
            input_queue=queue_paths,
            output_queue=queue_arrays,
            kind="io",
        )

        if params.im2im.im2im is not None:
            im2im_func = image2image.TopologyImage2Image.init_im2im(
                self, params.im2im
            )

            self.add_work(
                "image2image",
                func_or_cls=im2im_func,
                input_queue=queue_arrays,
                output_queue=queue_arrays1,
            )

        self.add_work(
            "make subsets of arrays",
            func_or_cls=self.make_subsets,
            input_queue=(queue_subsets_of_names, queue_arrays1),
            output_queue=queue_subsets_of_arrays,
            kind="global",
        )

        self.add_work(
            "preproc a subset of arrays",
            func_or_cls=self.preproc_work.calcul,
            params_cls=params,
            input_queue=queue_subsets_of_arrays,
            output_queue=queue_preproc_objects,
        )

        self.add_work(
            "save images",
            func_or_cls=self.save_preproc_object,
            input_queue=queue_preproc_objects,
            kind="io",
        )

    def save_preproc_object(self, obj: ArraySubset):
        """Save a preprocessing object"""
        ret = obj.save(path=self.path_dir_result)
        self.results.append(ret)

    def init_series(self) -> List[str]:
        """Initializes the SeriesOfArrays object `self.series` based on input
        parameters."""
        series = self.series
        if not series:
            logger.warning("encountered empty series. No images to preprocess.")
            return

        if self.how_saving == "complete":
            index_subsets = []
            for ind_subset, subset in self.series.items():
                names_serie = subset.get_name_arrays()
                name_preproc = get_name_preproc(
                    subset,
                    names_serie,
                    ind_subset,
                    series.nb_series,
                    self.params.saving.format,
                )
                if not (self.path_dir_result / name_preproc).exists():
                    index_subsets.append(ind_subset)
            series.set_index_series(index_subsets)
            if logger.isEnabledFor(DEBUG):
                logger.debug(
                    repr([subset.get_name_arrays() for subset in series])
                )

        nb_subsets = len(series)
        if nb_subsets == 0:
            logger.warning('topology in mode "complete" and work already done.')
            return
        elif nb_subsets == 1:
            plurial = ""
        else:
            plurial = "s"

        logger.info(f"Add {nb_subsets} image serie{plurial} to compute.")

    def fill_subsets_of_names_and_paths(
        self, input_queue: None, output_queues: Tuple[Dict]
    ) -> None:
        """Fill the two first queues"""
        assert input_queue is None
        queue_subsets_of_names, queue_paths = output_queues

        self.init_series()

        for ind_subset, subset in self.series.items():
            queue_subsets_of_names[ind_subset] = subset.get_name_arrays()
            for name, path in subset.get_name_path_arrays():
                queue_paths[name] = path

    def make_subsets(self, input_queues: Tuple[Dict], output_queue: Dict) -> bool:
        """Create the subsets of images"""
        queue_subsets_of_names, queue_arrays = input_queues
        # for each name subset
        for key, names in list(queue_subsets_of_names.items()):
            # if correspondant arrays have been loaded from images,
            # make an array subset
            if all([name in queue_arrays for name in names]):
                arrays = (queue_arrays[name] for name in names)
                serie = copy.copy(self.series.get_serie_from_index(key))

                array_subset = ArraySubset(
                    names=names, arrays=arrays, serie=serie
                )
                output_queue[key] = array_subset
                del queue_subsets_of_names[key]
                # remove the image_array if it not will be used anymore

                key_arrays = list(queue_arrays.keys())
                for key_array in key_arrays:
                    if not is_name_in_queue(key_array, queue_subsets_of_names):
                        del queue_arrays[key_array]


if "sphinx" in sys.modules:
    params = TopologyPreproc.create_default_params()

    doc = params._get_formatted_docs()

    lines = doc.split("\n")

    strings = ("Parameters", "References", "----------")
    doc = "\n".join(
        line
        for line in doc.split("\n")
        if not any(line.endswith(string) for string in strings)
    )
    __doc__ += doc
