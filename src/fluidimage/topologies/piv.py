"""Topology for PIV computation (:mod:`fluidimage.topologies.piv`)
==================================================================

.. autoclass:: TopologyPIV
   :members:
   :private-members:

"""

import copy
import sys
from pathlib import Path

from fluidimage import ParamContainer, SeriesOfArrays
from fluidimage.data_objects.piv import ArrayCouple, get_name_piv
from fluidimage.topologies import TopologyBaseFromSeries, prepare_path_dir_result
from fluidimage.topologies.splitters import SplitterFromSeries
from fluidimage.util import imread, logger
from fluidimage.works import image2image
from fluidimage.works.piv import WorkPIV


class TopologyPIV(TopologyBaseFromSeries):
    """Topology for PIV computation.

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

      Maximum numbers of "workers". If None, a number is estimated from the
      number of cores detected. If there are memory errors, you can try to
      decrease the number of workers.

    """

    _short_name = "piv"

    WorkVelocimetry = WorkPIV
    Splitter = SplitterFromSeries

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

        cls.WorkVelocimetry._complete_params_with_default(params)

        params._set_child("preproc")
        image2image.complete_im2im_params_with_default(params.preproc)

        return params

    def __init__(self, params, logging_level="info", nb_max_workers=None):
        self.params = params

        self.series = SeriesOfArrays(
            params.series.path,
            params.series.str_subset,
            ind_start=params.series.ind_start,
            ind_stop=params.series.ind_stop,
            ind_step=params.series.ind_step,
        )

        path_dir = self.series.serie.path_dir
        path_dir_result, self.how_saving = prepare_path_dir_result(
            path_dir, params.saving.path, params.saving.postfix, params.saving.how
        )

        super().__init__(
            path_dir_result=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        queue_couples_of_names = self.add_queue("couples of names")
        queue_paths = self.add_queue("paths")
        queue_arrays = queue_arrays1 = self.add_queue("arrays")
        queue_couples_of_arrays = self.add_queue("couples of arrays")
        queue_piv = self.add_queue("piv")

        if params.preproc.im2im is not None:
            queue_arrays1 = self.add_queue("arrays1")

        self.add_work(
            "fill (couples of names, paths)",
            func_or_cls=self.fill_couples_of_names_and_paths,
            output_queue=(queue_couples_of_names, queue_paths),
            kind=("global", "one shot"),
        )
        self.add_work(
            "read array",
            func_or_cls=imread,
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
            "make couples of arrays",
            func_or_cls=self.make_couples,
            params_cls=None,
            input_queue=(queue_couples_of_names, queue_arrays1),
            output_queue=queue_couples_of_arrays,
            kind="global",
        )

        self.work_piv = self.WorkVelocimetry(self.params)

        self.add_work(
            "compute piv",
            func_or_cls=self.work_piv.calcul,
            params_cls=params,
            input_queue=queue_couples_of_arrays,
            output_queue=queue_piv,
        )

        self.add_work(
            "save piv",
            func_or_cls=self.save_piv_object,
            input_queue=queue_piv,
            kind="io",
        )
        self.results = []

    def save_piv_object(self, obj):
        """Save a PIV object"""
        ret = obj.save(self.path_dir_result)
        self.results.append(ret)

    def compute_indices_to_be_computed(self):
        """Compute the indices corresponding to the series to be computed"""
        index_series = []
        for ind_serie, serie in self.series.items():
            name_piv = get_name_piv(serie, prefix="piv")
            if not (self.path_dir_result / name_piv).exists():
                index_series.append(ind_serie)
        return index_series

    _message_empty_series = "add 0 couple. No PIV to compute."

    def fill_couples_of_names_and_paths(self, input_queue, output_queues):
        """Fill the two first queues"""
        assert input_queue is None
        queue_couples_of_names = output_queues[0]
        queue_paths = output_queues[1]

        self.init_series()

        for iserie, serie in enumerate(self.series):
            if iserie > 1:
                break
            logger.info("Files of serie %s: %s", iserie, serie.get_name_arrays())

        for ind_serie, serie in self.series.items():
            queue_couples_of_names[ind_serie] = serie.get_name_arrays()
            for name, path in serie.get_name_path_arrays():
                queue_paths[name] = path

    def make_couples(self, input_queues, output_queue):
        """Make the couples of arrays"""
        queue_couples_of_names = input_queues[0]
        queue_arrays = input_queues[1]

        try:
            params_mask = self.params.mask
        except AttributeError:
            params_mask = None
        # for each name couple
        for key, couple in tuple(queue_couples_of_names.items()):
            # if correspondant arrays are available, make an array couple
            if (
                couple[0] in queue_arrays.keys()
                and couple[1] in queue_arrays.keys()
            ):
                serie = copy.copy(self.series.get_serie_from_index(key))
                array1 = queue_arrays[couple[0]]
                array2 = queue_arrays[couple[1]]

                if isinstance(array1, Exception):
                    array_couple = array1
                elif isinstance(array2, Exception):
                    array_couple = array2
                else:
                    array_couple = ArrayCouple(
                        names=(couple[0], couple[1]),
                        arrays=(array1, array2),
                        params_mask=params_mask,
                        serie=serie,
                    )

                output_queue[key] = array_couple
                del queue_couples_of_names[key]
                # remove the image_array if it not will be used anymore
                if not queue_couples_of_names.is_name_in_values(couple[0]):
                    del queue_arrays[couple[0]]
                if not queue_couples_of_names.is_name_in_values(couple[1]):
                    del queue_arrays[couple[1]]

    def make_text_at_exit(self, time_since_start):
        """Make a text printed at exit"""

        txt = f"Stop compute after t = {time_since_start:.2f} s"
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += f" ({nb_results} piv fields, {time_since_start / nb_results:.2f} s/field)."
        else:
            txt += "."

        txt += "\npath results:\n" + str(Path(self.path_dir_result).absolute())

        return txt


Topology = TopologyPIV

if "sphinx" in sys.modules:
    _params = Topology.create_default_params()
    __doc__ += _params._get_formatted_docs()
