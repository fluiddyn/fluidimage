"""Topology for PIV computation (:mod:`fluidimage.topologies.piv`)
==================================================================

.. autoclass:: TopologyPIV
   :members:
   :private-members:

"""
import copy
import json
import sys

from fluidimage import ParamContainer, SeriesOfArrays
from fluidimage.data_objects.piv import ArrayCouple, get_name_piv
from fluidimage.topologies import TopologyBase, prepare_path_dir_result
from fluidimage.util import DEBUG, imread, logger
from fluidimage.works.piv import WorkPIV

from . import image2image


def is_name_in_queue(image_name, queue):
    """Check if a name is in a queue of series"""
    for names in queue.values():
        if image_name in names:
            return True
    return False


class TopologyPIV(TopologyBase):
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

    WorkVelocimetry = WorkPIV

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

    To see what it gives, one can use IPython and range:

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

        cls.WorkVelocimetry._complete_params_with_default(params)

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

    def __init__(self, params, logging_level="info", nb_max_workers=None):

        self.params = params

        self.series = SeriesOfArrays(
            params.series.path,
            params.series.strcouple,
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

    def fill_couples_of_names_and_paths(self, input_queue, output_queues):
        """Fill the two first queues"""
        assert input_queue is None
        queue_couples_of_names = output_queues[0]
        queue_paths = output_queues[1]

        series = self.series
        if not series:
            logger.warning("add 0 couple. No PIV to compute.")
            return
        if self.how_saving == "complete":
            index_series = []
            for ind_serie, serie in self.series.items():
                name_piv = get_name_piv(serie, prefix="piv")
                if not (self.path_dir_result / name_piv).exists():
                    index_series.append(ind_serie)

            if not index_series:
                logger.warning(
                    'topology in mode "complete" and work already done.'
                )
                return

            series.set_index_series(index_series)

            if logger.isEnabledFor(DEBUG):
                logger.debug(repr([serie.get_name_arrays() for serie in series]))

        nb_series = len(series)
        logger.info(f"Add {nb_series} PIV fields to compute.")

        for iserie, serie in enumerate(series):
            if iserie > 1:
                break
            logger.info(
                "Files of serie {}: {}".format(iserie, serie.get_name_arrays())
            )

        for ind_serie, serie in series.items():
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
                array1 = queue_arrays[couple[0]]
                array2 = queue_arrays[couple[1]]
                serie = copy.copy(self.series.get_serie_from_index(key))

                # logger.debug(
                #     f"create couple {key}: {couple}, ({array1}, {array2})"
                # )
                array_couple = ArrayCouple(
                    names=(couple[0], couple[1]),
                    arrays=(array1, array2),
                    params_mask=params_mask,
                    serie=serie,
                )
                output_queue[key] = array_couple
                del queue_couples_of_names[key]
                # remove the image_array if it not will be used anymore
                if not is_name_in_queue(couple[0], queue_couples_of_names):
                    del queue_arrays[couple[0]]
                if not is_name_in_queue(couple[1], queue_couples_of_names):
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

        txt += "\npath results:\n" + str(self.path_dir_result)

        return txt


if "sphinx" in sys.modules:
    params = TopologyPIV.create_default_params()
    __doc__ += params._get_formatted_docs()


if __name__ == "__main__":
    from fluidimage import path_image_samples

    params = TopologyPIV.create_default_params()

    params.series.path = str(path_image_samples / "Karman/Images")
    params.series.ind_start = 1
    params.series.ind_step = 2

    params.piv0.shape_crop_im0 = 32
    params.multipass.number = 2
    params.multipass.use_tps = False

    params.mask.strcrop = ":, 50:500"

    params.preproc.im2im = "numpy.ones_like"

    # params.saving.how = 'complete'
    params.saving.postfix = "piv_example"

    topo = TopologyPIV(params, logging_level="info")

    topo.make_code_graphviz("tmp.dot")
