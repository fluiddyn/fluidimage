"""Topology for BOS computation (:mod:`fluidimage.experimental.topologies.bos_new`)
===================================================================================

New Topology for BOS computation.

.. autoclass:: TopologyBOS
   :members:
   :private-members:

"""
import os
import json
import copy
import sys

from fluidimage import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays
from fluidimage.experimental.topologies.base import TopologyBase
from fluidimage.topologies import prepare_path_dir_result
from fluidimage.works.piv import WorkPIV
from fluidimage.data_objects.piv import get_name_bos, ArrayCoupleBOS
from fluidimage.util.util import logger, imread
from fluidimage.topologies import image2image


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
            "saving", attribs={"path": None, "how": "ask", "postfix": "bos"}
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
                    "class": "Topologybos",
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

        super(TopologyBOS, self).__init__(
            path_output=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        queue_series_names_couples = self.add_queue("series_names_couple")
        queue_paths = self.add_queue("paths")
        queue_arrays = queue_arrays1 = self.add_queue("arrays")
        queue_array_couples = self.add_queue("couples of arrays")
        queue_bos = self.add_queue("bos")

        if params.preproc.im2im is not None:
            queue_arrays1 = self.add_queue("arrays1")

        self.add_work(
            "fill (series name couple, paths)",
            func_or_cls=self.fill_series_name_couple_and_path,
            output_queue=(queue_series_names_couples, queue_paths),
            kind=("global", "one shot"),
        )
        self.add_work(
            "path -> arrays",
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
                "image -> image",
                func_or_cls=im2im_func,
                input_queue=queue_arrays,
                output_queue=queue_arrays1,
            )

        self.add_work(
            "make couples arrays",
            func_or_cls=self.make_couple,
            params_cls=None,
            input_queue=(queue_series_names_couples, queue_arrays),
            output_queue=queue_array_couples,
            kind="global",
        )

        self.add_work(
            "couples -> bos",
            func_or_cls=self.calcul,
            params_cls=params,
            input_queue=queue_array_couples,
            output_queue=queue_bos,
        )

        self.add_work(
            "save bos",
            func_or_cls=self.save_bos_object,
            input_queue=queue_bos,
            kind="io",
        )

    def save_bos_object(self, o):
        ret = o.save(self.path_dir_result)
        return ret

    def calcul(self, array_couple):
        return WorkPIV(self.params).calcul(array_couple)

    def fill_series_name_couple_and_path(self, input_queue, output_queues):
        queue_series_name_couple = output_queues[0].queue
        queue_path = output_queues[1].queue

        series = self.series
        if len(series) == 0:
            logger.warning("add 0 couple. No bos to compute.")
            return
        if self.how_saving == "complete":
            names = []
            index_series = []
            for i, serie in enumerate(series):
                name_bos = get_name_bos(serie, prefix="bos")
                if os.path.exists(os.path.join(self.path_dir_result, name_bos)):
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

        nb_series = len(series)
        print("Add {} bos fields to compute.".format(nb_series))

        first_array_name = self.series.get_serie_from_index(1).filename_given
        self.first_array = imread(
            os.path.join(self.params.series.path, first_array_name)
        )
        for i, serie in enumerate(series):
            inew = i * self.series.ind_step + series.ind_start
            queue_series_name_couple[inew] = (
                first_array_name,
                serie.get_name_arrays()[1],
            )
            queue_path[serie.get_name_arrays()[1]] = serie.get_path_files()[1]
        try:
            del queue_path[first_array_name]
        except:
            pass
        print(queue_path)

    def make_couple(self, input_queues, output_queue):
        # for readablity
        queue_series_name_couple = input_queues[0].queue
        queue_array = input_queues[1].queue

        try:
            params_mask = self.params.mask
        except AttributeError:
            params_mask = None
        # for each name couple
        for key, couple in queue_series_name_couple.items():
            # if corresponding arrays are available, make an array couple
            if couple[1] in queue_array.keys():
                array2 = queue_array[couple[1]]
                serie = copy.copy(self.series.get_serie_from_index(key))
                paths = self.params.series.path

                array_couple = ArrayCoupleBOS(
                    names=(couple[0], couple[1]),
                    arrays=(self.first_array, array2),
                    params_mask=params_mask,
                    paths=paths,
                    serie=serie,
                )
                output_queue.queue[key] = array_couple
                del queue_series_name_couple[key]
                # remove the image_array if it will not be used anymore
                if not self.still_is_in_dict(couple[1], queue_series_name_couple):
                    del queue_array[couple[1]]
                return True
        return False

    @staticmethod
    def still_is_in_dict(image_name, dict):
        for key, names in dict.items():
            if image_name in names:
                return True
        return False

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


if "sphinx" in sys.modules:
    params = TopologyBOS.create_default_params()

    __doc__ += params._get_formatted_docs()

if __name__ == "__main__":
    params = TopologyBOS.create_default_params()
    params.series.path = "../../../image_samples/Karman/Images"
    params.series.ind_start = 1
    params.series.ind_step = 2

    params.piv0.shape_crop_im0 = 32
    params.multipass.number = 2
    params.multipass.use_tps = False

    params.mask.strcrop = ":, 50:500"

    # params.saving.how = 'complete'
    params.saving.postfix = "bos_example2"

    topo = TopologyBOS(params, logging_level="info")

    topo.make_code_graphviz("tmp.dot")
