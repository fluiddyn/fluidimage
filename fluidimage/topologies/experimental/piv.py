"""Topology for PIV computation (:mod:`fluidimage.topologies.piv`)
==================================================================

.. autoclass:: TopologyPIV
   :members:
   :private-members:

"""
import os
import json
import sys

from ... import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays

from .base import TopologyBase


from .. import prepare_path_dir_result
from ...works.piv import WorkPIV
from ...data_objects.piv import get_name_piv, ArrayCouple
from ...util.util import logger, imread
from .. import image2image
import scipy.io


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

        super(TopologyPIV, self).__init__(
            path_output=path_dir_result,
            logging_level=logging_level,
            nb_max_workers=nb_max_workers,
        )

        queue_names_piv = self.add_queue("names piv")
        queue_names_couples = self.add_queue("names couples")
        queue_paths = self.add_queue("paths")
        queue_arrays0 = queue_arrays1 = self.add_queue("arrays")
        queue_couples = self.add_queue("couples of arrays")
        queue_piv = self.add_queue("piv")

        if params.preproc.im2im is not None:
            queue_arrays1 = self.add_queue("arrays1")

        self.add_work(
            "fill names piv",
            func_or_cls=self.fill_name_piv,
            output_queue=queue_names_piv,
            kind=("global", "one shot"),
        )
        self.add_work(
            "fill (names couples, paths)",
            func_or_cls=self.fill_name_couple_and_path,
            input_queue=queue_names_piv,
            output_queue=(queue_names_couples, queue_paths),
            kind=("global", "one shot"),
        )
        self.add_work(
            "path -> arrays",
            func_or_cls=self.imread,
            input_queue=queue_paths,
            output_queue=queue_arrays0,
            kind="io",
        )

        if params.preproc.im2im is not None:
            im2im_func = image2image.TopologyImage2Image.init_im2im(
                self, params.preproc
            )

            self.add_work(
                "image -> image",
                func_or_cls=im2im_func,
                input_queue=queue_arrays0,
                output_queue=queue_arrays1,
            )

        self.add_work(
            "make couples arrays",
            func_or_cls=self.make_couple,
            params_cls=None,
            input_queue=(queue_arrays1, queue_names_couples),
            output_queue=queue_couples,
            kind="global",
        )

        self.add_work(
            "couples -> piv",
            func_or_cls = self.calcul,
            params_cls=params,
            input_queue=queue_couples,
            output_queue=queue_piv,
        )

        def save_piv_object(o):
            ret = o.save(path_dir_result)
            return ret

        self.add_work(
            "save piv",
            func_or_cls=self.save_piv,
            input_queue=queue_piv,
            kind="io",
        )
    def save_piv(self, input_queue, output_queue):
        print(input_queue.queue)
        if input_queue.queue:
            key, light_result = input_queue.queue.popitem()
            path_save = '../../../../fluidimage/image_samples/Karman/trio/resultsPIVTMP'+key
            scipy.io.savemat(
                path_save,
                mdict={
                    "deltaxs": light_result.deltaxs,
                    "deltays": light_result.deltays,
                    "xs": light_result.xs,
                    "ys": light_result.ys,
                },
            )
            print('###PIV {} SAVED !!!!!###'.format(key))

    def calcul(self, input_queue, output_queue):
        print(input_queue.queue)
        if input_queue.queue:
            key, array_couple = input_queue.queue.popitem()
            ret = WorkPIV(self.params).calcul(array_couple)
            lighPiv = ret.make_light_result()
            output_queue.queue[key] = lighPiv


    def imread(self, input_queue, output_queue):
        if input_queue.queue:
            key, path = input_queue.queue.popitem()
            output_queue.queue[key] = imread(path)
            print(output_queue.queue)

    def fill_name_piv(self, input_queue, output_queue):

        series = self.series
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

        print(type(output_queue))
        for name in names:
            output_queue.queue[name] = name

        # k, o = self.wq0.popitem()
        # im = self.wq0.work(o)
        # self.wq0.fill_destination(k, im)
        #
        # # a little bit strange, to apply mask...
        # try:
        #     params_mask = self.params.mask
        # except AttributeError:
        #     params_mask = None
        #
        # couple = ArrayCouple(
        #     names=("", ""), arrays=(im, im), params_mask=params_mask
        # )
        # im, _ = couple.get_arrays()
        #
        # self.piv_work._prepare_with_image(im)

    def fill_name_couple_and_path(self,input_queue, output_queues):
        previous_name = None
        input_queue = sorted(input_queue.queue)

        for name in input_queue:
            output_queues[1].queue[name[:-4]] = os.path.join(self.params.series.path,name)
            if previous_name is not None:
                output_queues[0].queue[previous_name] = (str(previous_name),name[:-4])
                previous_name = name[:-4]
            else:
                previous_name = name[:-4]
        return output_queues


    def make_couple(self, input_queue, output_queue):
        try:
            params_mask = self.params.mask
        except AttributeError:
            params_mask = None
        if ( input_queue[0].queue and input_queue[1].queue):
            key, couple = input_queue[1].queue.popitem() #pop a couple
            print(couple[0])
            if couple[0] in input_queue[0].queue and couple[1] in input_queue[0].queue:
                array1 = input_queue[0].queue[couple[0]]
                array2 = input_queue[0].queue[couple[1]]
                couple = ArrayCouple(
                    names=(couple[0], couple[1]),
                    arrays=(array1, array2),
                    params_mask=params_mask,
                )
                output_queue.queue[key] = couple
            else:
                input_queue[1].queue[key] = couple
        else:
            logger.error('Array or name couple is empty')

        print(output_queue.queue)
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


if "sphinx" in sys.modules:

    params = TopologyPIV.create_default_params()

    __doc__ += params._get_formatted_docs()


if __name__ == "__main__":
    params = TopologyPIV.create_default_params()

    params.series.path = "image_samples/Karman/Images"
    params.series.ind_start = 1
    params.series.ind_step = 2

    params.piv0.shape_crop_im0 = 32
    params.multipass.number = 2
    params.multipass.use_tps = False

    params.mask.strcrop = ":, 50:500"

    # params.saving.how = 'complete'
    params.saving.postfix = "piv_example"

    topo = TopologyPIV(params, logging_level="info")

    topo.make_code_graphviz("tmp.dot")
