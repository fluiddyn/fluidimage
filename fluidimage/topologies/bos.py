"""Topology for BOS computation (:mod:`fluidimage.topologies.bos`)
==================================================================

NotImplementedError!

.. autoclass:: TopologyBOS
   :members:
   :private-members:

"""
import os
import json

from .. import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays

from .base import TopologyBase

# todo WaitingQueueMakeCoupleBOS
from .waiting_queues.base import (
    WaitingQueueMultiprocessing, WaitingQueueThreading,
    WaitingQueueMakeCoupleBOS, WaitingQueueLoadImage)

from ..works.piv import WorkPIV
# todo warning get_name_piv -> get_name_bos
from ..data_objects.piv import get_name_piv, set_path_dir_result, ArrayCouple
from ..util.util import logger


class TopologyBOS(TopologyBase):
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
        params = ParamContainer(tag='params')

        params._set_child('series', attribs={'path': '',
                                             'strslice': None,
                                             'ind_start': 0,
                                             'ind_stop': None,
                                             'ind_step': 1})

        params.series._set_doc("""
Parameters indicating the input series of images.

path : str, {''}

    String indicating the input images (can be a full path towards an image
    file or a string given to `glob`).

strslice : None

    String indicating as a Python slicing how series of images are formed.
    See the parameters the PIV topology.

ind_start : int, {0}

ind_step : int, {1}

int_stop : None

""")

        params._set_child('saving', attribs={'path': None,
                                             'how': 'ask',
                                             'postfix': 'piv'})

        params.saving._set_doc(
            """Saving of the results.

path : None or str

    Path of the directory where the data will be saved. If None, the path is
    obtained from the input path and the parameter `postfix`.

how : str {'ask'}

    'ask', 'new_dir', 'complete' or 'recompute'.

postfix : str

    Postfix from which the output file is computed.
""")

        WorkPIV._complete_params_with_default(params)

        params._set_internal_attr(
            '_value_text',
            json.dumps({'program': 'fluidimage',
                        'module': 'fluidimage.topologies.bos',
                        'class': 'TopologyBOS'}))

        return params

    def __init__(self, params=None, logging_level='info', nb_max_workers=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.piv_work = WorkPIV(params)

        self.series = SeriesOfArrays(
            params.series.path, params.series.strslice,
            ind_start=params.series.ind_start,
            ind_stop=params.series.ind_stop,
            ind_step=params.series.ind_step)

        path_dir = self.series.serie.path_dir
        path_dir_result, self.how_saving = set_path_dir_result(
            path_dir, params.saving.path,
            params.saving.postfix, params.saving.how)

        self.path_dir_result = path_dir_result

        self.results = {}

        def save_piv_object(o):
            ret = o.save(path_dir_result)
            return ret

        self.wq_piv = WaitingQueueThreading(
            'delta', save_piv_object, self.results, topology=self)
        self.wq_couples = WaitingQueueMultiprocessing(
            'couple', self.piv_work.calcul, self.wq_piv,
            topology=self)
        self.wq_images = WaitingQueueMakeCouple(
            'array image', self.wq_couples, topology=self)
        self.wq0 = WaitingQueueLoadImage(
            destination=self.wq_images,
            path_dir=path_dir, topology=self)

        super(TopologyBOS, self).__init__(
            [self.wq0, self.wq_images, self.wq_couples, self.wq_piv],
            path_output=path_dir_result, logging_level=logging_level,
            nb_max_workers=nb_max_workers)

        self.add_series(self.series)

    def add_series(self, series):

        if len(series) == 0:
            logger.warning('add 0 image. No BOS to compute.')
            return

        names = series.get_name_all_arrays()

        raise NotImplementedError
        
        if self.how_saving == 'complete':
            names = []
            index_series = []
            for i, serie in enumerate(series):
                name_bos = get_name_bos(serie, prefix='bos')
                if os.path.exists(os.path.join(
                        self.path_dir_result, name_bos)):
                    continue
                for name in serie.get_name_arrays():
                    if name not in names:
                        names.append(name)

                index_series.append(i * series.ind_step + series.ind_start)

            if len(index_series) == 0:
                logger.warning(
                    'topology in mode "complete" and work already done.')
                return

            series.set_index_series(index_series)

            logger.debug(repr(names))
            logger.debug(repr([serie.get_name_arrays() for serie in series]))
        else:
            names = series.get_name_all_arrays()

        nb_series = len(series)
        print('Add {} PIV fields to compute.'.format(nb_series))

        for i, serie in enumerate(series):
            if i > 1:
                break
            print('Files of serie {}: {}'.format(i, serie.get_name_arrays()))

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
            names=('', ''), arrays=(im, im), params_mask=params_mask)
        im, _ = couple.get_arrays()

        self.piv_work._prepare_with_image(im)

    def _print_at_exit(self, time_since_start):

        txt = 'Stop compute after t = {:.2f} s'.format(time_since_start)
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += (' ({} bos fields, {:.2f} s/field).'.format(
                nb_results, time_since_start / nb_results))
        else:
            txt += '.'

        txt += '\npath results:\n' + self.path_dir_result

        print(txt)

params = TopologyBOS.create_default_params()

__doc__ += params._get_formatted_docs()
