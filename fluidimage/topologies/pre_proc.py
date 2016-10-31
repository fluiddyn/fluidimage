"""Topology for preprocessing images (:mod:`fluidimage.topologies.pre_proc`)
============================================================================
To preprocess series of images using multiprocessing and waiting queues.

.. currentmodule:: fluidimage.topologies.pre_proc

Provides:

.. autoclass:: TopologyPreproc
   :members:
   :private-members:

"""

import os

from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluiddyn.io.image import imread

from ..works.pre_proc import WorkPreproc

from ..data_objects.piv import set_path_dir_result
from ..data_objects.display import DisplayPreProc
from ..data_objects.pre_proc import get_name_preproc

from ..util.util import logger

from .base import TopologyBase

from .waiting_queues.base import (
    WaitingQueueMultiprocessing, WaitingQueueThreading)
from .waiting_queues.series import (
    WaitingQueueMakeSerie, WaitingQueueLoadImageSeries)


class TopologyPreproc(TopologyBase):
    """Preprocess series of images and provides interface for I/O and
    multiprocessing.

    """

    @classmethod
    def create_default_params(cls):
        params = WorkPreproc.create_default_params()
        params.preproc.series._set_attribs({'strcouple': 'i:i+1',
                                            'ind_start': 0,
                                            'ind_stop': None,
                                            'ind_step': 1,
                                            'sequential_loading': True})

        params.preproc.series._set_doc("""
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

""")

        params.preproc._set_child('saving', attribs={'path': None,
                                                     'strcouple': None,
                                                     'how': 'ask',
                                                     'format': 'img',
                                                     'postfix': 'pre'})

        params.preproc.saving._set_doc("""
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

""")

        return params

    def __init__(self, params=None, logging_level='info', nb_max_workers=None):
        if params is None:
            params = self.__class__.create_default_params()

        self.params = params.preproc
        self.preproc_work = WorkPreproc(params)
        serie_arrays = self.preproc_work.serie_arrays
        self.series = SeriesOfArrays(
            serie_arrays, params.preproc.series.strcouple,
            ind_start=params.preproc.series.ind_start,
            ind_stop=params.preproc.series.ind_stop,
            ind_step=params.preproc.series.ind_step)

        self.nb_items_per_serie = serie_arrays.get_nb_files()

        path_dir = params.preproc.series.path
        self.path_dir_result, self.how_saving = set_path_dir_result(
            path_dir, params.preproc.saving.path,
            params.preproc.saving.postfix, params.preproc.saving.how)

        self.params.saving.path = self.path_dir_result
        self.results = self.preproc_work.results
        self.display = self.preproc_work.display

        def save_preproc_results_object(o):
            return o.save(path=self.path_dir_result)

        self.wq_preproc = WaitingQueueThreading(
            'save results', save_preproc_results_object,
            self.results, work_name='save', topology=self)

        self.wq_serie = WaitingQueueMultiprocessing(
            'apply preprocessing', self.preproc_work.calcul,
            self.wq_preproc, work_name='preprocessing', topology=self)

        self.wq_images = WaitingQueueMakeSerie(
            'make serie', self.wq_serie, topology=self)

        self.wq0 = WaitingQueueLoadImageSeries(
            destination=self.wq_images, path_dir=path_dir, topology=self,
            sequential=params.preproc.series.sequential_loading)

        super(TopologyPreproc, self).__init__(
            [self.wq0, self.wq_images, self.wq_serie, self.wq_preproc],
            path_output=self.path_dir_result, logging_level=logging_level,
            nb_max_workers=nb_max_workers)

        self.add_series(self.series)

    def add_series(self, series):

        if len(series) == 0:
            logger.warning(
                'encountered empty series. No images to preprocess.')
            return

        if self.how_saving == 'complete':
            names = []
            index_series = []
            for i, serie in enumerate(series):
                names_serie = serie.get_name_files()
                name_preproc = get_name_preproc(
                    serie, names_serie, i, series.nb_series,
                    self.params.saving.format)
                if os.path.exists(os.path.join(
                        self.path_dir_result, name_preproc)):
                    continue

                for name in names_serie:
                    if name not in names:
                        names.append(name)

                index_series.append(i + series.ind_start)

            if len(index_series) == 0:
                logger.warning('topology in mode "complete" and '
                               'work already done.')
                return

            series.set_index_series(index_series)

            logger.debug(repr(names))
            logger.debug(repr([serie.get_name_files() for serie in series]))
        else:
            names = series.get_name_all_files()

        print('Add {} image series to compute.'.format(len(series)))

        self.wq0.add_name_files(names)
        self.wq_images.add_series(series)

        k, o = self.wq0.popitem()
        im = self.wq0.work(o)
        self.wq0.fill_destination(k, im)

    def compare(self, indices=[0, 1], suffix=None, hist=False):
        if not suffix:
            suffix = '.' + self.params.saving.postfix
        pathbase = self.params.series.path + '/'

        im0 = imread(pathbase + self.series.get_name_all_files()[indices[0]])
        im1 = imread(pathbase + self.series.get_name_all_files()[indices[1]])
        im0p = imread(pathbase[:-1] + suffix + '/' +
                      self.series.get_name_all_files()[indices[0]])
        im1p = imread(pathbase[:-1] + suffix + '/' +
                      self.series.get_name_all_files()[indices[1]])
        return DisplayPreProc(
            im0, im1, im0p, im1p, hist=hist)
