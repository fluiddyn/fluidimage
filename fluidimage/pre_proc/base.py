"""Preprocess base (:mod:`fluidimage.pre_proc.base`)
====================================================
To preprocess series of images.

.. currentmodule:: fluidimage.pre_proc.base

Provides:

.. autoclass:: PreprocBase
   :members:
   :private-members:

"""
import os
from .toolbox import PreprocTools
from .. import ParamContainer, SerieOfArraysFromFiles
from fluidimage.data_objects.display import DisplayPreProc


class PreprocBase(object):
    """Preprocess series of images with various tools. """

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters."""
        params = ParamContainer(tag='params')
        params._set_child('preproc')
        params.preproc._set_child('series', attribs={'path': ''})

        PreprocTools._complete_class_with_tools(params)

        return params

    def __init__(self, params=None):
        """Set path for results and loads images as SeriesOfArraysFromFiles."""
        if params is None:
            params = self.__class__.create_default_params()

        self.params = params.preproc

        path = params.preproc.series.path
        if not os.path.exists(path):
            path = params.preproc.series.path = os.path.expandvars(path)

        self.serie_arrays = SerieOfArraysFromFiles(path)
        self.tools = PreprocTools(params)
        self.results = {}

    def __call__(self):
        """Apply all enabled preprocessing tools on the series of arrays
        and saves them in self.results.

        """
        name_files = self.serie_arrays.get_name_files()
        for i, img in enumerate(self.serie_arrays.iter_arrays()):
            name = name_files[i]
            self.results[name] = self.tools(img)

    def display(self, ind=None, hist=False, results=None):
        nimages = 2
        if not ind:
            name_files = self.serie_arrays.get_name_files()[:nimages]
        else:
            name_files = self.serie_arrays.get_name_files()[ind:ind + nimages]

        before = {}
        for fname in name_files:
            before[fname] = self.serie_arrays.get_array_from_name(fname)

        if results is None:
            results = self.results

        return DisplayPreProc(
            before[name_files[0]], before[name_files[1]],
            results[name_files[0]], results[name_files[1]],
            hist=hist)
