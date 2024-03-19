"""Works - processing
=====================

This subpackage defines some works. A work does a processing. It has
initialization parameters and after initialization is able to produce an output
object from an input object. It can also take more than one input objects
and/or return more than one output objects.

A work is made of one or more work units. In particular, it could be useful to
define input/output and computational works.

.. autosummary::
   :toctree:

   image2image
   piv
   preproc
   bos
   surface_tracking
   optical_flow
   with_mask

Provides:

.. autoclass:: BaseWork
   :members:
   :private-members:

.. autoclass:: BaseWorkFromSerie
   :members:
   :private-members:

.. autoclass:: BaseWorkFromImage
   :members:
   :private-members:

"""

from copy import deepcopy
from typing import Optional

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles, SeriesOfArrays

from .. import imread


class BaseWork:
    def __init__(self, params=None):
        self.params = params

    @classmethod
    def create_default_params(cls):
        "Create an object containing the default parameters (class method)."
        params = ParamContainer(tag="params")
        cls._complete_params_with_default(params)
        return params


class BaseWorkFromSerie(BaseWork):
    """Base class for work taking as argument a SerieOfArraysFromFiles"""

    _series: SeriesOfArrays

    @classmethod
    def _complete_params_with_default(cls, params):

        params._set_child(
            "series",
            attribs={
                "path": "",
                "str_subset": "pairs",
                "ind_start": "first",
                "ind_stop": None,
                "ind_step": 1,
            },
            doc="""
Parameters indicating the input series of images.

path : str, {''}

    String indicating the input images (can be a full path towards an image
    file or a string given to `glob`).

str_subset : 'pairs'

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

    >>> params.series.str_subset = 'i, 0:2'

    In this case, the first couple will be (im1a, im1b).

    To get the first couple (im1a, im1a), we would have to write

    >>> params.series.str_subset = 'i:i+2, 0'

ind_start : int, {'first'}

ind_step : int, {1}

int_stop : None

""",
        )

    def get_serie(self, index_serie: Optional[int] = None):
        """Get a serie as defined by params.series"""
        if not hasattr(self, "_series"):
            p_series = self.params.series
            self._series = SeriesOfArrays(
                p_series.path,
                p_series.str_subset,
                ind_start=p_series.ind_start,
                ind_stop=p_series.ind_stop,
                ind_step=p_series.ind_step,
            )

        if index_serie is None:
            index_serie = self._series.ind_start

        return deepcopy(self._series.get_serie_from_index(index_serie))

    def process_1_serie(self, index_serie: Optional[int] = None):
        """Process one serie and return the result"""
        return self.calcul(self.get_serie(index_serie))

    def calcul_from_arrays(self, *arrays, names=None):
        """Calcul from images"""
        names = [f"array{i}" for i in range(len(arrays))]
        return self.calcul({"arrays": arrays, "names": names})


class BaseWorkFromImage(BaseWork):
    """Base class for work taking as argument an image"""

    serie: SerieOfArraysFromFiles

    @classmethod
    def _complete_params_with_default(cls, params):

        params._set_child("images", attribs={"path": "", "str_subset": None})

        params.images._set_doc(
            """
Parameters indicating the input image set.

path : str, {''}

    String indicating the input images (can be a full path towards an image
    file or a string given to `glob`).

str_subset : None

    String indicating as a Python slicing how to select images from the serie of
    images on the disk. If None, no selection so all images will be processed.

"""
        )

    def _init_serie(self):
        p_images = self.params.images
        self.serie = SerieOfArraysFromFiles(p_images.path, p_images.str_subset)
        return self.serie

    def get_tuple_image_name(self, index_image: int = 0):
        """Get an image and its name"""
        if not hasattr(self, "serie"):
            self._init_serie()

        return self.serie.get_tuple_array_name_from_index(index_image)

    def process_1_image(self, index_serie: int = 0):
        """Process one serie and return the result"""
        return self.calcul(self.get_tuple_image_name(index_serie))


def load_image(path):
    im = imread(path)
    return im
