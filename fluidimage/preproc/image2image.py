""""Image to image" processing (:mod:`fluidimage.preproc.image2image`)
======================================================================

.. autofunction:: init_im2im_function

.. autofunction:: complete_im2im_params_with_default

.. autofunction:: apply_im2im_filter

.. autofunction:: im2im_func_example

.. autoclass:: Im2ImExample
   :members:

"""

import types

import numpy as np

from fluiddyn.util import import_class
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles

from ..data_objects.piv import ArrayCouple


def im2im_func_example(tuple_image_path):
    """Process one image

    This is just an example to show how to write functions which can be plugged
    to the class
    :class:`fluidimage.topologies.image2image.TopologyImage2Image`.

    """
    image, path = tuple_image_path
    # the processing can be adjusted depending on the value of the path.
    print("process file:\n" + path)
    image_out = np.round(image * (255 / image.max())).astype(np.uint8)
    return image_out, path


class Im2ImExample:
    """Process one image

    This is just an example to show how to write classes which can be plugged
    to the class
    :class:`fluidimage.topologies.image2image.TopologyImage2Image`.

    """

    def __init__(self, arg0, arg1):
        print("init with arguments:", arg0, arg1)
        self.arg0 = arg0
        self.arg1 = arg1

    # time consuming tasks can be done here

    def calcul(self, tuple_image_path):
        """Method processing one image"""
        print(
            "calcul with arguments (unused in the example):", self.arg0, self.arg1
        )
        return im2im_func_example(tuple_image_path)


def complete_im2im_params_with_default(params):
    """Complete params for image-to-image processing."""

    params._set_attrib("im2im", None)
    params._set_attrib("args_init", tuple())

    params._set_doc(
        """
im2im : str {None}

    Function or class to be used to process the images.

args_init : object {None}

    An argument given to the init function of the class used to process the
    images.

"""
    )


def init_im2im_function(im2im=None, args_init=()):
    """Initialize the filter function."""

    if isinstance(im2im, str):
        str_package, str_obj = im2im.rsplit(".", 1)
        im2im = import_class(str_package, str_obj)

    if isinstance(im2im, types.FunctionType):
        obj = im2im
        im2im_func = im2im
    elif isinstance(im2im, type):
        print("in init_im2im", args_init)
        obj = im2im(*args_init)
        im2im_func = obj.calcul

    return obj, im2im_func


def apply_im2im_filter(serie, im2im=None, args_init=()):
    """Apply an image-to-image filter to a serie of images.

    Parameters
    ----------

    serie : :class:`fluiddyn.util.serieofarrays.SerieOfArraysFromFiles`

    im2im : Optional[str]

    args_init : tuple

    """
    if im2im is None:
        return serie

    obj, im2im_func = init_im2im_function(im2im, args_init)

    if not isinstance(serie, SerieOfArraysFromFiles):
        raise NotImplementedError

    arrays = serie.get_arrays()
    paths = serie.get_path_arrays()

    return tuple(im2im_func(t) for t in zip(arrays, paths))
