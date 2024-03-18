"""Image to image processing

Provides

.. autofunction:: init_im2im_function

.. autofunction:: complete_im2im_params_with_default

.. autofunction:: get_im2im_function_from_params

.. autoclass:: WorkImage2Image
   :members:
   :private-members:

"""

import types

from fluiddyn.util import import_class
from fluidimage.data_objects.display_pre import DisplayPreProc
from fluidimage.works import BaseWorkFromImage


def complete_im2im_params_with_default(params):
    """Complete params for image-to-image processing."""

    params._set_attribs({"im2im": None, "args_init": tuple()})

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


def get_im2im_function_from_params(params_im2im):
    """Helper for other topologies"""
    _, im2im_func = init_im2im_function(
        im2im=params_im2im.im2im, args_init=params_im2im.args_init
    )
    return im2im_func


class WorkImage2Image(BaseWorkFromImage):
    """Work for image to image processing"""

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)
        complete_im2im_params_with_default(params)

    def __init__(self, params):
        super().__init__(params)
        self._init_serie()

        self.im2im_obj, self.im2im_func = init_im2im_function(
            im2im=params.im2im, args_init=params.args_init
        )

    def calcul(self, tuple_image_name):
        return self.im2im_func(tuple_image_name)

    def display(self, ind=0, hist=False):
        """Display figures to study the preprocessing"""

        tuple_image_name0 = self.get_tuple_image_name(ind)
        tuple_image_name1 = self.get_tuple_image_name(ind + 1)

        arr_input0, _ = tuple_image_name0
        arr_input1, _ = tuple_image_name1

        arr_output0 = self.calcul(tuple_image_name0[::-1])
        arr_output1 = self.calcul(tuple_image_name1[::-1])

        return DisplayPreProc(
            arr_input0, arr_input1, arr_output0, arr_output1, hist=hist
        )


Work = WorkImage2Image
