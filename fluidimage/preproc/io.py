"""Handles I/O of images to tools(:mod:`fluidimage.preproc.io`)
===============================================================

Provides decorators to handle input and output of images for tools/functions in
:mod:`fluidimage.preproc.toolbox`.

"""

import numpy as np
from decorator import decorator


def _get_img_arg(*args, **kwargs):
    if "img" in kwargs:
        return kwargs["img"]

    else:
        return args[0]


def _replace_img_arg(new_img, *args, **kwargs):
    if not isinstance(args, list):
        args = list(args)

    if "img" in kwargs:
        kwargs["img"] = new_img
    else:
        args[0] = new_img

    return args, kwargs


def _get_array_like_type(array_like, convert_to_ndarray=False):
    if not isinstance(array_like, np.ndarray):
        ArrayLike = array_like.__class__
    else:
        ArrayLike = np.array

    if convert_to_ndarray:
        return np.array(array_like), ArrayLike

    else:
        return ArrayLike


@decorator
def iterate_multiple_imgs(tool, *args, **kwargs):
    """
    Feeds one image at a time to the function `tool`,
    typically a spatial filter, or a brightness/contrast adjustment tool.

    """
    img_array_in = _get_img_arg(*args, **kwargs)

    if isinstance(img_array_in, np.ndarray):
        if img_array_in.ndim == 2:
            return tool(*args, **kwargs)  # Function call!

    for i, img in enumerate(img_array_in):
        args, kwargs = _replace_img_arg(img, *args, **kwargs)
        img_array_in[i] = tool(*args, **kwargs)  # Function call!

    return img_array_in


@decorator
def multiple_imgs_as_ndarray(tool, *args, **kwargs):
    """
    Images are processed as follows, esp. for temporal filters:
    .. array-like (input) --> nd-array ---> [`tool`] --> array_like (output)

    """
    img_array_in = _get_img_arg(*args, **kwargs)

    if isinstance(img_array_in, np.ndarray) and img_array_in.ndim == 3:
        return tool(*args, **kwargs)  # Function call!

    img_ndarray_in, ImgArrayLike = _get_array_like_type(img_array_in, True)
    args, kwargs = _replace_img_arg(img_ndarray_in, *args, **kwargs)

    img_array_out = tool(*args, **kwargs)  # Function call!
    return ImgArrayLike(img_array_out)
