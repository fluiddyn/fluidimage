"""Preprocess toolbox (:mod:`fluidimage.pre_proc._toolbox_cv`)
==============================================================
A toolbox for preprocessing images. Based on OpenCV library.
cf. http://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/imgproc/doc/filtering.html

.. currentmodule:: fluidimage.pre_proc._toolbox_cv

Provides:

   :members:

"""

import numpy as np

from .io import iterate_multiple_imgs
from .._opencv import cv2

__all__ = [
    "sliding_median",
    "sliding_minima",
    "global_threshold",
    "adaptive_threshold",
]


# ----------------------------------------------------
#   SPATIAL FILTERS
# ----------------------------------------------------


@iterate_multiple_imgs
def sliding_median(img=None, weight=1.0, window_size=3):
    """
    Subtracts the median calculated within a sliding window from the centre of
    the window.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    weight : scalar
        Fraction of median to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0,1.0).
    window_size : int
        Sets the size of the sliding window.

    """
    try:
        img_out = img - weight * cv2.medianBlur(
            img.astype(np.uint8), int(window_size)
        )
    except:
        print(
            "Check img dtype={}, shape={}, and window_size={}".format(
                img.dtype, img.shape, window_size
            )
        )
        raise

    return img_out


@iterate_multiple_imgs
def sliding_minima(
    img=None, weight=1.0, window_size=3, boundary_condition="reflect"
):
    """
    Subtracts the minimum calculated within a sliding window from the centre of
    the window.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    weight : scalar
        Fraction of minima to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0,1.0).
    window_size : scalar
        Sets the size of the sliding window.

    boundary_condition : {'reflect', 'default', 'constant', 'wrap',
                          'transparent', 'replicate'}

        Mode of handling array borders.

    """
    kernel = np.ones((window_size, window_size), np.uint8)
    border = getattr(cv2, f"BORDER_{boundary_condition.upper()}")
    img_out = img - weight * cv2.erode(
        img.astype(np.uint8), kernel=kernel, borderType=border
    )
    return img_out


@iterate_multiple_imgs
def global_threshold(img=None, minima=0.0, maxima=65535.0):
    """
    Trims pixel intensities which are outside the interval (minima, maxima).

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object

    minima, maxima : float
        Sets the threshold

    """
    MAX_OPER = cv2.THRESH_TRUNC
    MIN_OPER = cv2.THRESH_TOZERO
    img_out = cv2.threshold(img, thresh=maxima, maxval=maxima, type=MAX_OPER)
    img_out = cv2.threshold(img_out, thresh=minima, maxval=maxima, type=MIN_OPER)
    return img_out


@iterate_multiple_imgs
def adaptive_threshold(img=None, window_size=5, offset=0):
    """
    Adaptive threshold transforms a grayscale image to a binary image.
    Useful in identifying particles.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    window_size : scalar
        Sets the size of the pixel neighbourhood to calculate threshold.
    offset : scalar
        Constant to be subtracted from the mean.

    """
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresholdType = cv2.THRESH_BINARY  # cv2.THRESH_BINARY_INV
    img_out = cv2.adaptiveThreshold(
        img, img.max(), adaptiveMethod, thresholdType, window_size, offset
    )
    return img_out
