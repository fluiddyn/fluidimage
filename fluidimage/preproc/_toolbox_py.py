"""Preprocess toolbox (:mod:`fluidimage.preproc._toolbox_py`)
==========================================================
A toolbox for preprocessing images.
Utilizes functions available from `scipy.ndimage` and `skimage` packages.
cf. http://www.scipy-lectures.org/advanced/image_processing/

Provides:

   :members:

"""

import numpy as np
import scipy.ndimage as ndi

from .io import iterate_multiple_imgs, multiple_imgs_as_ndarray

try:
    import skimage
    from skimage import exposure, filters, morphology

    if skimage.__version__ < "0.13.0":
        print(
            "Warning: to use fluidimage.preproc, "
            "first upgrade scikit-image to a version >= 0.13.0."
        )
except ImportError:
    print(
        "Warning: ImportError, to use fluidimage.preproc, "
        "first install scikit-image >= 0.13.0"
    )


__all__ = [
    "sliding_median",
    "sliding_minima",
    "sliding_percentile",
    "temporal_median",
    "temporal_minima",
    "temporal_percentile",
    "global_threshold",
    "adaptive_threshold",
    "rescale_intensity",
    "equalize_hist_global",
    "equalize_hist_local",
    "equalize_hist_adapt",
    "gamma_correction",
    "sharpen",
    "rescale_intensity_tanh",
]


def imstats(img, hist_bins=256):
    histogram = ndi.measurements.histogram(img, bins=hist_bins)
    return histogram


# ----------------------------------------------------
#   SPATIAL FILTERS
# ----------------------------------------------------


@iterate_multiple_imgs
def sliding_median(
    img=None, weight=1.0, window_size=30, boundary_condition="reflect"
):
    """
    Subtracts the median calculated within a sliding window from the centre of
    the window.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    weight : scalar
        Fraction of median to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0, 1.0).
    window_size : scalar or tuple
        Sets the size of the sliding window.
        Specifying `window_size=3` is equivalent to `window_size=(3,3)`.
    boundary_condition : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        Mode of handling array borders.

    """
    img_out = img - weight * ndi.median_filter(
        img, size=window_size, mode=boundary_condition
    )
    img_out[img_out < 0] = 0
    return img_out


@iterate_multiple_imgs
def sliding_percentile(
    img=None,
    percentile=10.0,
    weight=1.0,
    window_size=30,
    boundary_condition="reflect",
):
    """
    Flexible version of median filter. Low percentile values work well
    for dense images.

    Parameters
    ----------
    img : array_like
        Series of images as a 3D numpy array, or a list or a set
    percentile : scalar
        Percentile to filter. Setting `percentile = 50` is equivalent
        to a `sliding_median` filter.
    weight : scalar
        Fraction of median to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0, 1.0).
    window_shape : tuple of integers
        Specifies the shape of the window as follows (dt, dy, dx)
    boundary_condition : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        Mode of handling array borders.

    """
    img_out = img - weight * ndi.percentile_filter(
        img, percentile, size=window_size, mode=boundary_condition
    )
    img_out[img_out < 0] = 0
    return img_out


@iterate_multiple_imgs
def sliding_minima(
    img=None, weight=1.0, window_size=30, boundary_condition="reflect"
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
    window_size : scalar or tuple
        Sets the size of the sliding window.
        Specifying `window_size=3` is equivalent to `window_size=(3,3)`.
    boundary_condition : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        Mode of handling array borders.

    """
    img_out = img - weight * ndi.minimum_filter(
        img, size=window_size, mode=boundary_condition
    )
    img_out[img_out < 0] = 0
    return img_out


# ----------------------------------------------------
#   SPATIO-TEMPORAL FILTERS
# ----------------------------------------------------


def _calcul_windowshape(arr_shape, window_shape):
    nb_imgs = arr_shape[0]
    if len(arr_shape) <= 2 or nb_imgs <= 1:
        raise ValueError(
            "Need more than one image to apply temporal filtering "
            "(use sliding filter?)."
        )

    if window_shape is None:
        window_shape = (nb_imgs, 1, 1)
    elif isinstance(window_shape, int):
        window_shape = (window_shape, 1, 1)
    elif not isinstance(window_shape, (tuple, list)):
        raise ValueError("window_shape must be a tuple or a list.")

    elif len(window_shape) == 2:
        window_shape = (nb_imgs,) + window_shape
    elif window_shape[0] <= 1:
        raise ValueError(
            "Cannot perform temporal filtering, try spatial filtering."
        )

    return window_shape


@multiple_imgs_as_ndarray
def temporal_median(img=None, weight=1.0, window_shape=None):
    """
    Subtracts the median calculated in time and space, for each pixel.
    Median filter works well for sparse images.

    Parameters
    ----------
    img : array_like
        Series of images as a 3D numpy array, or a list or a set
    weight : scalar
        Fraction of median to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0,1.0).
    window_shape : tuple of integers
        Specifies the shape of the window as follows (nt, ny, nx) or (ny, nx).

    """
    window_shape = _calcul_windowshape(img.shape, window_shape)

    img_out = img - weight * ndi.median_filter(img, size=window_shape)
    return img_out


@multiple_imgs_as_ndarray
def temporal_percentile(img=None, percentile=10.0, weight=1.0, window_shape=None):
    """
    Flexible version of median filter. Low percentile values work well
    for dense images.

    Parameters
    ----------
    img : array_like
        Series of images as a 3D numpy array, or a list or a set
    percentile : scalar
        Percentile to filter. Setting `percentile = 50` is equivalent
        to a `temporal_median` filter.
    weight : scalar
        Fraction of median to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0,1.0).
    window_shape : tuple of integers
        Specifies the shape of the window as follows (nt, ny, nx) or (ny, nx).

    """
    window_shape = _calcul_windowshape(img.shape, window_shape)

    img_out = img - weight * ndi.percentile_filter(
        img, percentile, size=window_shape
    )
    return img_out


@multiple_imgs_as_ndarray
def temporal_minima(img=None, weight=1.0, window_shape=None):
    """Subtracts the minima calculated in time and space, for each pixel.

    Parameters
    ----------
    imgs : array_like
        Series of images as a 3D numpy array, or a list or a set
    weight : scalar
        Fraction of minima to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0,1.0).
    window_shape : tuple of integers
        Specifies the shape of the window as follows (nt, ny, nx) or (ny, nx).

    """
    window_shape = _calcul_windowshape(img.shape, window_shape)
    img_out = img - weight * ndi.minimum_filter(img, size=window_shape)
    return img_out


# ----------------------------------------------------
#   BRIGHTNESS / CONTRAST TOOLS
# ----------------------------------------------------


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
    img_out = img.copy()
    img_out[img_out < minima] = minima
    img_out[img_out > maxima] = maxima
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
    img_out = filters.threshold_local(img, window_size, offset=offset)
    return img_out


@iterate_multiple_imgs
def rescale_intensity(img=None, minima=0.0, maxima=4096):
    """
    Rescale image intensities, between the specified minima and maxima,
    by using a multiplicative factor.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    minima, maxima : float
        Sets the range to which current intensities have to be rescaled.

    """
    out_range = (minima, maxima)
    img_out = exposure.rescale_intensity(img, out_range=out_range)
    return img_out


@iterate_multiple_imgs
def rescale_intensity_tanh(img=None, threshold=None):
    """
    Rescale image intensities, using a tanh fit. The maximum intensity of the
    output is set by the threshold parameter.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    threshold:
        Value of intensity with which img is normalized
        img_out = max(img) * tanh( img / threshold)
        If threshold is None: threshold = 2 * np.sqrt(np.mean(img**2))

    """
    if threshold is None:
        threshold = 2 * np.sqrt(np.mean(img ** 2))

    if threshold == 0:
        return img

    img_out = np.tanh(img / threshold)
    img_out = np.floor(np.max(img) / np.max(img_out) * img_out)
    return img_out


@iterate_multiple_imgs
def equalize_hist_global(img=None, nbins=256):
    """Increases global contrast of the image. Equalized image would have a
    roughly linear cumulative distribution function for each pixel
    neighborhood. It works well when pixel intensities are nearly uniform
    [1,2].

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    nbins : integer
        Number of bins to calculate histogram

    References
    ----------
    - http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html
    - https://en.wikipedia.org/wiki/Histogram_equalization

    """
    img_out = exposure.equalize_hist(img, nbins=nbins, mask=None)
    return img_out


@iterate_multiple_imgs
def equalize_hist_adapt(img=None, window_shape=(10, 10), nbins=256):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Increases local contrast.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    window_shape : tuple of integers
        Specifies the shape of the window as follows (dx, dy)
    nbins : integer
        Number of bins to calculate histogram

    References
    ----------
    - http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html
    - https://en.wikipedia.org/wiki/Histogram_equalization

    """
    minimum = img.min()
    maximum = img.max()
    img = rescale_intensity(img, 0, 1)
    img = exposure.equalize_adapthist(img, kernel_size=window_shape, nbins=nbins)
    img_out = rescale_intensity(img, minimum, maximum)
    return img_out


@iterate_multiple_imgs
def equalize_hist_local(img=None, radius=10):
    """
    Adaptive histogram equalization (AHE) emphasizes every local graylevel variations [1].
    Caution: It has a tendency to overamplify noise in homogenous regions [2].

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    radius : integer
        Radius of the disk shaped window.

    References
    ----------
    - http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html
    - https://en.wikipedia.org/wiki/Adaptive_histogram_equalization

    """
    selem = morphology.disk(radius)
    minimum = img.min()
    maximum = img.max()
    img = rescale_intensity(img, 0, 1)
    img = filters.rank.equalize(img, selem, mask=None)
    img_out = rescale_intensity(img, minimum, maximum)
    return img_out


@iterate_multiple_imgs
def gamma_correction(img=None, gamma=1.0, gain=1.0):
    r"""
    Gamma correction or power law transform. It can be expressed as:

    .. math::
        I_{out} = gain \times {I_{in}} ^ {\gamma}

    Adjusts contrast without changing the shape of the histogram. For the values
    .. :math:`\gamma > 1` : Histogram shifts towards left (darker)
    .. :math:`\gamma < 1` : Histogram shifts towards right (lighter)

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    gamma : float
        Non-negative real number
    gain : float
        Multiplying factor

    References
    ----------
    - http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_log_gamma.html
    - http://en.wikipedia.org/wiki/Gamma_correction

    """
    img_out = exposure.adjust_gamma(img, gamma, gain)
    return img_out


@iterate_multiple_imgs
def sharpen(img=None, sigma1=3.0, sigma2=1.0, alpha=30.0):
    """
    Sharpen image edges.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    sigma1, sigma2 : float
        Std deviation for two passes gaussian filters. sigma1 > sigma2
    alpha : float
        Factor by which the image will be sharpened

    """
    blurred = ndi.gaussian_filter(img, sigma1)
    filter_blurred = ndi.gaussian_filter(blurred, sigma2)

    img_out = blurred + alpha * (blurred - filter_blurred)
    return img_out
