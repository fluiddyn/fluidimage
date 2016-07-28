"""Preprocess toolbox (:mod:`fluidimage.pre_proc.toolbox`)
==========================================================
A toolbox of filters which operate on a single image (numpy array).
cf. http://www.scipy-lectures.org/advanced/image_processing/

.. currentmodule:: fluidimage.pre_proc.toolbox

Provides:

   :members:

"""

import inspect

import numpy as np
import scipy.ndimage as nd
try:
    import skimage
    from skimage import exposure, filters, morphology
    if skimage.__version__ < '0.12.3':
        print('Warning: to use fluidimage.preproc, '
              'first upgrade scikit-image to a version >= 0.12.3.')
except ImportError:
    print('Warning: ImportError, to use fluidimage.preproc, '
          'first install scikit-image >= 0.12.3.')

from .io import iterate_multiple_imgs, multiple_imgs_as_ndarray
from ..util.util import logger


available_tools = ['sliding_median', 'sliding_minima', 'sliding_percentile',
                   'temporal_median', 'temporal_minima', 'temporal_percentile',
                   'global_threshold', 'rescale_intensity',
                   'equalize_hist_global', 'equalize_hist_local',
                   'equalize_hist_adapt',
                   'gamma_correction', 'sharpen', 'rescale_intensity_tanh']

__all__ = available_tools + ['PreprocTools']


def imstats(img, hist_bins=256):
    histogram = nd.measurements.histogram(bins=hist_bins)
    return histogram


# ----------------------------------------------------
#   SPATIAL FILTERS
# ----------------------------------------------------

@iterate_multiple_imgs
def sliding_median(img=None, weight=1., window_size=30,
                   boundary_condition='reflect'):
    '''
    Subtracts the median calculated within a sliding window from the centre of
    the window.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    weight : scalar
        Fraction of median to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0,1.0).
    window_size : scalar or tuple
        Sets the size of the sliding window.
        Specifying `window_size=3` is equivalent to `window_size=(3,3)`.
    boundary_condition : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        Mode of handling array borders.

    '''
    img_out = img - weight * nd.median_filter(img,
                                              size=window_size,
                                              mode=boundary_condition)
    return img_out


@iterate_multiple_imgs
def sliding_percentile(img=None, percentile=10., weight=1., window_size=30,
                       boundary_condition='reflect'):
    '''
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
        Value of `weight` should be in the interval (0.0,1.0).
    window_shape : tuple of integers
        Specifies the shape of the window as follows (dt, dy, dx)
    boundary_condition : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        Mode of handling array borders.

    '''
    img_out = img - weight * nd.percentile_filter(img,
                                                  percentile,
                                                  size=window_size,
                                                  mode=boundary_condition)
    return img_out


@iterate_multiple_imgs
def sliding_minima(img=None, weight=1., window_size=30,
                   boundary_condition='reflect'):
    '''
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

    '''
    img_out = img - weight * nd.minimum_filter(img,
                                               size=window_size,
                                               mode=boundary_condition)
    return img_out


# ----------------------------------------------------
#   SPATIO-TEMPORAL FILTERS
# ----------------------------------------------------

@multiple_imgs_as_ndarray
def temporal_median(img=None, weight=1., window_shape=None):
    '''
    Subtracts the median calculated in time, for each pixel.
    Median filter works well for sparse images.

    Parameters
    ----------
    img : array_like
        Series of images as a 3D numpy array, or a list or a set
    weight : scalar
        Fraction of median to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0,1.0).
    window_shape : tuple of integers
        Specifies the shape of the window as follows (dt, dy, dx)

    '''
    time_axis = 0
    nb_imgs = img.shape[time_axis]
    if img.ndim <= 2 or nb_imgs <= 1:
        raise ValueError(
            'Need more than one image to apply temporal filtering.')

    if window_shape is None:
        window_shape = (nb_imgs, 1, 1)
    elif not isinstance(window_shape, tuple):
        raise ValueError('window_shape must be a tuple.')
    elif window_shape[0] <= 1:
        raise ValueError(
            'Cannot perform temporal filtering, try spatial filtering.')

    img_out = img - weight * nd.median_filter(img,
                                              size=window_shape)
    return img_out


@multiple_imgs_as_ndarray
def temporal_percentile(img=None, percentile=10., weight=1.,
                        window_shape=None):
    '''
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
        Specifies the shape of the window as follows (dt, dy, dx)

    '''
    time_axis = 0
    nb_imgs = img.shape[time_axis]
    if img.ndim <= 2 or nb_imgs <= 1:
        raise ValueError(
            'Need more than one image to apply temporal filtering.')

    if window_shape is None:
        window_shape = (nb_imgs, 1, 1)
    elif not isinstance(window_shape, tuple):
        raise ValueError('window_shape must be a tuple.')
    elif window_shape[0] <= 1:
        raise ValueError(
            'Cannot perform temporal filtering, try spatial filtering.')

    img_out = img - weight * nd.percentile_filter(img,
                                                  percentile,
                                                  size=window_shape)
    return img_out


@multiple_imgs_as_ndarray
def temporal_minima(img=None, weight=1.):
    '''
    Subtracts the minima calculated in time,for each pixel.

    Parameters
    ----------
    imgs : array_like
        Series of images as a 3D numpy array, or a list or a set
    weight : scalar
        Fraction of minima to be subtracted from each pixel.
        Value of `weight` should be in the interval (0.0,1.0).

    '''
    time_axis = 0
    nb_imgs = img.shape[time_axis]
    if img.ndim < 3 or nb_imgs <= 1:
        raise ValueError(
            'Need more than one image to apply temporal filtering.')

    window_size = img.shape[time_axis]
    img_out = img - weight * nd.minimum_filter1d(img,
                                                 size=window_size,
                                                 axis=time_axis)
    return img_out


# ----------------------------------------------------
#   BRIGHTNESS / CONTRAST TOOLS
# ----------------------------------------------------

@iterate_multiple_imgs
def global_threshold(img=None, minima=0., maxima=65535.):
    '''
    Trims pixel intensities which are outside the interval (minima, maxima).

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object

    minima, maxima : float
        Sets the threshold

    '''
    img_out = img.copy()
    img_out[img_out < minima] = minima
    img_out[img_out > maxima] = maxima
    return img_out


@iterate_multiple_imgs
def rescale_intensity(img=None, minima=0., maxima=65535.):
    '''
    Rescale image intensities, between the specified minima and maxima,
    by using a multiplicative factor.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    minima, maxima : float
        Sets the range to which current intensities have to be rescaled.

    '''
    out_range = (minima, maxima)
    img_out = exposure.rescale_intensity(img, out_range=out_range)
    return img_out


@iterate_multiple_imgs
def rescale_intensity_tanh(img=None, threshold=None):
    '''
    Rescale image intensities, between the specified minima and maxima,
    by using a multiplicative factor.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    threshold:
        Value of intensity with which img is normalized
        img_out = max(img) * tanh( img / threshold)

    '''
    if not threshold:
        threshold = 2 * np.sqrt(np.mean(img**2))

    img_out = np.tanh(img/threshold)
    img_out = np.floor(np.max(img) / np.max(img_out) * img_out)
    return img_out


@iterate_multiple_imgs
def equalize_hist_global(img=None, nbins=256):
    '''
    Increases global contrast of the image. Equalized image would have a
    roughly linear cumulative distribution function for each pixel
    neighborhood. It works well when pixel intensities are nearly uniform [1,2]. 

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    nbins : integer
        Number of bins to calculate histogram

    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html # noqa
    .. [2] https://en.wikipedia.org/wiki/Histogram_equalization

    '''
    img_out = exposure.equalize_hist(img, nbins=nbins, mask=None)
    return img_out


@iterate_multiple_imgs
def equalize_hist_adapt(img=None, window_shape=(10, 10), nbins=256):
    '''
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
    .. [1] http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html # noqa
    .. [2] https://en.wikipedia.org/wiki/Histogram_equalization

    '''
    minimum = img.min()
    maximum = img.max()
    img = rescale_intensity(img, 0, 1)
    img = exposure.equalize_adapthist(img, kernel_size=window_shape,
                                      nbins=nbins)
    img_out = rescale_intensity(img, minimum, maximum)
    return img_out


@iterate_multiple_imgs
def equalize_hist_local(img=None, radius=10):
    '''
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
    .. [1] http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html # noqa
    .. [2] https://en.wikipedia.org/wiki/Adaptive_histogram_equalization

    '''
    selem = morphology.disk(radius)
    minimum = img.min()
    maximum = img.max()
    img = rescale_intensity(img, 0, 1)
    img = filters.rank.equalize(img, selem, mask=None)
    img_out = rescale_intensity(img, minimum, maximum)
    return img_out


@iterate_multiple_imgs
def gamma_correction(img=None, gamma=1., gain=1.):
    r'''
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
    .. [1] http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_log_gamma.html # noqa
    .. [2] http://en.wikipedia.org/wiki/Gamma_correction

    '''
    img_out = exposure.adjust_gamma(img, gamma, gain)
    return img_out


@iterate_multiple_imgs
def sharpen(img=None, sigma1=3., sigma2=1., alpha=30.):
    '''
    Sharpen image edges.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    sigma1, sigma2 : float
        Std deviation for two passes gaussian filters. sigma1 > sigma2
    alpha : float
        Factor by which the image will be sharpened

    '''
    blurred = nd.gaussian_filter(img, sigma1)
    filter_blurred = nd.gaussian_filter(blurred, sigma2)

    img_out = blurred + alpha * (blurred - filter_blurred)
    return img_out


class PreprocTools(object):
    """Wrapper class for functions in the current module."""

    @classmethod
    def _complete_class_with_tools(cls, params):
        """
        Dynamically add the global functions in this module as staticmethods of
        the present class. Also create default parameters from the function
        argument list.

        """
        params.preproc._set_child('tools')
        params = params.preproc.tools
        params._set_attribs({'available_tools': available_tools,
                             'sequence': None})

        for tool in available_tools:
            func = globals()[tool]

            # Add tools as `staticmethods` of the class
            setattr(PreprocTools, tool, func)

            # TODO: Replace with inspect.getfullargspec (Python >= 3).
            func_args = inspect.getcallargs(func)
            for arg in func_args.keys():
                if arg in ['img']:
                    # Remove arguments which are not parameters
                    del(func_args[arg])

            func_args.update({'enable': False})

            # New parameter child for each tool and parameter attributes
            # from its function arguments and default values
            params._set_child(tool, attribs=func_args)

            # Adds docstring to the parameter
            if func.func_doc is not None:
                enable_doc = 'enable : bool\n' + \
                             '        Set as `True` to enable the tool'
                params.__dict__[tool]._set_doc(func.func_doc + enable_doc)

    def __init__(self, params):
        self.params = params.preproc.tools

    def __call__(self, img):
        """
        Apply all preprocessing tools for which `enable` is `True`.
        Return the preprocessed image (numpy array).

        Parameters
        ----------
        img : array_like
            Single image as numpy array or multiple images as array-like object

        """
        sequence = self.params.sequence
        if sequence is None:
            sequence = self.params.available_tools

        for tool in sequence:
            tool_params = self.params.__dict__[tool]
            if tool_params.enable:
                logger.debug('Apply ' + tool)
                kwargs = tool_params._make_dict()
                for k in kwargs.keys():
                    if k in ['_attribs', '_tag', '_tag_children', 'enable']:
                        kwargs.pop(k)

                cls = self.__class__
                img = cls.__dict__[tool](img, **kwargs)

        return img
