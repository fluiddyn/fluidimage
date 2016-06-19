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

from .io import iterate_multiple_imgs, multiple_imgs_as_ndarray


available_tools = ['sliding_median', 'sliding_minima',
                   'temporal_median', 'temporal_minima',
                   'global_threshold', 'rescale_intensities',
                   'tanh_intensities', 'sharpen']


def imstats(img, hist_bins=256):
    histogram = nd.measurements.histogram(bins=hist_bins)
    return histogram


# ----------------------------------------------------
#   SPATIAL FILTERS
# ----------------------------------------------------

@iterate_multiple_imgs
def sliding_median(img=None, weight=1., window_size=3,
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
def sliding_minima(img=None, weight=1., window_size=3,
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
#   TEMPORAL FILTERS
# ----------------------------------------------------

@multiple_imgs_as_ndarray
def temporal_median(img=None, weight=1., window_shape=None):
    '''
    Subtracts the median calculated in time,for each pixel.

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
        raise ValueError('Need more than one image to apply temporal filtering.')

    if window_shape is None:
        window_shape = (nb_imgs, 1, 1)
    elif not isinstance(window_shape, tuple):
        raise ValueError('window_shape must be a tuple.')
    elif window_shape[0] <= 1:
        raise ValueError('Cannot perform temporal filtering, try spatial filtering.')

    img_out = img - weight * nd.median_filter(img,
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
        raise ValueError('Need more than one image to apply temporal filtering.')

    window_size = img.shape[time_axis]
    img_out = img - weight * nd.minimum_filter1d(img,
                                                 size=window_size,
                                                 axis=time_axis)
    return img_out


# ----------------------------------------------------
#   BRIGHTNESS / CONTRAST TOOLS
# ----------------------------------------------------

@iterate_multiple_imgs
def global_threshold(img=None, minima=0., maxima=1e4):
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
def rescale_intensities(img=None, minima=0., maxima=1e4):
    '''
    Rescale image intensities, between the specified minima and maxima,
    by using a multiplicative factor.

    Parameters
    ----------
    img : array_like
        Single image as numpy array or multiple images as array-like object
    minima, maxima : float
        Sets the range within which current intensities
        have to be rescaled.

    '''
    offset = minima - img.min()
    img += offset
    initial_min = img.min()
    initial_max = img.max()
    mfactor = (maxima - minima) / (initial_max - initial_min)
    img_out = img * mfactor
    return img_out


@iterate_multiple_imgs
def tanh_intensities(img=None, maxima=1.):
    ''' FIXME: Doesn't work as of now. '''

    img_out = rescale_intensities(img, maxima=maxima)
    img_out = np.tanh(img_out)
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
                enable_doc = 'enable : bool\n        Set as `True` to enable the tool'
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
                kwargs = tool_params._make_dict()
                for k in kwargs.keys():
                    if k in ['_attribs', '_tag', '_tag_children', 'enable']:
                        kwargs.pop(k)

                cls = self.__class__
                img = cls.__dict__[tool](img, **kwargs)

        return img
