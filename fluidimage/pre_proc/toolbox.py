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


available_tools = ['sliding_median', 'sliding_minima',
                   'global_threshold', 'rescale_intensities',
                   'tanh_intensities', 'sharpen']


def imstats(img, hist_bins=256):
    histogram = nd.measurements.histogram(bins=hist_bins)
    return histogram


# ----------------------------------------------------
#   SPATIAL FILTERS
# ----------------------------------------------------


def sliding_mean(img=None, weight=1.):
    pass


def sliding_median(img=None, weight=1., window_size=3,
                   boundary_condition='reflect'):
    '''
    Subtracts the median calculated within a sliding window from the centre of
    the window.

    Parameters
    ----------
    img : nd-array
        Image as a numpy array
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


def sliding_minima(img=None, weight=1., window_size=3,
                   boundary_condition='reflect'):
    '''
    Subtracts the minimum calculated within a sliding window from the centre of
    the window.

    Parameters
    ----------
    img : nd-array
        Image as a numpy array
    weight : scalar
        Fraction of median to be subtracted from each pixel.
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
#   BRIGHTNESS / CONTRAST TOOLS
# ----------------------------------------------------


def global_threshold(img=None, minima=0., maxima=1e4):
    '''
    Trims pixel intensities which are outside the interval (minima, maxima).

    Parameters
    ----------
    img : nd-array
        Image as a numpy array# Replace with inspect.getfullargspec

    minima, maxima : float
        Sets the threshold

    '''
    img_out = img.copy()
    img_out[img_out < minima] = minima
    img_out[img_out > maxima] = maxima
    return img_out


def rescale_intensities(img=None, minima=0., maxima=1e4):
    '''
    Rescale image intensities, between the specified minima and maxima,
    by using a multiplicative factor.

    Parameters
    ----------
    img : nd-array
        Image as a numpy array
    minima, maxima : float
        Sets the range within which current intensities
        have to be rescaled.

    '''
    initial_min = img.min()
    initial_max = img.max()
    mfactor = (maxima - minima) / (initial_max - initial_min)
    offset = minima - initial_min
    img_out = img * mfactor + offset
    return img_out


def tanh_intensities(img=None, maxima=1e4):
    # TODO: mean filter
    img_out = img
    img_out = np.tanh(img_out)
    # TODO: maxima from histogram??
    img_out = rescale_intensities(img_out, maxima=maxima)
    return img_out


def sharpen(img=None, sigma1=3., sigma2=1., alpha=30.):
    '''
    Sharpen image edges.

    Parameters
    ----------
    img : nd-array
        Image as a numpy array
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
        img : nd-array
            Image as a numpy array

        """
        if type(img) is not np.ndarray:
            raise ValueError('Expected a numpy array, instead received %s = %s'
                             % (type(img), img))

        img_out = img.copy()

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
                img_out = cls.__dict__[tool](img_out, **kwargs)

        return img_out
