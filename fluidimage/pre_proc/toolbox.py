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
import matplotlib.pyplot as plt
import scipy.ndimage as nd


available_tools = ['sliding_median',
                   'global_threshold',
                   'sharpen']


def sliding_mean(img=None, weight=1.):
    pass


def sliding_median(img=None, weight=1., filter_size=3.):
    img_out = img - weight * nd.median_filter(img, filter_size)
    return img_out


def sliding_mode(img=None, weight=1.):
    pass


def sliding_threshold(img=None, minima=0., maxima=1e4):
    pass


def global_threshold(img=None, minima=0., maxima=1e4):
    '''
    Parameters
    ----------
    img : nd-array
        Image as a numpy array
    minima, maxima : float
        Sets the threshold
    '''

    img_out = img.copy()
    img_out[img_out < minima] = minima
    img_out[img_out > maxima] = maxima
    return img_out


def sharpen(img=None, sigma1=3., sigma2=1., alpha=30.):
    '''
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
        Dynamically add the global functions in this module as classmethods of
        the present class. Also create default parameters from the function
        argument list.
        """

        params.preproc._set_child('tools')
        params = params.preproc.tools
        params._set_attrib('available_tools', available_tools)

        for tool in available_tools:
            func = globals()[tool]
            setattr(PreprocTools, tool, func)

            func_args = inspect.getcallargs(func)
            for arg in func_args.keys():
                if arg in ['img']:  # Remove arguments which are not parameters
                    del(func_args[arg])

            func_args.update({'enable': False})
            params._set_child(tool, func_args)

    def __init__(self, params):
        self.params = params.preproc.tools

    def __call__(self, img, sequence=None):
        """
        Apply all preprocessing tools for which `enable` is `True`.
        Return the preprocessed image (numpy array).
        """

        if type(img) is not np.ndarray:
            raise ValueError('Expected a numpy array, instead received %s = %s'
                             % (type(img), img))

        img_out = img.copy()

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
