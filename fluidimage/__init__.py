"""
FluidImage
==========

"""

from ._version import __version__

import os
from logging import getLogger as _getLogger
from fluiddyn.io.image import (imread as _imread,
                               imsave as _imsave,
                               imsave_h5)

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles, SeriesOfArrays
from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.util import create_object_from_file

from fluiddyn.util import config_logging as _cl_fluiddyn


logger = _getLogger('fluidimage')


def config_logging(level='info', name='fluidimage'):
    _cl_fluiddyn(level=level, name=name)


def imread(path):
    """Flatten image as a single gray-scale layer and
    loads as a numpy floating point array.

    """
    try:
        array = _imread(path, flatten=True)
    except:
        array = _imread(path, flatten=1)

    fname = os.path.basename(path)
    logger.info('Load %s with intensity range (%f, %f) and type %s',
                fname, array.min(), array.max(), array.dtype)
    return array


def imsave(path, array, **kwargs):
    _imsave(path, array, **kwargs)
    fname = os.path.basename(path)
    logger.info('Save %s with intensity range (%f, %f) and type %s',
                fname, array.min(), array.max(), array.dtype)
