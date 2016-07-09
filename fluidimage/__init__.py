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
from fluiddyn.util.util import create_object_from_file, get_memory_usage

from fluiddyn.util import (config_logging as _cl_fluiddyn,
                           terminal_colors as term)


logger = _getLogger('fluidimage')


def config_logging(level='info', name='fluidimage', file=None):
    _cl_fluiddyn(level=level, name=name, file=file)


def imread(path):
    """Flatten image as a single gray-scale layer and
    loads as a numpy floating point array.

    """
    array = _imread(path, flatten=True)

    # fname = os.path.basename(path)
    # logger.info('Load %s with intensity range (%d, %d) and type %s',
    #             fname, array.min(), array.max(), array.dtype)
    return array


def imsave(path, array, **kwargs):
    _imsave(path, array, **kwargs)
    # fname = os.path.basename(path)
    # logger.info('Save %s with intensity range (%d, %d) and type %s',
    #             fname, array.min(), array.max(), array.dtype)


def _get_txt_memory_usage(string='Memory usage', color='OKGREEN'):
    mem = get_memory_usage()
    color_dict = {'HEADER': term.HEADER,
                  'OKBLUE': term.OKBLUE,
                  'OKGREEN': term.OKGREEN,
                  'WARNING': term.WARNING,
                  'FAIL': term.FAIL}
    begin = color_dict[color]
    end = term.ENDC
    return begin + (string + ': ').ljust(30) + '{:.3f} Mb'.format(mem) + end


def log_memory_usage(string='Memory usage', color='OKGREEN'):
    """Log the memory usage when info is on."""
    logger.info(_get_txt_memory_usage(string, color))


def print_memory_usage(string='Memory usage', color='OKGREEN'):
    """Print the memory usage."""
    print(_get_txt_memory_usage(string, color))
