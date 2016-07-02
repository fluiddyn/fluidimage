"""
Miscellaneous utilities
=======================

"""
import os
from logging import getLogger
from fluiddyn.util.util import get_memory_usage
from fluiddyn.io.image import (imread as _imread,
                               imsave as _imsave,
                               imsave_h5)

from fluiddyn.util import terminal_colors as term


logger = getLogger('fluidimage')


def imread(path):
    """Flatten image as a single gray-scale layer and
    loads as a numpy floating point array.

    """
    array = _imread(path, flatten=True)

    fname = os.path.basename(path)
    logger.info('Load %s with intensity range (%d, %d) and type %s',
                fname, array.min(), array.max(), array.dtype)
    return array


def imsave(path, array, **kwargs):
    _imsave(path, array, **kwargs)
    fname = os.path.basename(path)
    logger.info('Save %s with intensity range (%d, %d) and type %s',
                fname, array.min(), array.max(), array.dtype)


def log_memory_usage(string='Memory usage', color='WARNING'):
    """Log the memory usage when debug is on."""

    mem = get_memory_usage()
    color_dict = {'HEADER': term.HEADER,
                  'OKBLUE': term.OKBLUE,
                  'OKGREEN': term.OKGREEN,
                  'WARNING': term.WARNING,
                  'FAIL': term.FAIL,
                  'ENDC': term.ENDC}
    begin = color_dict[color]
    end = color_dict['ENDC']
    logger.debug(begin + (string + ':').ljust(30) + '%d %s' + end, mem, 'Mb')
