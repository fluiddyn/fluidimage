"""
Miscellaneous utilities
=======================

Provides:

.. autofunction:: imread

.. autofunction:: log_memory_usage

.. autofunction:: print_memory_usage

.. autofunction:: cstring

.. autofunction:: cprint

.. autofunction:: is_memory_full

.. autofunction:: raise_exception

"""
from __future__ import print_function

import sys
import six
import psutil

from logging import getLogger
from fluiddyn.util import get_memory_usage
from fluiddyn.io.image import imread as _imread, imsave as _imsave, imsave_h5

from fluiddyn.util import terminal_colors as term


logger = getLogger("fluidimage")
color_dict = {
    "HEADER": term.HEADER,
    "OKBLUE": term.OKBLUE,
    "OKGREEN": term.OKGREEN,
    "WARNING": term.WARNING,
    "FAIL": term.FAIL,
    "ENDC": term.ENDC,
}


def imread(path):
    """Flatten image as a single gray-scale layer and
    loads as a numpy floating point array.

    """
    try:
        array = _imread(path)
    except IOError as e:
        raise_exception(e, path)

    # fname = os.path.basename(path)
    # logger.debug('Load %s with intensity range (%d, %d) and type %s',
    #              fname, array.min(), array.max(), array.dtype)
    return array


def imsave(path, array, **kwargs):
    _imsave(path, array, **kwargs)


# fname = os.path.basename(path)
# logger.info('Save %s with intensity range (%d, %d) and type %s',
#             fname, array.min(), array.max(), array.dtype)


def _get_txt_memory_usage(string="Memory usage", color="OKGREEN"):
    mem = get_memory_usage()
    cstr = cstring(
        (string + ": ").ljust(30) + "{:.3f} Mb".format(mem), color=color
    )
    return cstr


def log_memory_usage(string="Memory usage", color="OKGREEN", mode="info"):
    """Log the memory usage."""

    if mode == "debug":
        log = logger.debug
    elif mode == "error":
        log = logger.error
    else:
        log = logger.info

    log(_get_txt_memory_usage(string, color))


def print_memory_usage(string="Memory usage", color="OKGREEN"):
    """Print the memory usage."""
    print(_get_txt_memory_usage(string, color))


def cstring(*args, **kwargs):
    """Return a coloured string."""

    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = "OKBLUE"

    cstr = " ".join(args)
    return color_dict[color] + cstr + color_dict["ENDC"]


def cprint(*args, **kwargs):
    """Print with terminal colors."""

    cstr = cstring(*args, **kwargs)
    print(cstr)


def is_memory_full():
    """Checks if available system virtual memory is nearly saturated.

    Returns
    -------
    type: bool
        `True` if memory usage > 90 % and available memory < 500 MB.
        `False` otherwise
    """
    mem = psutil.virtual_memory()

    if mem.percent > 90 or mem.available < 500 * 1024 ** 2:
        log_memory_usage("Memory full! Current process using", mode="info")
        return True

    else:
        return False


def raise_exception(exc, msg=""):
    """Raise an exception with a custom message
    cf. http://python-future.org/compatible_idioms.html

    """
    traceback = sys.exc_info()[2]
    six.reraise(type(exc), msg, traceback)
