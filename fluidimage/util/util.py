"""
Miscellaneous utilities
=======================

Provides:

.. autofunction:: imread

.. autofunction:: print_memory_usage

.. autofunction:: cstring

.. autofunction:: cprint

.. autofunction:: is_memory_full

.. autofunction:: raise_exception

"""

import sys
import psutil
from pathlib import Path

import six

from fluiddyn.util import get_memory_usage
from fluiddyn.util import terminal_colors as term

from fluiddyn.io.image import imread as _imread, imsave as _imsave

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
    if isinstance(path, Path):
        path = str(path)
    try:
        array = _imread(path)
    except IOError as e:
        raise_exception(e, path)
    return array


def imsave(path, array, **kwargs):
    _imsave(path, array, **kwargs)


def _get_txt_memory_usage(string="Memory usage", color="OKGREEN"):
    mem = get_memory_usage()
    cstr = cstring(
        (string + ": ").ljust(30) + "{:.3f} Mb".format(mem), color=color
    )
    return cstr


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
    from .log import log_memory_usage

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
