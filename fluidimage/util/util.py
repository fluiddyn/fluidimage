"""
Miscellaneous utilities
=======================

Provides:

.. autofunction:: imread

.. autofunction:: print_memory_usage

.. autofunction:: cstring

.. autofunction:: cprint

.. autofunction:: is_memory_full

.. autofunction:: str_short

"""

import sys
from pathlib import Path

import psutil
import six
from IPython.lib.pretty import pretty

from fluiddyn.io.image import imread as _imread
from fluiddyn.io.image import imsave as _imsave
from fluiddyn.util import get_memory_usage
from fluiddyn.util import terminal_colors as term

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
    # pylint: disable=W0703
    try:
        array = _imread(path)
    except Exception as error:
        raise type(error)(path).with_traceback(error.__traceback__)

    return array


def imsave(path, array, **kwargs):
    "tmp docstring"
    _imsave(path, array, **kwargs)


imsave.__doc__ = _imsave.__doc__


def _get_txt_memory_usage(string="Memory usage", color="OKGREEN"):
    mem = get_memory_usage()
    cstr = cstring((string + ": ").ljust(30) + f"{mem:.3f} Mb", color=color)
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


def str_short(obj):
    """Give a short str for classes, function, etc."""

    try:
        return obj.__module__ + "." + obj.__name__
    except AttributeError:
        return pretty(obj)
