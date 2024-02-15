"""logging (:mod:`fluidimage.util.log`)
=======================================

.. autofunction:: log_memory_usage

.. autofunction:: reset_logger

.. autofunction:: log_debug

.. autofunction:: log_error

"""

from logging import DEBUG, getLogger

from fluiddyn.util import config_logging as _cl_fluiddyn

from .util import _get_txt_memory_usage, cstring

__all__ = ["logger", "DEBUG", "reset_logger", "log_memory_usage"]

logger = getLogger("fluidimage")


def config_logging(level="info", name="fluidimage", file=None):
    _cl_fluiddyn(level=level, name=name, file=file)


def reset_logger():
    """Remove all handlers (files) linked to the fluidimage logger"""
    for handler in logger.handlers:
        logger.removeHandler(handler)


def log_memory_usage(string="Memory usage", color="OKGREEN", mode="info"):
    """Log the memory usage."""

    if mode == "debug":
        log = logger.debug
    elif mode == "error":
        log = logger.error
    else:
        log = logger.info

    log(_get_txt_memory_usage(string, color))


def log_debug(string):
    """Log in debug mode with WARNING color"""
    logger.debug(cstring(string, color="WARNING"))


def log_error(string):
    """Log in error mode with FAIL color"""
    logger.error(cstring(string, color="FAIL"))
