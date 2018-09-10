"""logging (:mod:`fluidimage.util.log`)
=======================================

.. autofunction:: log_memory_usage


.. autofunction:: reset_logger


"""

from logging import getLogger, DEBUG

from .util import _get_txt_memory_usage

__all__ = ["logger", "DEBUG", "reset_logger", "log_memory_usage"]

logger = getLogger("fluidimage")


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
