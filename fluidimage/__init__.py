"""
FluidImage
==========

"""

import sys
import os
from warnings import warn

if "OMP_NUM_THREADS" not in os.environ:
    if "numpy" in sys.modules:
        warn(
            "The environment variable OMP_NUM_THREADS "
            "was not set and numpy is already imported "
            'so fluidimage can not set OMP_NUM_THREADS to "1". '
            "It can be very bad for the performance of fluidimage topologies!"
        )
    else:
        warn(
            "The environment variable OMP_NUM_THREADS "
            'was not set so fluidimage fixes it to "1", '
            "which is needed for fluidimage topologies."
        )
        os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from ._version import __version__

from fluiddyn.io.image import imread as _imread, imsave as _imsave, imsave_h5

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles, SeriesOfArrays
from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util import create_object_from_file, get_memory_usage
from fluiddyn.util import config_logging as _cl_fluiddyn
from .util.util import (
    imread,
    imsave,
    logger,
    log_memory_usage,
    print_memory_usage,
)
from fluidimage.topologies.log import LogTopology

try:
    from ._path_image_samples import path_image_samples
except ImportError:
    pass


def config_logging(level="info", name="fluidimage", file=None):
    _cl_fluiddyn(level=level, name=name, file=file)


__all__ = [
    "__version__",
    "create_object_from_file",
    "get_memory_usage",
    "imread",
    "imsave",
    "logger",
    "log_memory_usage",
    "print_memory_usage",
    "ParamContainer",
    "LogTopology",
    "path_image_samples",
    "SeriesOfArrays",
    "SerieOfArraysFromFiles",
    "np",
]
