"""
FluidImage
==========

"""

from ._version import __version__

import os
from fluiddyn.io.image import (imread as _imread,
                               imsave as _imsave,
                               imsave_h5)

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles, SeriesOfArrays
from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.util import create_object_from_file, get_memory_usage
from fluiddyn.util import config_logging as _cl_fluiddyn
from .util.util import (
    imread, imsave, imsave_h5, log_memory_usage, print_memory_usage)


def config_logging(level='info', name='fluidimage', file=None):
    _cl_fluiddyn(level=level, name=name, file=file)
