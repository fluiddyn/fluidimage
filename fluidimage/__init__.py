"""
FluidImage
==========

"""

from ._version import __version__

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles, SeriesOfArrays
from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.util import create_object_from_file
from fluiddyn.util import config_logging as _cl_fluiddyn
from .util.util import imread, imsave, imsave_h5


def config_logging(level='info', name='fluidimage'):
    _cl_fluiddyn(level=level, name=name)
