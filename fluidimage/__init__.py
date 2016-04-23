"""
FluidImage
==========

"""

from ._version import __version__

try:
    from scipy.ndimage import imread
except ImportError:
    from scipy.misc import imread

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles, SeriesOfArrays
from fluiddyn.util.paramcontainer import ParamContainer

from fluiddyn.util import config_logging
