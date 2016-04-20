"""
FluidImage
==========

"""

from ._version import __version__

try:
    from scipy.ndimage import imread
except ImportError:
    from scipy.misc import imread

from fluiddyn.util.serieofarrays import SeriesOfArrays
