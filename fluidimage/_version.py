from importlib import metadata

__version__ = metadata.version(__package__)


try:
    from fluidimage._hg_rev import hg_rev
except ImportError:
    hg_rev = "?"
