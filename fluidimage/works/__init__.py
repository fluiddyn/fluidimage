"""Works - treatments
=====================

.. autosummary::
   :toctree:

   piv

"""
try:
    from scipy.ndimage import imread
except ImportError:
    from scipy.misc import imread


class BaseWork(object):
    def __init__(self, params=None):
        self.params = params


def load_image(path):
    return imread(path)
