"""Works - treatments
=====================

.. autosummary::
   :toctree:

   piv

"""
from .. import imread


class BaseWork(object):
    def __init__(self, params=None):
        self.params = params


def load_image(path):
    return imread(path)
