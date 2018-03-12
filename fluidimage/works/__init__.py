"""Works - processing
=====================

This subpackage defines some works. A work does a processing. It has
initialization parameters and after initialization is able to produce an output
object from an input object. It can also take more than one input objects
and/or return more than one output objects.

A work is made of one or more work units. In particular, it could be useful to
define input/output and computational works.

.. autosummary::
   :toctree:

   piv
   preproc

"""
from .. import imread


class BaseWork(object):
    def __init__(self, params=None):
        self.params = params


def load_image(path):
    im = imread(path)
    return im
