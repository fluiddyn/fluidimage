"""Preprocessing of images
==========================

Images acquired from experiments can suffer from noise, reflection and other
undesirable effects. Preprocessing images by careful use of thresholds and
filters can significantly improve the quality of PIV.

Provides:

.. autosummary::
   :toctree:

   toolbox
   _toolbox_cv
   _toolbox_py
   io

Provides

.. autoclass:: Work
   :members:
   :private-members:

.. autoclass:: Topology
   :members:
   :private-members:

"""

from fluidimage.topologies.preproc import Topology, TopologyPreproc
from fluidimage.works.preproc import Work, WorkPreproc

__all__ = ["Work", "WorkPreproc", "TopologyPreproc", "Topology"]
