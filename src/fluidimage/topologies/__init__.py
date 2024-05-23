"""Topologies representing asynchronous computations
====================================================

A topology represents an asynchronous computation as a graph of **waiting
queues** and **works**. Works take (an) item(s) in (an) input queue(s), process
it/them and put the result in (an) output queue(s).

All topologies inherit from a base class
:class:`fluidimage.topologies.base.TopologyBase`, which has methods to define
the topology (``add_queue`` and ``add_work``), to represent the computational
graph (``make_code_graphviz``) and finally to execute it (``compute``).

Users are particularly concerned with the following already defined topologies:

.. autosummary::
   :toctree:

   piv
   bos
   preproc
   image2image
   surface_tracking
   optical_flow

These others modules defined classes and functions useful for developers.

.. autosummary::
   :toctree:

   base
   log
   launcher
   splitters

"""

from .base import TopologyBase, TopologyBaseFromSeries
from .log import LogTopology

__all__ = ["LogTopology", "TopologyBase"]
