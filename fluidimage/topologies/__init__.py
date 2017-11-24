"""Topologies representing asynchronous computations
====================================================

A topology is responsible for the organization of the treatment of a succession
of input "unit" objects. It contains the description as a "topology" of the
treatment of one "unit" object (for example the production of 1 PIV field from
a couple of images). A topology is formed of a set of unit processes linked by
waiting queues.

The Topology object also organizes the "loop" over the input set. It organizes
the asynchronous (parallel) calls of the work units of the topology with the
correct transfer of data between them.

Users are particularly concerned with the PIV and preprocessing topologies:

.. autosummary::
   :toctree:

   piv
   bos
   preproc

These others modules defined classes and functions useful for developers.

.. autosummary::
   :toctree:

   base
   waiting_queues
   log
   launcher

"""
