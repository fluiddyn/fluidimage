Overview of the organization of the package
===========================================

Data objects
------------

FluidImage uses data objects. These objects represent particular types of
data. They should be able to be loaded from a file, saved into a file,
displayed to the screen, etc.

These objects are defined in the package :mod:`fluidimage.data_objects`.

Works
-----

A work does a processing. It has initialization parameters and after
initialization is able to produce an output object from an input object.

The works are defined in the package :mod:`fluidimage.works`. Internally, the
works use utilities for processing defined in the package
:mod:`fluidimage.calcul`.

Topologies, waiting queues and executors
----------------------------------------

A **computational topology** contains the description of the processing of one
object (for example the production of one PIV file from a couple of image
files). A topology is formed of a set of **waiting queues** linked by
**works**.

The topology classes are defined in the package :mod:`fluidimage.topologies`.

The execution of a topology is done by an **executor**. Executor classes are
defined in the package :mod:`fluidimage.executors`. The executor organizes the
asynchronous calls of the work units of the topology with the correct transfer
of data between them.


