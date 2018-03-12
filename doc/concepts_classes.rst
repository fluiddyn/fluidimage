Concepts, classes and organization of the package
=================================================

Data objects
------------

FluidImage uses data objects. These objects represent particular types of
data. They should be able to be loaded from a file, saved into a file,
displayed to the screen, etc.

These objects are defined in the package :mod:`fluidimage.data_objects`.

"Unit" objects
~~~~~~~~~~~~~~

We can define simple "unit" object as for example:

- ImageInFile
- ImageInFilm
- CoupleOfImages
- UnstructureVelocityFieldWithCorr
- UnstructureVelocityField
- StructureVelocityField
- ...

"Set" objects
~~~~~~~~~~~~~

There are also set of objects, i.e. a succession of "unit" objects.

- SetOfImagesInFiles
- SetOfImagesInFilm
- SetOfCouplesOfImages  
- SetOfUnstructureVelocityFieldWithCorr
- SetOfUnstructureVelocityField
- SetOfStructureVelocityField
- ...

A "SetOf" instance can iterate over the objects that it contains.


Works and Work units
--------------------

A work does a processing. It has initialization parameters and after
initialization is able to produce an output object from an input object. It can
also take more than one input objects and/or return more than one output
objects.

A work is made of one or more work units. In particular, it could be useful to
define input/output and computational works.

The works are defined in the package :mod:`fluidimage.works`.  Internally, the
works use utilities for processing defined in the package
:mod:`fluidimage.calcul`.


Topologies and waiting queues
-----------------------------

A topology is responsible for the organization of the processing of a
succession of input "unit" objects. It contains the description as a "topology"
of the processing of one "unit" object (for example the production of 1 PIV
field from a couple of images). A topology is formed of a set of unit processes
linked by waiting queues.

The Topology object also organizes the "loop" over the input set. It organizes
the asynchronous calls of the work units of the topology with the correct
transfer of data between them.

In principle, the links between the work units can be done by:

- Python objects as arguments (preferred because it is faster and simpler)
- save file / load file
- serialize and transfer by socket (maybe pyzmq?)

The topologies and the waiting queues are defined in the package
:mod:`fluidimage.topologies`.
