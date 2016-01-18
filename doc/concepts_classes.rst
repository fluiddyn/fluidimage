Concepts and classes
====================

Data objects
------------

"Unit" objects
~~~~~~~~~~~~~~

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
  
Fluxes, waiting queues and working topologies
---------------------------------------------

A flux is responsible for the organization of the treatment of a
succession of input "unit" objects. It contains the description as a
"topology" of the treatment of one "unit" object (for example the
production of 1 PIV field from a couple of images). A topology is
formed of a set of unit processes linked by waiting queues.

The flux and the waiting queues organize the "loop" over the input
set. They organizes the asynchronous calls of the work units of the
topology with the correct transfer of data between them.

The Links between the work units is handled by the waiting queues and
can be done by:

- Python objects as arguments
- save file / load file
- serialize and transfer by socket (maybe pyzmq?)

Works, Work units and workers
-----------------------------

A work does a treatment. It has initialization parameters and after
initialization is able to produce an output object from an input
object. It can also take more than one input objects and/or return
more than one output objects.

A work is made of one or more work units. In particular, it should be
useful to split input/output and computational works.

The simplest way to define a work unit is by a Python function taking 
input arguments and returning its result.

A worker is a process (or a thread) running one unit work function.
