# Overview of the package

## Data objects ({mod}`fluidimage.data_objects`)

FluidImage uses data objects. These objects represent particular types of data. They
should be able to be loaded from a file, saved into a file, displayed to the screen, etc.

## Works ({mod}`fluidimage.works`)

A work is a function processing input data and potentially returning input data. Some
works are actually classes which have to be initialized with parameters.

For example, the work {class}`fluidimage.works.piv.multipass.WorkPIV` provide a method
`calcul(couple)` which compute a PIV field from a couple of arrays.

The works are defined in the package {mod}`fluidimage.works`. Internally, the works use
utilities for processing defined in the package {mod}`fluidimage.calcul`.

## Topologies, waiting queues and executors

A **computational topology** contains the description of an asynchronous computation as a
graph made of **waiting queues** and **works**. The base class
{class}`fluidimage.topologies.base.TopologyBase` provides the methods `add_queue` and
`add_work` to define the topology.

The topology classes are defined in the package {mod}`fluidimage.topologies`.

The execution of a topology is done by an **executor**. Executor classes are defined in
the package {mod}`fluidimage.executors`. The executor organizes the asynchronous calls of
the work units of the topology with the correct transfer of data between them.
