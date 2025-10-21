# Overview

FluidImage is a free and open-source Python framework to process images of fluids (in
particular with [PIV]), and analyse the resulting fields.

Image processing for fluid mechanics is still dominated by proprietary tools. Such tools
are not ideal when you want to understand and tweak the algorithms and/or to use
clusters. There are also good and useful PIV software ([PIVlab], [UVmat]) written in
Matlab, which is itself proprietary.

With the improvement of the Python numerical ecosystem and of tools for collaborative
development, one can think it is possible to build together a good community-driven
library/toolkit specialized in image processing for fluid mechanics. This is our project
with FluidImage.

FluidImage has now grown into a clean software reimplementing in modern Python algorithms
and ideas taken from [UVmat], [OpenPIV], [PIVlab] and [PIVmat] with a focus on
performance, usability and maintainability. However, FluidImage is not restricted to
Particle Image Velocimetry computations ([PIV], i.e. displacements of pattern obtained by
correlations of cropped images) and can be used to

- display and pre-process images,

- compute displacement or velocity fields with PIV, Background-Oriented Schlieren
  ([BOS](https://en.wikipedia.org/wiki/Background-oriented_schlieren_technique)) and
  [optical flow](https://en.wikipedia.org/wiki/Optical_flow),

- analyze and display vector and scalar fields.

We want to make FluidImage easy (useful documentation, easy installation, nice API,
usable with simple scripts and few simple graphical user interfaces), reliable (with good
[unittests](https://codecov.io/gh/fluiddyn/fluidimage/)) and very efficient, in
particular when the number of images is large. Thus we want FluidImage to be able to run
efficiently and easily on a personal computer and on big clusters. The efficiency is
achieved by using

- a framework for asynchronous computations (currently, we use
  [Trio](https://trio.readthedocs.io)) and an associated API to define "topologies" of
  parallel computations.

- parallelism to efficiently use the available cores of the Central Processing Units
  (CPU),

- good profiling and efficient and specialized algorithms,

- cutting-edge tools for fast computations with Python (in particular
  [Pythran](https://pythran.readthedocs.io) through
  [Transonic](https://transonic.readthedocs.io)).

## Few very simple examples

### Visualize images

Let's fetch Fluidimage example data and go to a directory containing images:

```sh
cd $(python -c "import fluidimage as fli; print(fli.get_path_image_samples() / 'Karman/Images')")
```

```{note}
Here, we used a Bash command. On Windows, you can use WSL or launch just the
Python command and `cd` into it manually.
```

One can visualize the files with:

```sh
fluidimviewer
```

### Preprocess images

To preprocess several images in parallel, we can use a Fluidimage `Topology`.

For good performance, one needs to informe the libraries used for the computation that
each image has to be treated sequentially, i.e. without trying to use more than one CPU
core. This has to be done by setting the environment variable `OMP_NUM_THREADS`. With a
Posix shell like Bash, this can be done with

```sh
export OMP_NUM_THREADS=1
```

One can now run in Python:

```py
from fluidimage import get_path_image_samples
from fluidimage.preproc import Topology

params = Topology.create_default_params()
# location of the input images
params.series.path = get_path_image_samples() / "Karman/Images"
# set other parameters
...
topology = Topology(params, logging_level="info")
# Compute in parallel
topology.compute()
```

- To understand the concept of Fluidimage Topology and how it works under the hood, see
  [](./overview_orga_package.md).

- There are [few examples about image preprocessing](./examples/preproc.md).

- The documentation of the preprocessing topology is here:
  {class}`fluidimage.preproc.Topology`.

### PIV computation

For Particle Image Velocimetry, one can use something like:

```py
from fluidimage import get_path_image_samples
from fluidimage.piv import Topology

params = Topology.create_default_params()
# location of the input images
params.series.path = get_path_image_samples() / "Karman/Images"
# set other parameters
...
topology = Topology(params, logging_level="info")
# Compute in parallel
topology.compute()
```

- See the [PIV tutorial](./tutorials/tuto_piv.md).

- There are few examples about PIV: [](./examples/piv_parallel.md) and
  [](./examples/piv_cluster.md).

- The command `fluidimage-monitor` can be used to monitor parallel computations.

- The command `fluidpivviewer` can be used to visualize the results.

[openpiv]: http://www.openpiv.net/
[piv]: https://en.wikipedia.org/wiki/Particle_image_velocimetry
[pivlab]: https://pivlab.blogspot.com/p/what-is-pivlab.html
[pivmat]: http://www.fast.u-psud.fr/pivmat/
[uvmat]: http://servforge.legi.grenoble-inp.fr/projects/soft-uvmat/wiki/UvmatHelp
