# FluidImage

[![Latest version](https://img.shields.io/pypi/v/fluidimage.svg)](https://pypi.python.org/pypi/fluidimage/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/fluidimage.svg)
[![Documentation status](https://readthedocs.org/projects/fluidimage/badge/?version=latest)](http://fluidimage.readthedocs.org)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Code coverage](https://codecov.io/gh/fluiddyn/fluidimage/branch/branch%2Fdefault/graph/badge.svg)](https://codecov.io/gh/fluiddyn/fluidimage/branch/branch%2Fdefault/)
[![Heptapod CI](https://foss.heptapod.net/fluiddyn/fluidimage/badges/branch/default/pipeline.svg)](https://foss.heptapod.net/fluiddyn/fluidimage/-/pipelines)
[![Github Actions Linux](https://github.com/fluiddyn/fluidimage/actions/workflows/ci-linux.yml/badge.svg?branch=branch/default)](https://github.com/fluiddyn/fluidimage/actions/workflows/ci-linux.yml)
[![Github Actions Pixi](https://github.com/fluiddyn/fluidimage/actions/workflows/ci-pixi.yml/badge.svg?branch=branch/default)](https://github.com/fluiddyn/fluidimage/actions/workflows/ci-pixi.yml)

FluidImage is a free and open-source Python framework to process images of fluids (in
particular with [PIV]), and analyse the resulting fields.

**Documentation:** <http://fluidimage.readthedocs.org>

Image processing for fluid mechanics is still dominated by proprietary tools. Such tools
are not ideal when you want to understand and tweak the algorithms and/or to use
clusters. There are also good and useful PIV software ([PIVlab], [UVmat]) written in
Matlab, which is itself proprietary.

With the improvement of the Python numerical ecosystem and of tools for collaborative
development, one can think it is possible to build together a good community-driven
library/toolkit specialized in image processing for fluid mechanics. This is our project
with FluidImage.

Fluidimage has now grown into a clean software reimplementing in modern Python algorithms
and ideas taken from [UVmat], [OpenPIV], [PIVlab] and [PIVmat] with a focus on
performance, usability and maintanability. However, Fluidimage is not restricted to
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

## Installation and setup

Fluidimage can be installed from
[wheels and sdist available on PyPI](https://pypi.python.org/pypi/fluidimage/) with tools
like pip, [UV] and [PDM]. Alternatively, it can be installed from conda-forge packages
(for example available on https://prefix.dev/channels/conda-forge/packages/fluidimage)
with tools like conda (installed with [Miniforge]) or [Pixi].

In some cases, Fluidimage will require that the environment variable `OMP_NUM_THREADS` is
set to `1`. With a POSIX shell like Bash, it can be done with `export OMP_NUM_THREADS=1`.

For more details, see
[the installation page in the documentation](https://fluidimage.readthedocs.io/en/latest/install.html).

## Basic usage

Few minimalist examples about image visualisation, preprocessing and PIV computation are
presented in
[the overview of the project](https://fluidimage.readthedocs.io/en/latest/overview.html).
Other usage cases can be found in our
[tutorials](https://fluidimage.readthedocs.io/en/latest/tutorial.html) and
[examples](https://fluidimage.readthedocs.io/en/latest/examples.html).

## Citation

If you use Fluidimage to produce scientific articles, please cite
[our metapaper presenting the FluidDyn project](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.237/):

```bibtex

@article{fluiddyn,
doi = {10.5334/jors.237},
year = {2019},
publisher = {Ubiquity Press,  Ltd.},
volume = {7},
author = {Pierre Augier and Ashwin Vishnu Mohanan and Cyrille Bonamy},
title = {{FluidDyn}: A Python Open-Source Framework for Research and Teaching in Fluid Dynamics
    by Simulations,  Experiments and Data Processing},
journal = {Journal of Open Research Software}
}

```

[miniforge]: https://github.com/conda-forge/miniforge
[openpiv]: http://www.openpiv.net/
[pdm]: https://pdm-project.org
[piv]: https://en.wikipedia.org/wiki/Particle_image_velocimetry
[pivlab]: https://pivlab.blogspot.com/p/what-is-pivlab.html
[pivmat]: http://www.fast.u-psud.fr/pivmat/
[pixi]: https://pixi.sh
[uv]: https://docs.astral.sh/uv/
[uvmat]: http://servforge.legi.grenoble-inp.fr/projects/soft-uvmat/wiki/UvmatHelp
