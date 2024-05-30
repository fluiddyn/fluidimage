# FluidImage

[![Latest version](https://img.shields.io/pypi/v/fluidimage.svg)](https://pypi.python.org/pypi/fluidimage/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/fluidimage.svg)
[![Documentation status](https://readthedocs.org/projects/fluidimage/badge/?version=latest)](http://fluidimage.readthedocs.org)
[![Code coverage](https://codecov.io/gh/fluiddyn/fluidimage/branch/branch%2Fdefault/graph/badge.svg)](https://codecov.io/gh/fluiddyn/fluidimage/branch/branch%2Fdefault/)
[![Heptapod CI](https://foss.heptapod.net/fluiddyn/fluidimage/badges/branch/default/pipeline.svg)](https://foss.heptapod.net/fluiddyn/fluidimage/-/pipelines)
[![Github Actions Linux](https://github.com/fluiddyn/fluidimage/actions/workflows/ci-linux.yml/badge.svg?branch=branch/default)](https://github.com/fluiddyn/fluidimage/actions/workflows/ci-linux.yml)
[![Github Actions Pixi](https://github.com/fluiddyn/fluidimage/actions/workflows/ci-pixi.yml/badge.svg?branch=branch/default)](https://github.com/fluiddyn/fluidimage/actions/workflows/ci-pixi.yml)

<!-- start description -->

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
with FluidImage. Fluidimage can be thought as a partial rewrite in Python of [UVmat] with
a focus on performance and usability. Moreover, we try to integrate good ideas taken from
[OpenPIV], [PIVlab] and [PIVmat].

This package is quite young but can be used to

- display and pre-process images,

- compute displacement or velocity fields with Particle Image Velocimetry ([PIV], i.e.
  displacements of pattern obtained by correlations of cropped images),
  Background-Oriented Schlieren
  ([BOS](https://en.wikipedia.org/wiki/Background-oriented_schlieren_technique)) and
  [optical flow](https://en.wikipedia.org/wiki/Optical_flow),

- analyze and display PIV fields.

We want to make FluidImage easy (useful documentation, easy installation, good API,
usable with simple scripts and simple graphical user interfaces), reliable (with good
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

[openpiv]: http://www.openpiv.net/
[piv]: https://en.wikipedia.org/wiki/Particle_image_velocimetry%20(PIV)
[pivlab]: https://pivlab.blogspot.com/p/what-is-pivlab.html
[pivmat]: http://www.fast.u-psud.fr/pivmat/
[uvmat]: http://servforge.legi.grenoble-inp.fr/projects/soft-uvmat/wiki/UvmatHelp
