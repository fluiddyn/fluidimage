FluidImage
==========

|release| |docs| |coverage| |heptapod_ci| |github_actions|

.. |release| image:: https://img.shields.io/pypi/v/fluidimage.svg
   :target: https://pypi.python.org/pypi/fluidimage/
   :alt: Latest version

.. |docs| image:: https://readthedocs.org/projects/fluidimage/badge/?version=latest
   :target: http://fluidimage.readthedocs.org
   :alt: Documentation status

.. |coverage| image:: https://codecov.io/gh/fluiddyn/fluidimage/branch/branch%2Fdefault/graph/badge.svg
   :target: https://codecov.io/gh/fluiddyn/fluidimage/branch/branch%2Fdefault/
   :alt: Code coverage

.. |heptapod_ci| image:: https://foss.heptapod.net/fluiddyn/fluidimage/badges/branch/default/pipeline.svg
   :target: https://foss.heptapod.net/fluiddyn/fluidimage/-/pipelines
   :alt: Heptapod CI

.. |github_actions| image:: https://github.com/fluiddyn/fluidimage/actions/workflows/ci.yml/badge.svg?branch=branch/default
   :target: https://github.com/fluiddyn/fluidimage/actions
   :alt: Github Actions

FluidImage is a libre Python framework for scientific processing of large
series of images.

**Documentation:** http://fluidimage.readthedocs.org

Image processing for fluid mechanics is highly dominated by proprietary tools.
Such tools are not ideal when you want to understand and tweak the processes
and/or to use clusters. With the improvement of the open-source tools for
scientific computing and collaborative development, one can think it is
possible to build together a good library/toolkit specialized in image
processing for fluid mechanics. This is our project with FluidImage.

This package is young but already good enough to be used "in production" to

- display and pre-process images,

- compute displacement or velocity fields with `Particle Image Velocimetry
  <https://en.wikipedia.org/wiki/Particle_image_velocimetry (PIV)>`_ (PIV, i.e.
  displacements of pattern obtained by correlations of cropped images) and
  `optical flow <https://en.wikipedia.org/wiki/Optical_flow>`_,

- analyze and display PIV fields.

We want to make FluidImage easy (useful documentation, easy installation,
usable with scripts and GUI in Qt), reliable (with good `unittests
<https://codecov.io/gh/fluiddyn/fluidimage/>`_) and very efficient, in
particular when the number of images to process becomes large. Thus we want
FluidImage to be able to run efficiently and easily on a personal computer and
on big clusters. The efficiency is achieved by using

- a framework for asynchronous computations (currently, we use `Trio
  <https://trio.readthedocs.io>`_ + multiprocessing, and in the long term we want
  to be able to plug FluidImage to distributed computational systems like `Dask
  <http://dask.pydata.org>`_, `Spark <https://spark.apache.org/>`_ or `Storm
  <http://storm.apache.org/>`_),

- the available cores of the central processing units (CPU) and the available
  graphics processing units (GPU),

- good profiling and efficient and specialized algorithms,

- cutting-edge tools for fast computations with Python (in particular `Pythran
  <https://pythonhosted.org/pythran/>`_ and `Theano
  <http://deeplearning.net/software/theano>`_).
