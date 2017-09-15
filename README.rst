FluidImage
==========

|release| |docs| |coverage| |travis|

.. |release| image:: https://img.shields.io/pypi/v/fluidimage.svg
   :target: https://pypi.python.org/pypi/fluidimage/
   :alt: Latest version

.. |docs| image:: https://readthedocs.org/projects/fluidimage/badge/?version=latest
   :target: http://fluidimage.readthedocs.org
   :alt: Documentation status

.. |coverage| image:: https://codecov.io/bb/fluiddyn/fluidimage/branch/default/graph/badge.svg
   :target: https://codecov.io/bb/fluiddyn/fluidimage/branch/default/
   :alt: Code coverage

.. |travis| image:: https://travis-ci.org/fluiddyn/fluidimage.svg?branch=master
    :target: https://travis-ci.org/fluiddyn/fluidimage

FluidImage is a libre Python framework for scientific treatments of large
series of images. This package is very young but it is already mature enough to
be used "in production" to

- pre-process images,
- compute `Particle Image Velocimetry
  <https://en.wikipedia.org/wiki/Particle_image_velocimetry (PIV)>`_ (PIV,
  i.e. displacements of pattern obtained by correlations of cropped images),
- analyze PIV fields.

We want to make FluidImage easy (useful documentation, easy installation,
usable with scripts and GUI in Qt), reliable (with good `unittests
<https://codecov.io/bb/fluiddyn/fluidimage/>`_) and very efficient, in
particular when the number of images to treat becomes large. Thus we want
FluidImage to be able to run efficiently and easily on a personal computer and
on big clusters. The efficiency is achieved by using

- an asynchronous framework (and in the long term we want to be able to plug
  FluidImage to distributed computational systems like `Storm
  <http://storm.apache.org/>`_),
- the available cores of the central processing units (CPU) and the available
  graphics processing units (GPU),
- good profiling and efficient and specialized algorithms,
- cutting-edge tools for fast computations with Python (in particular `Pythran
  <https://pythonhosted.org/pythran/>`_ and `Theano
  <http://deeplearning.net/software/theano>`_).
