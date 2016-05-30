FluidImage
==========

|release| |docs|

.. |release| image:: https://img.shields.io/pypi/v/fluidimage.svg
   :target: https://pypi.python.org/pypi/fluidimage/
   :alt: Latest version

.. |docs| image:: https://readthedocs.org/projects/fluidimage/badge/?version=latest
   :target: http://fluidimage.readthedocs.org
   :alt: Documentation status

FluidImage is a Python framework for scientific treatments of large
series of images.  Today, this package is in a very early stage of
development and the only treatment available is Particle Image
Velocimetry (PIV), i.e. computation of velocity fields by correlations
of images.

What do we want?
----------------

There is still work to do, but we want:

- easy installation

- cross-platform

- no need for Matlab

- better written nearly all in Python (Cython or Pythran ok)

- well tested

- well documented
  
- efficient

  * able to use GPU
  
  * parallel, asynchronous, distributed

  * running on cluster

- today, only a very minimal GUI (Qt)

  * display images (zoom, colorbar, colormaps, info on a pixel)

  * display vectors (and information on a selected vector)
