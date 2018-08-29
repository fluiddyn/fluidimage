.. FluidImage documentation master file

FluidImage documentation
========================

FluidImage is a libre Python framework for scientific processing of large
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
particular when the number of images to process becomes large. Thus we want
FluidImage to be able to run efficiently and easily on a personal computer and
on big clusters. The efficiency is achieved by using

- an asynchronous framework (and in the long term we want to be able to plug
  FluidImage to distributed computational systems like `Dask
  <http://dask.pydata.org>`_, `Spark <https://spark.apache.org/>`_ or `Storm
  <http://storm.apache.org/>`_),
- the available cores of the central processing units (CPU) and the available
  graphics processing units (GPU),
- good profiling and efficient and specialized algorithms,
- cutting-edge tools for fast computations with Python (in particular `Pythran
  <https://pythonhosted.org/pythran/>`_ and `Theano
  <http://deeplearning.net/software/theano>`_).


User Guide
----------

.. toctree::
   :maxdepth: 1

   overview
   install
   tutorial
   examples


Modules Reference
-----------------

Here is presented the general organization of the package (see also
:doc:`concepts_classes`) and the documentation of the modules, classes and
functions.

.. autosummary::
   :toctree: generated/

   fluidimage.topologies
   fluidimage.data_objects
   fluidimage.works
   fluidimage.calcul
   fluidimage.preproc
   fluidimage.calibration
   fluidimage.postproc
   fluidimage.util
   fluidimage.gui
   fluidimage.experimental

More
----

.. toctree::
   :maxdepth: 1

   FluidImage forge on Bitbucket <https://bitbucket.org/fluiddyn/fluidimage>
   FluidImage in PyPI  <https://pypi.python.org/pypi/fluidimage/>
   to_do
   changes
   authors
   for_dev


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
