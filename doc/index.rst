.. FluidImage documentation master file

FluidImage documentation
========================

FluidImage is a free Python software to compute scientific quantities
from many images.

The development is just starting and you won't do anything useful with
FluidImage as it is now.

We now focus on the computation of simple `Particle Image Velocimetry
<https://en.wikipedia.org/wiki/Particle_image_velocimetry>`_.

We want to make FluidImage easy (useful documentation, easy
installation, usable with scripts and GUI in Qt), reliable (with good
unittests) and very efficient, in particular when the number of images
to treat becomes large. Thus we want FluidImage to be able to run
efficiently and easily on a personal computer and on a big
cluster. The efficiency will be achieved by using

- an asynchronous framework (and in the long term we want to be able to
  plug FluidImage to distributed computational systems like `Storm
  <http://storm.apache.org/>`_),
- the available cores of the central processing units (CPU) and the
  available graphics processing units (GPU),
- good profiling and efficient and specialized algorithms,
- cutting-edge tools for fast computations with Python (in particular
  `Pythran <https://pythonhosted.org/pythran/>`_ and `Theano
  <http://deeplearning.net/software/theano>`_).


User Guide
----------

.. toctree::
   :maxdepth: 1

   overview
   install


Modules Reference
-----------------

.. autosummary::
   :toctree: generated/

   fluidimage.calcul
   fluidimage.data_objects
   fluidimage.topologies
   fluidimage.works
   fluidimage.waiting_queues
   fluidimage.gui     

More
----

.. toctree::
   :maxdepth: 1

   FluidImage forge on Bitbucket <https://bitbucket.org/fluiddyn/fluidimage>
   FluidImage in PyPI  <https://pypi.python.org/pypi/fluidimage/>
   to_do
   changes
   for_dev


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
