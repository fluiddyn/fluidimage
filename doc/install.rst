Installation and advice
=======================

See our general advice `on using Python
<https://fluiddyn.readthedocs.io/en/latest/advice_on_Python.html>`__ and and
`on installing a good scientific Python environment
<https://fluiddyn.readthedocs.io/en/latest/get_good_Python_env.html>`__.

The simplest method (with conda)
--------------------------------

The simplest method to install FluidImage is to use ``conda`` (installed with
miniconda) and the conda-forge channel (activated with the command ``conda
config --add channels conda-forge``).

To just install FluidImage, you can run::

  conda install fluidimage

Alternativally, you can install FluidImage in a dedicated conda environment
with::

  conda create -n env_fluidimage fluidimage

If you use an environment, you will need to activate it with ``conda activate
env_fluidimage``.

Slightly more complicated: with pip, from the package on PyPI or the repository
-------------------------------------------------------------------------------

FluidImage depends on Python >= 3.6 and on Python packages that are today very
simple to install with pip or conda, namely numpy, scipy, matplotlib, h5py,
scikit-image, pyfftw and IPython. You should not care about these dependencies
because they are going to be installed automatically for you.

PyQt5 is used only for some graphical user interfaces, so you need to install
it manually if needed. This can be done with ``pip install pyqt5`` or ``conda
install pyqt``.

I would also advice to install Jupyterlab, which interacts nicely with FluidImage.

To compile, or not to compile?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We choose to use the static Python compiler `Pythran
<https://github.com/serge-sans-paille/pythran>`__ for some numerical functions.
Our microbenchmarks show that the performances are as good as what we are able
to get with Fortran or C++!

We don't upload "wheels" (package with already compiled binaries) on PyPI, so
installing with pip *can* involve local compilation. Some code in FluidImage
will be compiled *only if* `Pythran
<https://github.com/serge-sans-paille/pythran>`__ is available during
installation.

therefore, to install FluidImage with compilation, you need to install Pythran
before FluidImage, which can be done with::

  pip install pythran colorlog

Moreover, you need a C++ compiler (we recommend clang). With conda, it's very
easy to install clang with ``conda install clangdev``.

Finally, for better performance, Pythran needs a configuration file with
something like (see the `Pythran documentation
<https://pythran.readthedocs.io/en/latest/MANUAL.html>`__)::

  [pythran]
  complex_hook = True

  [compiler]
  CXX = clang++
  CC = clang

Ready? Let's install with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the last version of FluidImage uploaded to the Python Package Index::

  pip install fluidimage -U

However, the project is in an active phase of development so it can be better
to use the last version (from the mercurial repository hosted on Heptapod).
Moreover, like that, you get all examples and tutorials! For FluidImage, we use
the revision control software Mercurial and the main repository is hosted in
https://foss.heptapod.net/fluiddyn/fluidimage, so you can get the source with
the command::

  hg clone https://foss.heptapod.net/fluiddyn/fluidimage

If you are new with Mercurial and Heptapod, you can also read `this short
tutorial <http://fluiddyn.readthedocs.org/en/latest/mercurial_heptapod.html>`_.

If you really can't use Mercurial, you can also just manually download the
package from `the Heptapod page
<https://foss.heptapod.net/fluiddyn/fluidimage>`_ or from `the PyPI page
<https://pypi.python.org/pypi/fluidimage>`_.

To install in development mode (with a virtualenv or with conda)::

  cd fluidimage
  pip install -e .

or (without virtualenv)::

  pip install -e . --user

After the installation, it is a good practice to run the unit tests by running
``pytest`` from the root directory.
