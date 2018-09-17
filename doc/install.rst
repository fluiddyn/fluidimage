Installation and advice
=======================

Dependencies and useful packages
--------------------------------

- Python >= 3.6

- FFTW

- numpy, scipy, matplotlib, h5py

- fluiddyn

- pyfftw (simplest way to compute fft quite efficiently)

- `Pythran <https://github.com/serge-sans-paille/pythran>`_

  We choose to use the new static Python compiler `Pythran
  <https://github.com/serge-sans-paille/pythran>`_ for some functions of the
  operators. Our microbenchmarks show that the performances are as good as what
  we are able to get with Fortran or C++!

  .. warning::

     To reach good performance, we advice to try to put in the file
     `~/.pythranrc` the lines (it seems to work well on Linux, see the `Pythran
     documentation <https://pythonhosted.org/pythran/MANUAL.html>`_):

     .. code:: bash

        [pythran]
        complex_hook = True

  .. warning::

     The compilation of C++ files produced by Pythran can be long and can
     consume a lot of memory. If you encounter any problems, you can try to use
     clang (for example with ``conda install clangdev``) and to enable its use
     in the file `~/.pythranrc` with:

     .. code:: bash

        [compiler]
        CXX = clang++
        CC = clang

- h5netcdf (only if you need netcdf files)

- scikit-image (only for preprocessing of images)

- PyQt5 (only for GUI)

- IPython (important to play interactively with parameters, images and results)

- Jupyter (to try the tutorials yourself)

The simplest way to get a good environment for fluidimage is by using conda
(with anaconda or miniconda). If you use conda, install the main packages
with::

  conda config --add channels conda-forge

  conda install numpy scipy matplotlib h5py imageio scikit-image pyqt \
                ipython jupyterlab clangdev fftw

and the other packages with pip::

  pip install pyfftw pythran h5netcdf colorlog fluiddyn

Install in development mode (recommended)
-----------------------------------------

FluidImage is still in beta version ("testing for users").  So it can be good
to work "as a developer", i.e. to get the source code and to use revision
control and the development mode of the Python installer.

For FluidImage, we use the revision control software Mercurial and the main
repository is hosted `here <https://bitbucket.org/fluiddyn/fluidimage>`_ in
Bitbucket, so you can get the source with the command::

  hg clone https://bitbucket.org/fluiddyn/fluidimage

I would advice to fork this repository (click on "Fork") and to
clone your newly created repository to get the code on your computer (click on
"Clone" and run the command that will be given). If you are new with Mercurial
and Bitbucket, you can also read `this short tutorial
<http://fluiddyn.readthedocs.org/en/latest/mercurial_bitbucket.html>`_.

If you really don't want to use Mercurial, you can also just manually
download the package from `the Bitbucket page
<https://bitbucket.org/fluiddyn/fluidimage>`_ or from `the PyPI page
<https://pypi.python.org/pypi/fluidimage>`_.

To install in development mode (with a virtualenv or with conda)::

  cd fluidimage
  python setup.py develop

or (without virtualenv)::

  python setup.py develop --user

Of course you can also install FluidDyn with the install command ``python
setup.py install``.

After the installation, it is a good practice to run the unit tests by running
``python -m unittest discover`` (or just ``make tests``) from the root
directory or from any of the "test" directories.

Installation with pip
---------------------

FluidImage can also be installed from the Python Package Index::

  pip install fluidimage

However, the project is in an active phase of development so it can be better
to use the last version (from the mercurial repository hosted on Bitbucket).
