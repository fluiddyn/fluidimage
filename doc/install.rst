Installation and advice
=======================


Dependencies
------------

- Python 2.7

- fftw (better to build with OpenMP enabled)


Install in development mode (recommended)
-----------------------------------------

FluidImage is still in alpha version ("testing for developers"!).  So I would
advice to work "as a developer", i.e. to get the source code and to use
revision control and the development mode of the Python installer.

For FluidImage, I use the revision control software Mercurial and the main
repository is hosted `here <https://bitbucket.org/fluiddyn/fluidimage>`_ in
Bitbucket. I would advice to fork this repository (click on "Fork") and to
clone your newly created repository to get the code on your computer (click on
"Clone" and run the command that will be given). If you are new with Mercurial
and Bitbucket, you can also read `this short tutorial
<http://fluiddyn.readthedocs.org/en/latest/mercurial_bitbucket.html>`_.

If you really don't want to use Mercurial, you can also just manually
download the package from `the Bitbucket page
<https://bitbucket.org/fluiddyn/fluidimage>`_ or from `the PyPI page
<https://pypi.python.org/pypi/fluidimage>`_.

To install in development mode (with a virtualenv)::

  python setup.py develop

or (without virtualenv)::

  python setup.py develop --user

Of course you can also install FluidDyn with the install command ``python
setup.py install``.

After the installation, it is a good practice to run the unit tests by
running ``python -m unittest discover`` from the root directory or
from any of the "test" directories (or just ``make tests``).

Installation with pip
---------------------

FluidImage can also be installed from the Python Package Index::

  pip install fluidimage --pre

The ``--pre`` option of pip allows the installation of a pre-release version.

However, the project is so new that it is better to have the last version (from
the mercurial repository hosted on Bitbucket).
