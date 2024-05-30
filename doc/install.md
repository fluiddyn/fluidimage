# Installation

First, ensure that you have a recent Python installed, since Fluidimage requires Python
\>= 3.9. Some issues regarding the installation of Python and Python packages are
discussed in
[the main documentation of the Fluiddyn project](http://fluiddyn.readthedocs.org/en/latest/install.html).

Here, we describe installation methods that do not involve local compilation. One can
also install Fluidimage from source as described [here](./build-from-source.md).

## Install with pip

```{note}

We strongly advice to install Fluidimage in a virtual environment. See the
official guide [Install packages in a virtual environment using pip and
venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

```

Fluidimage can be installed without compilation with `pip`:

```sh
pip install pip -U
pip install fluidimage
```

### Optional dependencies

Fluidimage has 2 sets of optional dependencies, which can be installed with commands like
`pip install fluidimage[opencv]` or `pip install fluidimage[opencv, pims]`:

- `opencv`: [OpenCV](https://opencv.org/) can be used for some algorithms,
- `pims`: [pims: Python Image Sequence](https://github.com/soft-matter/pims) is used to
  read `.cine` files.

## Install the conda-forge package with conda or mamba

We recommend installing `conda` and `mamba` with the
[miniforge installer](https://github.com/conda-forge/miniforge) so that the installed
packages will be uploaded from the [conda-forge] channel. Then, one can run:

```sh
mamba install fluidimage
```

One can also create a dedicated environment:

```sh
mamba create -n env_fluidimage fluidimage
```

The environment can then be activated with `conda activate env_fluidimage`.

[conda-forge]: https://conda-forge.org/
