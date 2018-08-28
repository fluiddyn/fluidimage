
import os
import subprocess
import sys
from datetime import datetime

from runpy import run_path

from distutils.sysconfig import get_config_var
from setuptools import setup, find_packages

import numpy as np

try:
    from pythran.dist import PythranExtension

    use_pythran = True
except ImportError:
    use_pythran = False

here = os.path.abspath(os.path.dirname(__file__))

path_image_samples = os.path.join(here, "image_samples")
print(path_image_samples)
if os.path.exists(path_image_samples):
    with open("fluidimage/_path_image_samples.py", "w") as f:
        f.write(
            "from pathlib import Path\n"
            'path_image_samples = Path("{}")\n'.format(path_image_samples))

# I have not yet manage to use Pythran on Windows...
if sys.platform == "win32":
    use_pythran = False

# Get the long description from the relevant file
with open("README.rst") as f:
    long_description = f.read()
lines = long_description.splitlines(True)
for i, line in enumerate(lines):
    if line.endswith(":alt: Code coverage\n"):
        iline_coverage = i
        break

long_description = "".join(lines[iline_coverage + 2 :])

# Get the version from the relevant file
d = run_path("fluidimage/_version.py")
__version__ = d["__version__"]


def write_rev(rev):
    with open("fluidimage/_hg_rev.py", "w") as f:
        f.write('hg_rev = "{}"\n'.format(rev))


try:
    hg_rev = subprocess.check_output(["hg", "id", "--id"]).decode("utf-8").strip()
    write_rev(hg_rev)
except (OSError, subprocess.CalledProcessError):
    try:
        hg_rev = (
            "git:"
            + subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        write_rev(hg_rev)
    except (OSError, subprocess.CalledProcessError):
        pass

install_requires = ["fluiddyn >= 0.2.0"]


install_requires.extend(
    [
        "scipy >= 0.14.1",
        "numpy >= 1.8",
        "matplotlib >= 1.4.2",
        "pyfftw >= 0.10.4",
        # 'scikit-image >= 0.12.3',
        "h5py",
        "h5netcdf",
        "imageio",
        "pims",
        "trio",
    ]
)


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)


def make_pythran_extensions(modules):
    develop = sys.argv[-1] == "develop"
    extensions = []
    for mod in modules:
        base_file = mod.replace(".", os.path.sep)
        py_file = base_file + ".py"
        # warning: does not work on Windows
        suffix = get_config_var("EXT_SUFFIX") or ".so"
        bin_file = base_file + suffix
        if (
            not develop
            or not os.path.exists(bin_file)
            or modification_date(bin_file) < modification_date(py_file)
        ):
            print('pythran extension "' + mod + '" needs to be built')
            pext = PythranExtension(mod, [py_file])
            pext.include_dirs.append(np.get_include())
            extensions.append(pext)
    return extensions


if use_pythran:
    ext_modules = make_pythran_extensions(
        [
            "fluidimage.calcul.correl_pythran",
            "fluidimage.calcul.interpolate.tps_pythran",
            "fluidimage.calcul.subpix_pythran",
            "fluidimage.experimental.cpu_bounded_task_examples_pythran",
        ]
    )
else:
    ext_modules = []

setup(
    name="fluidimage",
    version=__version__,
    description=("fluid image processing with Python."),
    long_description=long_description,
    keywords="PIV",
    author="Pierre Augier",
    author_email="pierre.augier@legi.cnrs.fr",
    url="https://bitbucket.org/fluiddyn/fluidimage",
    python_requires=">=3.6",
    license="CeCILL",
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        # actually CeCILL License (GPL compatible license for French laws)
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(exclude=["doc", "include", "scripts"]),
    install_requires=install_requires,
    scripts=[
        "bin/fluidimviewer",
        "bin/fluidimlauncher",
        "bin/fluidimviewer-pg",
        "bin/fluidimslideshow-pg",
    ],
    ext_modules=ext_modules,
)
