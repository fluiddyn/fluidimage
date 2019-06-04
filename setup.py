import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from logging import ERROR, INFO, DEBUG

from runpy import run_path

from distutils.sysconfig import get_config_var

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution


if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")


def install_setup_requires():
    dist = Distribution()
    # Honor setup.cfg's options.
    dist.parse_config_files(ignore_option_errors=True)
    if dist.setup_requires:
        dist.fetch_build_eggs(dist.setup_requires)


install_setup_requires()


from transonic.dist import ParallelBuildExt, get_logger

if "egg_info" in sys.argv:
    level = ERROR
else:
    level = INFO

logger = get_logger("fluidimage")
logger.setLevel(level)


fluid_build_ext = build_ext
try:
    from pythran.dist import PythranExtension

    try:
        # pythran > 0.8.6
        from pythran.dist import PythranBuildExt as fluid_build_ext
    except ImportError:
        pass
    use_pythran = True
except ImportError:
    use_pythran = False

here = Path(__file__).parent.absolute()

path_image_samples = here / "image_samples"
if path_image_samples.exists():
    with open("fluidimage/_path_image_samples.py", "w") as file:
        file.write(
            "from pathlib import Path\n\n"
            f'path_image_samples = Path(r"{path_image_samples}")\n'
        )

# Get the long description from the relevant file
with open("README.rst") as file:
    long_description = file.read()
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
    with open("fluidimage/_hg_rev.py", "w") as file:
        file.write(f'hg_rev = "{rev}"\n')


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


def transonize():

    from transonic.dist import make_backend_files

    paths = [
        "fluidimage/calcul/correl.py",
        "fluidimage/topologies/example.py",
        "fluidimage/calcul/interpolate/thin_plate_spline.py",
        "fluidimage/calcul/subpix.py",
    ]
    make_backend_files([here / path for path in paths])


def create_pythran_extensions():
    import numpy as np
    from transonic.dist import init_pythran_extensions

    compile_arch = os.getenv("CARCH", "native")
    extensions = init_pythran_extensions(
        "fluidimage",
        include_dirs=np.get_include(),
        compile_args=("-O3", f"-march={compile_arch}", "-DUSE_XSIMD"),
    )
    return extensions


def create_extensions():
    if "egg_info" in sys.argv:
        return []

    logger.info("Running fluidimage setup.py on platform " + sys.platform)

    transonize()

    ext_modules = create_pythran_extensions()

    logger.info(
        "The following extensions could be built if necessary:\n"
        + "".join([ext.name + "\n" for ext in ext_modules])
    )

    return ext_modules


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
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        # actually CeCILL License (GPL compatible license for French laws)
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["doc", "include", "scripts"]),
    scripts=["bin/fluidimviewer", "bin/fluidimlauncher"],
    entry_points={
        "console_scripts": ["fluidimviewer-pg = fluidimage.gui.pg_main:main"]
    },
    ext_modules=create_extensions(),
    cmdclass={"build_ext": ParallelBuildExt},
)
