import os
import sys
from pathlib import Path
from logging import ERROR, INFO
from distutils.util import strtobool

from runpy import run_path

from setuptools import setup, find_packages

from transonic.dist import ParallelBuildExt, get_logger
import setuptools_scm

if sys.version_info[:2] < (3, 9):
    raise RuntimeError("Python version >= 3.9 required.")

if "egg_info" in sys.argv:
    level = ERROR
else:
    level = INFO

logger = get_logger("fluidimage")
logger.setLevel(level)

build_dependencies_backends = {
    "pythran": ["pythran>=0.9.7"],
    "cython": ["cython"],
    "python": [],
    "numba": [],
}

TRANSONIC_BACKEND = os.environ.get("FLUIDIMAGE_TRANSONIC_BACKEND", "pythran")

if "DISABLE_PYTHRAN" in os.environ:
    DISABLE_PYTHRAN = strtobool(os.environ["DISABLE_PYTHRAN"])

    if (
        "FLUIDIMAGE_TRANSONIC_BACKEND" in os.environ
        and DISABLE_PYTHRAN
        and TRANSONIC_BACKEND == "pythran"
    ):
        raise ValueError

    if DISABLE_PYTHRAN:
        TRANSONIC_BACKEND = "python"


if TRANSONIC_BACKEND not in build_dependencies_backends:
    raise ValueError(
        f"FLUIDIMAGE_TRANSONIC_BACKEND={TRANSONIC_BACKEND} "
        f"not in {list(build_dependencies_backends.keys())}"
    )

setup_requires = []
setup_requires.extend(build_dependencies_backends[TRANSONIC_BACKEND])

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
    full_version = setuptools_scm.get_version()
except (LookupError, OSError):
    revision = "?"
else:
    try:
        revision = full_version.split("+")[1]
    except IndexError:
        revision = full_version
    if "." in revision:
        revision = revision.split(".")[0]
write_rev(revision)


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
    if "egg_info" in sys.argv or "dist_info" in sys.argv:
        return []

    logger.info("Running fluidimage setup.py on platform " + sys.platform)

    transonize()

    ext_modules = create_pythran_extensions()

    if ext_modules:
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
    url="https://foss.heptapod.net/fluiddyn/fluidimage",
    python_requires=">=3.9",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(exclude=["doc", "include", "scripts"]),
    scripts=["bin/fluidimviewer", "bin/fluidimlauncher"],
    entry_points={
        "console_scripts": ["fluidimviewer-pg = fluidimage.gui.pg_main:main"]
    },
    setup_requires=setup_requires,
    ext_modules=create_extensions(),
    cmdclass={"build_ext": ParallelBuildExt},
)
