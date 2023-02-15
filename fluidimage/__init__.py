"""
FluidImage
==========

"""

import sys
from subprocess import getoutput
from pathlib import Path

import numpy as np

from fluiddyn.util import create_object_from_file, get_memory_usage
from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles, SeriesOfArrays
from fluidimage.topologies.log import LogTopology

from ._version import __version__
from .util import (
    config_logging,
    imread,
    imsave,
    log_memory_usage,
    logger,
    print_memory_usage,
    reset_logger,
)

if any(
    any(test_tool in arg for arg in sys.argv)
    for test_tool in ("pytest", "unittest")
):
    print(
        "Fluidimage guesses that it is tested so it"
        " loads the Agg Matplotlib backend."
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda: None


def get_path_image_samples():
    try:
        from ._path_image_samples import path_image_samples
    except ImportError:
        pass
    else:
        if path_image_samples.exists():
            return path_image_samples

    path_image_samples = (
        Path.home() / ".local/fluidimage/repository/image_samples"
    )
    path_repo = path_image_samples.parent
    path_repo.mkdir(parents=True, exist_ok=True)
    if not path_image_samples.exists():
        getoutput(
            f"hg clone https://foss.heptapod.net/fluiddyn/fluidimage {path_repo}"
        )
    return path_image_samples


__all__ = [
    "__version__",
    "create_object_from_file",
    "get_memory_usage",
    "imread",
    "imsave",
    "logger",
    "config_logging",
    "reset_logger",
    "log_memory_usage",
    "print_memory_usage",
    "ParamContainer",
    "LogTopology",
    "get_path_image_samples",
    "SeriesOfArrays",
    "SerieOfArraysFromFiles",
    "np",
]
