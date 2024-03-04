"""
FluidImage
==========

"""

import subprocess
import sys
from pathlib import Path
from shutil import which

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
    path_image_samples = (
        Path.home() / ".local/fluidimage/repository/image_samples"
    )
    path_repo = path_image_samples.parent
    path_repo.mkdir(parents=True, exist_ok=True)
    if not path_image_samples.exists():
        cmd = None
        hg = which("hg")
        if hg is not None:
            cmd = hg
            path_https = "https://foss.heptapod.net/fluiddyn/fluidimage"
        else:
            git = which("git")
            if git is not None:
                cmd = git
                path_https = "https://github.com/fluiddyn/fluidimage/"
        if cmd is not None:
            subprocess.run(
                [cmd, "clone", path_https, str(path_repo)], check=False
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
