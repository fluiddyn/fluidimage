"""
FluidImage
==========

"""

import os
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

if any("pytest" in part for part in sys.argv):
    print(
        "Fluidimage guesses that it is tested so it"
        " loads the Agg Matplotlib backend."
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda: None


def _get_user_data_dir(appname: str) -> Path:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path(os.getenv("XDG_DATA_HOME", "~/.local/share")).expanduser()
    return ans.joinpath(appname)


def get_path_image_samples():

    # First try next to this file in case of editable install
    path_image_samples = Path(__file__).absolute().parent / "../../image_samples"
    if path_image_samples.exists():
        return path_image_samples.resolve()

    # Gitlab and Github CI
    for name_env_var_project_dir in ("CI_PROJECT_DIR", "GITHUB_WORKSPACE"):
        ci_project_dir = os.getenv(name_env_var_project_dir)
        if ci_project_dir is not None:
            path_image_samples = Path(ci_project_dir) / "image_samples"
            if path_image_samples.exists():
                return path_image_samples

    path_image_samples = (
        _get_user_data_dir("fluidimage") / "repository/image_samples"
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
