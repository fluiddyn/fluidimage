import os
import shutil
from pathlib import Path
from runpy import run_path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fluidimage import get_path_image_samples


def _show(*args, **kwargs):
    pass


plt.show = _show

os.environ["FLUIDIMAGE_TESTS_EXAMPLES"] = "1"


def clean_example_data():
    for path_samples in (
        Path(__file__).absolute().parent.parent.parent / "image_samples",
        get_path_image_samples(),
    ):
        assert path_samples.exists(), path_samples
        for path in path_samples.rglob("*_example"):
            print("delete", path)
            shutil.rmtree(path, ignore_errors=True)


def setup_module(module):
    clean_example_data()


def teardown_module(module):
    clean_example_data()


exclude = set(
    [
        "submit_job_legi.py",
        "surface_tracking.py",
        "preproc_sback1_filter.py",
        "preproc_sback2_rescale.py",
    ]
)

scripts = sorted(
    path.name
    for path in Path(__file__).absolute().parent.glob("*.py")
    if path.name not in exclude
)


@pytest.mark.parametrize("script", scripts)
def test_script(script):
    run_path(script)
