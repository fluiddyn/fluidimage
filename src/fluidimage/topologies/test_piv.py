import shutil
import sys
import unittest
from pathlib import Path
from time import sleep

import pytest

from fluidimage import get_path_image_samples
from fluidimage.piv import TopologyPIV

path_image_samples = get_path_image_samples()

on_linux = sys.platform == "linux"
postfix = "test_piv"

skip_except_on_linux = unittest.skipIf(not on_linux, "Only supported on Linux")


def create_tmp_dir_image(tmp_path, name):

    path_dir_images = tmp_path / "Images"
    path_dir_images.mkdir()

    for path_im in (path_image_samples / name).glob("Images/*"):
        shutil.copy(path_im, path_dir_images)

    return path_dir_images


@skip_except_on_linux
@pytest.mark.parametrize("executor", [None, "multi_exec_subproc"])
def test_piv_oseen(tmp_path, executor):

    path_dir_images = create_tmp_dir_image(tmp_path, "Oseen")

    params = TopologyPIV.create_default_params()

    params.series.path = str(path_dir_images)

    params.piv0.shape_crop_im0 = 32
    params.multipass.number = 2
    params.multipass.use_tps = True

    params.saving.how = "recompute"
    params.saving.postfix = postfix

    topology = TopologyPIV(params, logging_level="info")

    topology.make_code_graphviz(topology.path_dir_result / "topo.dot")
    topology.compute(executor)


@skip_except_on_linux
def test_piv_jet(tmp_path):

    path_dir_images = create_tmp_dir_image(tmp_path, "Jet")

    params = TopologyPIV.create_default_params()

    params.series.path = str(path_dir_images)

    params.piv0.shape_crop_im0 = 128
    params.multipass.number = 2
    params.multipass.use_tps = True

    params.saving.how = "recompute"
    params.saving.postfix = postfix

    topology = TopologyPIV(params, logging_level="info")
    topology.compute()

    topology = TopologyPIV(params, logging_level="info")
    topology.compute(nb_max_workers=2)

    # remove one file to test params.saving.how = "complete"
    path_files = list(Path(topology.path_dir_result).glob("piv*"))

    if not path_files:
        sleep(0.2)
        path_files = list(Path(topology.path_dir_result).glob("piv*"))

    path_files[0].unlink()

    params.saving.how = "complete"
    topology = TopologyPIV(params, logging_level="debug")
    topology.compute("exec_sequential")

    assert len(topology.results) == 1
