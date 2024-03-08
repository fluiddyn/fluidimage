from pathlib import Path
from time import sleep

import pytest

from fluidimage.executors import supported_multi_executors
from fluidimage.piv import TopologyPIV

postfix = "test_piv"


@pytest.mark.parametrize("executor", supported_multi_executors)
def test_piv_oseen(tmp_path_oseen, executor):
    path_dir_images = tmp_path_oseen

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


def test_piv_jet(tmp_path_jet):
    path_dir_images = tmp_path_jet

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
