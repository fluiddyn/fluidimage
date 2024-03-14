from pathlib import Path

import pytest

from fluidimage.executors import supported_multi_executors
from fluidimage.piv import TopologyPIV

postfix = "test_piv"


@pytest.mark.usefixtures("close_plt_figs")
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
    topology.compute(executor, nb_max_workers=2)

    log = topology.read_log_data()
    assert log.topology_name == "fluidimage.topologies.piv.TopologyPIV"
    assert log.nb_max_workers == 2

    if [len(log.durations[key]) for key in ("compute_piv", "save_piv")] != [3, 3]:
        print("Issue with this log file:")
        print(log.path_log_file.read_text())

    assert len(log.durations["compute_piv"]) == 3
    assert len(log.durations["save_piv"]) == 3

    log.plot_durations()
    log.plot_memory()
    log.plot_nb_workers()

    path_files = list(Path(topology.path_dir_result).glob("piv*"))
    for i in range(2):
        path_files[i].unlink()

    params.saving.how = "complete"
    topology = TopologyPIV(params, logging_level="info")
    topology.compute(executor, nb_max_workers=2)
    assert len(topology.results) == 2

    topology.make_code_graphviz(topology.path_dir_result / "topo.dot")


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
    path_files[0].unlink()

    params.saving.how = "complete"
    topology = TopologyPIV(params, logging_level="debug")
    topology.compute("exec_sequential")

    assert len(topology.results) == 1
