from pathlib import Path

import pytest

from fluidimage.executors import supported_multi_executors
from fluidimage.preproc import TopologyPreproc


def test_preproc_exec_async_sequential(tmp_path_jet_small):
    """Test preproc subpackage on image sample Jet with two indices."""
    params = TopologyPreproc.create_default_params()

    params.series.path = str(tmp_path_jet_small)
    params.series.str_subset = "i:i+2,1"

    for tool in params.tools.available_tools:
        if "temporal" in tool:
            tool = params.tools.__getitem__(tool)
            tool.enable = True

    params.saving.how = "recompute"
    params.saving.postfix = "preproc_test"

    topology = TopologyPreproc(params, logging_level="debug")
    topology.compute("exec_async_sequential")
    assert len(topology.results) == 1

    params.saving.how = "complete"
    topology = TopologyPreproc(params, logging_level="debug")
    topology.compute("exec_async_sequential")
    assert len(topology.results) == 0


@pytest.mark.parametrize("executor", supported_multi_executors)
def test_preproc_multi_exec(tmp_path_jet_small, executor):

    params = TopologyPreproc.create_default_params()

    params.series.path = str(tmp_path_jet_small)

    params.saving.how = "recompute"
    params.saving.postfix = "preproc_test_" + executor

    topology = TopologyPreproc(params, logging_level="debug")
    topology.compute(executor, nb_max_workers=2)
    assert len(topology.results) == 4

    path_files = sorted(Path(topology.path_dir_result).glob("*.png"))
    for i in range(2):
        path_files[i].unlink()

    params.saving.how = "complete"
    topology = TopologyPreproc(params, logging_level="info")
    topology.compute(executor, nb_max_workers=2)
    assert len(topology.results) == 2
