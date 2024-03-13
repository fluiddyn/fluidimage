from pathlib import Path

import pytest

from fluidimage._opencv import error_import_cv2
from fluidimage.executors import supported_multi_executors
from fluidimage.optical_flow import Topology

postfix = "test_optical_flow"


@pytest.mark.parametrize("executor", supported_multi_executors)
def test_optical_flow(tmp_path_oseen, executor):

    if error_import_cv2:
        with pytest.raises(ModuleNotFoundError):
            Topology.create_default_params()
        return

    params = Topology.create_default_params()

    params.series.path = str(tmp_path_oseen)

    params.saving.how = "recompute"
    params.saving.postfix = postfix

    params.filters.displacement_max = 10.0

    topology = Topology(params, logging_level="info")
    topology.compute(executor, nb_max_workers=2)

    path_files = sorted(Path(topology.path_dir_result).glob("piv_*.h5"))
    for i in range(2):
        path_files[i].unlink()

    params.saving.how = "complete"
    topology = Topology(params, logging_level="info")
    topology.compute(executor, nb_max_workers=2)
    assert len(topology.results) == 2

    if executor != "multi_exec_async":
        return

    topology.make_code_graphviz(topology.path_dir_result / "topo.dot")
