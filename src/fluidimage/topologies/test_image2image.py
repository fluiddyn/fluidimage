from pathlib import Path

import pytest

from fluidimage.executors import supported_multi_executors
from fluidimage.image2image import TopologyImage2Image

postfix = "test_im2im"


@pytest.mark.parametrize("executor", supported_multi_executors)
def test_im2im(tmp_path_karman, executor):
    params = TopologyImage2Image.create_default_params()

    params.images.path = str(tmp_path_karman)

    params.im2im = "fluidimage.image2image.Im2ImExample"
    params.args_init = ((1024, 2048), "clip")

    params.saving.how = "recompute"
    params.saving.postfix = postfix

    if executor == "multi_exec_async":
        topology = TopologyImage2Image(params, logging_level="info")
        topology.compute("exec_async", stop_if_error=True)

        # remove files
        path_files = list(Path(topology.path_dir_result).glob("*.bmp"))
        assert len(path_files) == 4
        for path in path_files:
            path.unlink()

    topology = TopologyImage2Image(params, logging_level="info")
    topology.compute(executor)
    assert len(topology.results) == 4

    # remove one file
    path_files = list(Path(topology.path_dir_result).glob("*.bmp"))
    assert len(path_files) == 4
    path_files[0].unlink()

    params.saving.how = "complete"
    topology = TopologyImage2Image(params, logging_level="info")
    topology.compute(executor, nb_max_workers=2)
    assert len(topology.results) == 1

    path_files = list(Path(topology.path_dir_result).glob("*.bmp"))
    assert len(path_files) == 4

    if executor != "multi_exec_async":
        return

    topology.make_code_graphviz(topology.path_dir_result / "topo.dot")
