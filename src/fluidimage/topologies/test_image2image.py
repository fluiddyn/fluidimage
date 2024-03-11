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

    topology = TopologyImage2Image(params, logging_level="info")

    if executor == "multi_exec_async":
        topology.compute("exec_async", stop_if_error=True)

        topology = TopologyImage2Image(params, logging_level="info")

    topology.compute(executor, nb_max_workers=2)

    if executor != "multi_exec_async":
        return

    topology.make_code_graphviz(topology.path_dir_result / "topo.dot")
