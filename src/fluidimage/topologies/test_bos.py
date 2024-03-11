from pathlib import Path

import pytest

from fluidimage.bos import TopologyBOS
from fluidimage.executors import supported_multi_executors

postfix = "test_bos"


@pytest.mark.parametrize("executor", supported_multi_executors)
def test_bos_new_multiproc(tmp_path_karman, executor):
    params = TopologyBOS.create_default_params()

    params.images.path = str(tmp_path_karman)
    params.images.str_subset = "1:3"

    params.piv0.shape_crop_im0 = 32
    params.multipass.number = 2
    params.multipass.use_tps = False

    params.mask.strcrop = ":, 50:500"

    # temporary, avoid a bug on Windows
    params.piv0.method_correl = "pythran"
    params.piv0.shape_crop_im0 = 16

    # compute only few vectors
    params.piv0.grid.overlap = -8

    params.saving.how = "recompute"
    params.saving.postfix = postfix

    topology = TopologyBOS(params, logging_level="info")
    topology.compute("exec_async", stop_if_error=True)

    # remove one file
    path_files = list(Path(topology.path_dir_result).glob("bos*"))
    path_files[0].unlink()

    params.saving.how = "complete"
    topology = TopologyBOS(params, logging_level="info")
    topology.compute(executor, nb_max_workers=2)
