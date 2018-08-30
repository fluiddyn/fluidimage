
import unittest
from shutil import rmtree
from pathlib import Path

from fluiddyn.io import stdout_redirected

from fluidimage.experimental.topologies.bos_new import TopologyBOS
from fluidimage.experimental.executors.executor_await import (
    ExecutorAwaitMultiprocs
)

from fluidimage import path_image_samples


class TestBOSNew(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_input_files = path_image_samples / "Karman/Images"
        cls.postfix = "test_bos_new"

    @classmethod
    def tearDownClass(cls):
        paths = (cls.path_input_files,)
        for path in paths:
            path_out = Path(path.parent.as_posix() + "." + cls.postfix)
            if path_out.exists():
                rmtree(path_out)

    def test_bos_new_multiproc(self):
        params = TopologyBOS.create_default_params()

        params.series.path = str(self.path_input_files)
        params.series.ind_start = 1
        params.series.ind_step = 2

        params.piv0.shape_crop_im0 = 32
        params.multipass.number = 2
        params.multipass.use_tps = False

        params.mask.strcrop = ':, 50:500'

        # temporary, avoid a bug on Windows
        params.piv0.method_correl = "pythran"
        params.piv0.shape_crop_im0 = 16

        # compute only few vectors
        params.piv0.grid.overlap = -8

        params.saving.how = "recompute"
        params.saving.postfix = 'bos_test'

        with stdout_redirected():
            topology = TopologyBOS(params, logging_level="info")

            executer = ExecutorAwaitMultiprocs(
                topology,
                multi_executor=False,
                sleep_time=0.1,
                worker_limit=4,
                queues_limit=5,
            )
            topology.compute(executer)

            # remove one file
            path_file = next(Path(topology.path_dir_result).glob("bos*"))
            path_file.unlink()

            params.saving.how = "complete"
            topology = TopologyBOS(params, logging_level="info")
            executer = ExecutorAwaitMultiprocs(
                topology,
                multi_executor=False,
                sleep_time=0.1,
                worker_limit=4,
                queues_limit=5,
            )
            topology.compute(executer)


if __name__ == "__main__":
    unittest.main()
