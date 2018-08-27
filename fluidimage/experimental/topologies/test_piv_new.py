
import unittest
from shutil import rmtree
from pathlib import Path

from fluiddyn.io import stdout_redirected

from fluidimage.experimental.topologies.piv_new import TopologyPIV
from fluidimage.experimental.executors.executor_await import (
    ExecutorAwaitMultiprocs
)

here = Path(__file__).parent.absolute()


class TestPivNew(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_Oseen = here / "../../../image_samples/Oseen/Images/Oseen*"
        cls.path_Jet = here / "../../../image_samples/Jet/Images/c06*"
        cls.postfix = "test_piv_new"

    @classmethod
    def tearDownClass(cls):
        paths = (cls.path_Oseen, cls.path_Jet)
        for path in paths:
            path_out = Path(path.parent.as_posix() + "." + cls.postfix)
            if path_out.exists():
                rmtree(path_out)

    def test_piv_new(self):
        params = TopologyPIV.create_default_params()

        params.series.path = self.path_Oseen.as_posix()
        params.series.ind_start = 1
        params.series.ind_step = 1

        params.piv0.shape_crop_im0 = 32
        params.multipass.number = 2
        params.multipass.use_tps = True

        # params.saving.how has to be equal to 'complete' for idempotent jobs
        # (on clusters)
        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

        with stdout_redirected():
            topology = TopologyPIV(params, logging_level="info")
            # topology.make_code_graphviz('topo.dot')

            executer = ExecutorAwaitMultiprocs(
                topology,
                multi_executor=False,
                sleep_time=0.1,
                worker_limit=4,
                queues_limit=5,
            )

            topology.compute(executer)
            # executer.compute()

    def test_piv_new_multiproc(self):
        params = TopologyPIV.create_default_params()

        params.series.path = self.path_Jet.as_posix()
        params.series.ind_start = 60
        params.series.ind_step = 1
        params.series.strcouple = "i, 0:2"

        params.piv0.shape_crop_im0 = 128
        params.multipass.number = 2
        params.multipass.use_tps = True

        # params.saving.how has to be equal to 'complete' for idempotent jobs
        # (on clusters)
        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

        with stdout_redirected():
            topology = TopologyPIV(params, logging_level="info")
            # topology.make_code_graphviz('topo.dot')

            executer = ExecutorAwaitMultiprocs(
                topology,
                multi_executor=True,
                sleep_time=0.1,
                worker_limit=4,
                queues_limit=5,
            )
            topology.compute(executer)
            # executer.compute()


if __name__ == "__main__":
    unittest.main()
