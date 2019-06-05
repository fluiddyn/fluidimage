import unittest
from pathlib import Path
from shutil import rmtree

from fluidimage import path_image_samples
from fluidimage.topologies.optical_flow import TopologyOpticalFlow


class TestPivNew(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_Oseen = path_image_samples / "Oseen/Images/Oseen*"
        cls.path_Jet = path_image_samples / "Jet/Images/c06*"
        cls.postfix = "test_optical_flow"

    @classmethod
    def tearDownClass(cls):
        paths = (cls.path_Oseen, cls.path_Jet)
        for path in paths:
            path_out = Path(str(path.parent) + "." + cls.postfix)
            if path_out.exists():
                rmtree(path_out)

    def test_optical_flow(self):
        params = TopologyOpticalFlow.create_default_params()

        params.series.path = str(self.path_Oseen)
        params.series.ind_start = 1
        params.series.ind_step = 1

        # params.saving.how has to be equal to 'complete' for idempotent jobs
        # (on clusters)
        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

        params.filters.displacement_max = 10.0

        topology = TopologyOpticalFlow(params, logging_level="info")

        topology.make_code_graphviz(topology.path_dir_result / "topo.dot")
        topology.compute("exec_sequential")


if __name__ == "__main__":
    unittest.main()
