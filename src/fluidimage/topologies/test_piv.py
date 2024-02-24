import sys
import unittest
from pathlib import Path
from shutil import rmtree
from time import sleep

from fluidimage import get_path_image_samples
from fluidimage.topologies.piv import TopologyPIV

path_image_samples = get_path_image_samples()

on_linux = sys.platform == "linux"


@unittest.skipIf(not on_linux, "Only supported on Linux")
class TestPivNew(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_Oseen = path_image_samples / "Oseen/Images/Oseen*"
        cls.path_Jet = path_image_samples / "Jet/Images/c06*"
        cls.postfix = "test_piv"

    @classmethod
    def tearDownClass(cls):
        paths = (cls.path_Oseen, cls.path_Jet)
        for path in paths:
            path_out = Path(str(path.parent) + "." + cls.postfix)
            if path_out.exists():
                rmtree(path_out, ignore_errors=True)

    def test_piv_new(self):
        params = TopologyPIV.create_default_params()

        params.series.path = str(self.path_Oseen)
        params.series.ind_start = 1
        params.series.ind_step = 1

        params.piv0.shape_crop_im0 = 32
        params.multipass.number = 2
        params.multipass.use_tps = True

        # params.saving.how has to be equal to 'complete' for idempotent jobs
        # (on clusters)
        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

        topology = TopologyPIV(params, logging_level="info")

        topology.make_code_graphviz(topology.path_dir_result / "topo.dot")
        topology.compute()

    def test_piv_new_multiproc(self):
        params = TopologyPIV.create_default_params()

        params.series.path = str(self.path_Jet)
        params.series.ind_start = 60
        params.series.ind_step = 1
        params.series.str_subset = "i, 0:2"

        params.piv0.shape_crop_im0 = 128
        params.multipass.number = 2
        params.multipass.use_tps = True

        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

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