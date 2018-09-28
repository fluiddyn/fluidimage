import unittest
from shutil import rmtree
from pathlib import Path

from fluiddyn.io import stdout_redirected

from fluidimage.topologies.log import LogTopology

from fluidimage import path_image_samples

from .piv import TopologyPIV


class TestPIV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_input_files = path_image_samples / "Karman/Images"
        cls.postfix = "test_piv_old"

    @classmethod
    def tearDownClass(cls):
        path = cls.path_input_files
        path_out = Path(str(path) + "." + cls.postfix)
        if path_out.exists():
            rmtree(path_out)

    def test_piv(self):
        params = TopologyPIV.create_default_params()

        params.series.path = str(self.path_input_files)
        params.series.ind_start = 1

        # temporary, avoid a bug on Windows
        params.piv0.method_correl = "pythran"
        params.piv0.shape_crop_im0 = 16

        # compute only few vectors
        params.piv0.grid.overlap = -8

        params.multipass.number = 2
        params.multipass.use_tps = False

        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

        with stdout_redirected():
            topology = TopologyPIV(params, logging_level="info")
            topology.compute()

            log = LogTopology(topology.path_dir_result)
        log.plot_durations()
        log.plot_nb_workers()
        log.plot_memory()


if __name__ == "__main__":
    unittest.main()
