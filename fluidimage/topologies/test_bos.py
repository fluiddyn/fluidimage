import unittest
from pathlib import Path
from shutil import rmtree

from fluidimage import path_image_samples
from fluidimage.topologies.bos import TopologyBOS


class TestBOSNew(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_input_files = path_image_samples / "Karman/Images"
        cls.postfix = "test_bos_new"

    @classmethod
    def tearDownClass(cls):
        path = cls.path_input_files
        path_out = Path(str(path) + "." + cls.postfix)
        if path_out.exists():
            rmtree(path_out)

    def test_bos_new_multiproc(self):
        params = TopologyBOS.create_default_params()

        params.images.path = str(self.path_input_files)
        params.images.str_slice = "1:3"

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
        params.saving.postfix = self.postfix

        topology = TopologyBOS(params, logging_level="info")
        topology.compute("exec_async", stop_if_error=True)

        # remove one file
        path_files = list(Path(topology.path_dir_result).glob("bos*"))
        path_files[0].unlink()

        params.saving.how = "complete"
        topology = TopologyBOS(params, logging_level="info")
        topology.compute()


if __name__ == "__main__":
    unittest.main()
