import unittest
from shutil import rmtree
from pathlib import Path

from fluiddyn.io import stdout_redirected

from .image2image import TopologyImage2Image as Topo

from fluidimage import path_image_samples


class TestImage2Image(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_src = path_image_samples / "Karman/Images"
        cls.postfix = "test_im2im_old"

    @classmethod
    def tearDownClass(cls):
        path_out = Path(str(cls.path_src) + "." + cls.postfix)
        if path_out.exists():
            rmtree(path_out)

    def test_im2im(self):
        params = Topo.create_default_params()
        params.series.path = str(path_image_samples / "Karman/Images")

        params.series.ind_start = 1

        params.im2im = "fluidimage.preproc.image2image.Im2ImExample"
        params.args_init = ((1024, 2048), "clip")

        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

        with stdout_redirected():
            topology = Topo(params, logging_level=False)
            topology.compute()


if __name__ == "__main__":
    unittest.main()
