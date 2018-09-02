
import unittest
from shutil import rmtree
from pathlib import Path

from fluiddyn.io import stdout_redirected

from fluidimage.experimental.topologies.image2image_new import TopologyImage2Image

from fluidimage import path_image_samples


class TestPivNew(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_src = path_image_samples / "Karman/Images"
        cls.postfix = "test_im2im_new"

    @classmethod
    def tearDownClass(cls):
        path_out = Path(str(cls.path_src) + "." + cls.postfix)
        if path_out.exists():
            rmtree(path_out)

    def test_piv_new(self):
        params = TopologyImage2Image.create_default_params()

        params.series.path = str(self.path_src)

        params.series.ind_start = 1

        params.im2im = "fluidimage.preproc.image2image.Im2ImExample"
        params.args_init = ((1024, 2048), "clip")

        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

        with stdout_redirected():
            topology = TopologyImage2Image(params, logging_level="info")

            topology.make_code_graphviz(topology.path_dir_result / "topo.dot")
            topology.compute()


if __name__ == "__main__":
    unittest.main()
