import unittest
from pathlib import Path
from shutil import rmtree

from fluidimage import path_image_samples
from fluidimage.topologies.image2image import TopologyImage2Image


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

        params.images.path = str(self.path_src)

        params.im2im = "fluidimage.preproc.image2image.Im2ImExample"
        params.args_init = ((1024, 2048), "clip")

        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

        topology = TopologyImage2Image(params, logging_level="info")
        topology.compute("exec_async", stop_if_error=True)

        topology = TopologyImage2Image(params, logging_level="info")
        topology.compute()

        topology.make_code_graphviz(topology.path_dir_result / "topo.dot")


if __name__ == "__main__":
    unittest.main()
