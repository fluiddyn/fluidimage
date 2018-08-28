
import unittest

from fluiddyn.io import stdout_redirected

from fluidimage.topologies.image2image import TopologyImage2Image as Topo

from fluidimage import path_image_samples


class TestPIV(unittest.TestCase):
    def test_piv(self):
        params = Topo.create_default_params()
        params.series.path = str(path_image_samples / "Karman/Images")

        params.series.ind_start = 1

        params.im2im = "fluidimage.preproc.image2image.Im2ImExample"
        params.args_init = ((1024, 2048), "clip")

        params.saving.how = "recompute"
        params.saving.postfix = "pre_test"

        with stdout_redirected():
            topology = Topo(params, logging_level=False)
            topology.compute()


if __name__ == "__main__":
    unittest.main()
