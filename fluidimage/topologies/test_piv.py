
import unittest

from fluiddyn.io import stdout_redirected

from fluidimage.topologies.piv import TopologyPIV

from fluidimage import path_image_samples


class TestPIV(unittest.TestCase):
    def test_piv(self):
        params = TopologyPIV.create_default_params()

        params.series.path = str(path_image_samples / "Karman/Images")
        params.series.ind_start = 1

        # temporary, avoid a bug on Windows
        params.piv0.method_correl = "pythran"
        params.piv0.shape_crop_im0 = 16

        # compute only few vectors
        params.piv0.grid.overlap = -8

        params.multipass.number = 2
        params.multipass.use_tps = False

        params.saving.how = "recompute"
        params.saving.postfix = "piv_test"

        with stdout_redirected():
            topology = TopologyPIV(params, logging_level=False)
            topology.compute()


if __name__ == "__main__":
    unittest.main()
