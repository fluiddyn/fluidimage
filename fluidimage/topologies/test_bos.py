
import unittest

from fluiddyn.io import stdout_redirected

from fluidimage.topologies.bos import TopologyBOS
from fluidimage.topologies.log import LogTopology

from fluidimage import path_image_samples


class TestPIV(unittest.TestCase):
    def test_piv(self):
        params = TopologyBOS.create_default_params()

        params.series.path = str(path_image_samples / "Karman/Images")
        params.series.ind_start = 1
        params.series.ind_step = 2

        params.piv0.shape_crop_im0 = 32
        params.multipass.number = 2
        params.multipass.use_tps = False

        params.mask.strcrop = ':, 50:500'

        # temporary, avoid a bug on Windows
        params.piv0.method_correl = "pythran"
        params.piv0.shape_crop_im0 = 16

        # compute only few vectors
        params.piv0.grid.overlap = -8

        params.saving.how = "recompute"
        params.saving.postfix = 'bos_test'

        with stdout_redirected():
            topology = TopologyBOS(params, logging_level="info")
            topology.compute()

            log = LogTopology(topology.path_dir_result)
        log.plot_durations()
        log.plot_nb_workers()
        log.plot_memory()


if __name__ == "__main__":
    unittest.main()
