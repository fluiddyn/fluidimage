
import unittest

import os

from fluiddyn.io import stdout_redirected

from fluidimage.topologies.image2image import TopologyImage2Image as Topo

here = os.path.abspath(os.path.dirname(__file__))


class TestPIV(unittest.TestCase):
    def test_piv(self):
        params = Topo.create_default_params()

        params.series.path = os.path.join(
            here, '..', '..', 'image_samples', 'Karman', 'Images')
        params.series.ind_start = 1

        params.im2im = 'fluidimage.preproc.image2image.Im2ImExample'
        params.args_init = ((1024, 2048), 'clip')

        params.saving.how = 'recompute'
        params.saving.postfix = 'pre_test'

        with stdout_redirected():
            topology = Topo(params, logging_level=False)
            topology.compute()


if __name__ == '__main__':
    unittest.main()
