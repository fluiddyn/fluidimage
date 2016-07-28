
import unittest

import os

from fluiddyn.io import stdout_redirected

from fluidimage.topologies.pre_proc import TopologyPreproc

here = os.path.abspath(os.path.dirname(__file__))


class TestPreproc(unittest.TestCase):
    def test_preproc(self):
        params = TopologyPreproc.create_default_params()

        params.preproc.series.path = os.path.join(
            here, '..', '..', 'image_samples', 'Karman', 'Images')
        params.preproc.series.strcouple = 'i:i+3'
        params.preproc.series.ind_start = 1

        params.preproc.tools.temporal_median.enable = True
        params.preproc.tools.global_threshold.enable = True

        params.preproc.saving.how = 'recompute'
        params.preproc.saving.postfix = 'preproc_test'

        with stdout_redirected():
            topology = TopologyPreproc(params, logging_level=False)
            topology.compute()

if __name__ == '__main__':
    unittest.main()
