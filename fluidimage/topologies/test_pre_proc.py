
import unittest

import os

from fluiddyn.io import stdout_redirected

from fluidimage.topologies.pre_proc import TopologyPreproc

here = os.path.abspath(os.path.dirname(__file__))


class TestPreproc(unittest.TestCase):
    def test_preproc(self):
        '''Test pre_proc subpackage on image sample Karman with one index.'''
        params = TopologyPreproc.create_default_params()

        params.preproc.series.path = os.path.join(
            here, '..', '..', 'image_samples', 'Karman', 'Images')
        params.preproc.series.strcouple = 'i:i+3'
        params.preproc.series.ind_start = 1

        for tool in params.preproc.tools.available_tools:
            if 'sliding' not in tool and 'temporal' not in tool:
                tool = params.preproc.tools.__getitem__(tool)
                tool.enable = True

        params.preproc.saving.how = 'recompute'
        params.preproc.saving.postfix = 'preproc_test'

        with stdout_redirected():
            topology = TopologyPreproc(params, logging_level=False)
            topology.compute()

    def test_preproc_two_indices(self):
        '''Test pre_proc subpackage on image sample Jet with two indices.'''
        params = TopologyPreproc.create_default_params()

        params.preproc.series.path = os.path.join(
            here, '..', '..', 'image_samples', 'Jet', 'Images')
        params.preproc.series.strcouple = 'i:i+2,1'
        params.preproc.series.ind_start = 60

        for tool in params.preproc.tools.available_tools:
            if 'sliding' in tool or 'temporal' in tool:
                tool = params.preproc.tools.__getitem__(tool)
                tool.enable = True

        params.preproc.saving.how = 'recompute'
        params.preproc.saving.postfix = 'preproc_test'

        with stdout_redirected():
            topology = TopologyPreproc(params, logging_level=False)
            topology.compute()


if __name__ == '__main__':
    unittest.main()
