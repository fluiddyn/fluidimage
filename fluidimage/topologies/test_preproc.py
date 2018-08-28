
import unittest

import os
from glob import glob
from shutil import rmtree

from fluiddyn.io import stdout_redirected

from fluidimage.topologies.preproc import TopologyPreproc
from fluiddyn.io.image import imread, imsave

from fluidimage import path_image_samples


class TestPreprocKarman(unittest.TestCase):

    name = "Karman"

    @classmethod
    def setUpClass(cls):
        path_in = str(path_image_samples / cls.name / "Images")

        cls._work_dir = os.path.join(
            "test_fluidimage_topo_preproc_" + cls.name, "Images"
        )
        if not os.path.exists(cls._work_dir):
            os.makedirs(cls._work_dir)

        paths = glob(path_in + "/*")

        for path in paths:
            name = os.path.split(path)[-1]
            im = imread(path)
            im = im[::6, ::6]
            imsave(os.path.join(cls._work_dir, name), im, as_int=True)

    @classmethod
    def tearDownClass(cls):
        rmtree(os.path.split(cls._work_dir)[0], ignore_errors=True)

    def test_preproc(self):
        """Test preproc subpackage on image sample Karman with one index."""
        params = TopologyPreproc.create_default_params()

        params.preproc.series.path = self._work_dir
        params.preproc.series.strcouple = "i:i+3"
        params.preproc.series.ind_start = 1

        for tool in params.preproc.tools.available_tools:
            if "sliding" not in tool and "temporal" not in tool:
                tool = params.preproc.tools.__getitem__(tool)
                tool.enable = True

        params.preproc.saving.how = "recompute"
        params.preproc.saving.postfix = "preproc_test"

        with stdout_redirected():
            topology = TopologyPreproc(params, logging_level="debug")
            topology.compute()


class TestPreprocTime(TestPreprocKarman):
    name = "Jet"

    def test_preproc(self):
        """Test preproc subpackage on image sample Jet with two indices."""
        params = TopologyPreproc.create_default_params()

        params.preproc.series.path = self._work_dir
        params.preproc.series.strcouple = "i:i+2,1"
        params.preproc.series.ind_start = 60

        for tool in params.preproc.tools.available_tools:
            if "sliding" in tool or "temporal" in tool:
                tool = params.preproc.tools.__getitem__(tool)
                tool.enable = True

        params.preproc.saving.how = "recompute"
        params.preproc.saving.postfix = "preproc_test"

        with stdout_redirected():
            topology = TopologyPreproc(params, logging_level="debug")
            topology.compute()


if __name__ == "__main__":
    unittest.main()
