import os
import unittest
from glob import glob
from shutil import rmtree

from fluiddyn.io import stdout_redirected
from fluiddyn.io.image import imread, imsave
from fluidimage import path_image_samples
from fluidimage.preproc.base import PreprocBase


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
        params = PreprocBase.create_default_params()

        params.preproc.series.path = self._work_dir

        for tool in params.preproc.tools.available_tools:
            if "sliding" not in tool and "temporal" not in tool:
                tool = params.preproc.tools.__getitem__(tool)
                tool.enable = True

        preproc = PreprocBase(params)
        preproc()
        preproc.display(1, hist=True)


class TestPreprocTime(TestPreprocKarman):
    name = "Jet"

    def test_preproc(self):
        """Test preproc subpackage on image sample Jet with two indices."""
        params = PreprocBase.create_default_params()

        params.preproc.series.path = self._work_dir

        for tool in params.preproc.tools.available_tools:
            if "sliding" in tool:
                tool = params.preproc.tools.__getitem__(tool)
                tool.enable = True

        preproc = PreprocBase(params)
        preproc()
        preproc.display(1, hist=False)


if __name__ == "__main__":
    unittest.main()
