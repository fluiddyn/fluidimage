import os
import unittest
from pathlib import Path
from shutil import rmtree

from fluiddyn.io.image import imread, imsave

from fluidimage import get_path_image_samples
from fluidimage.works.preproc import Work


class TestPreprocKarman(unittest.TestCase):
    name = "Karman"

    @classmethod
    def setUpClass(cls):
        path_in = get_path_image_samples() / cls.name / "Images"

        cls._work_dir = (
            Path("test_fluidimage_topo_preproc_" + cls.name) / "Images"
        )

        if not cls._work_dir.exists():
            cls._work_dir.mkdir(parents=True)

        for path in sorted(path_in.glob("*")):
            name = path.name
            im = imread(path)
            im = im[::6, ::6]
            imsave(cls._work_dir / name, im, as_int=True)

    @classmethod
    def tearDownClass(cls):
        rmtree(os.path.split(cls._work_dir)[0], ignore_errors=True)

    def test_preproc(self):
        """Test preproc subpackage on image sample Karman with one index."""
        params = Work.create_default_params()

        params.preproc.series.path = self._work_dir

        for tool in params.preproc.tools.available_tools:
            if "sliding" not in tool and "temporal" not in tool:
                tool = params.preproc.tools.__getitem__(tool)
                tool.enable = True

        preproc = Work(params)
        preproc.display(1, hist=True)


class TestPreprocTime(TestPreprocKarman):
    name = "Jet"

    def test_preproc(self):
        """Test preproc subpackage on image sample Jet with two indices."""
        params = Work.create_default_params()

        params.preproc.series.path = self._work_dir
        params.preproc.series.str_subset = "i,0"
        params.preproc.series.ind_start = 60

        for tool in params.preproc.tools.available_tools:
            if "sliding" in tool:
                tool = params.preproc.tools.__getitem__(tool)
                tool.enable = True

        preproc = Work(params)
        preproc.display(60, hist=False)
