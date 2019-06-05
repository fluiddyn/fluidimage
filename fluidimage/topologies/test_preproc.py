import unittest
from pathlib import Path
from shutil import rmtree

from fluiddyn.io.image import imread, imsave
from fluidimage import path_image_samples
from fluidimage.topologies.preproc import TopologyPreproc


class TestPreprocTemporal(unittest.TestCase):
    name = "Jet"

    @classmethod
    def setUpClass(cls):
        path_in = path_image_samples / cls.name / "Images"

        cls._work_dir = Path("test_topo_preproc_" + cls.name) / "Images"
        cls._work_dir.mkdir(parents=True, exist_ok=True)

        paths = path_in.glob("*")

        for path in paths:
            name = path.name
            im = imread(path)
            im = im[::6, ::6]
            imsave(str(cls._work_dir / name), im, as_int=True)

    @classmethod
    def tearDownClass(cls):
        rmtree(cls._work_dir.parent, ignore_errors=True)

    def test_preproc(self):
        """Test preproc subpackage on image sample Jet with two indices."""
        params = TopologyPreproc.create_default_params()

        params.preproc.series.path = self._work_dir
        params.preproc.series.strcouple = "i:i+2,1"
        params.preproc.series.ind_start = 60

        for tool in params.preproc.tools.available_tools:
            if "temporal" in tool:
                tool = params.preproc.tools.__getitem__(tool)
                tool.enable = True

        params.preproc.saving.how = "recompute"
        params.preproc.saving.postfix = "preproc_test"

        topology = TopologyPreproc(params, logging_level="debug")
        topology.compute("exec_async_sequential")
        assert len(topology.results) == 1

        params.preproc.saving.how = "complete"
        topology = TopologyPreproc(params, logging_level="debug")
        topology.compute("exec_async_sequential")
        assert len(topology.results) == 0


if __name__ == "__main__":
    unittest.main()
