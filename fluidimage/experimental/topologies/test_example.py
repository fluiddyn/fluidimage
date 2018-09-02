import unittest
from shutil import rmtree

from fluiddyn.io import stdout_redirected

from fluidimage.experimental.topologies.example import TopologyExample

from fluidimage import path_image_samples

path_input = path_image_samples / "Karman/Images"


class TestPivNew(unittest.TestCase):
    def tearDown(self):
        path_out = self.topology.path_dir_result
        if path_out.exists():
            rmtree(path_out)

    def test_piv_new(self):

        with stdout_redirected():
            self.topology = TopologyExample(path_input, logging_level="debug")
            self.topology.compute()


if __name__ == "__main__":
    unittest.main()
