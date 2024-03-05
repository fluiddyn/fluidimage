import unittest
from pathlib import Path
from shutil import rmtree

import pytest

from fluidimage import get_path_image_samples
from fluidimage._opencv import error_import_cv2
from fluidimage.topologies.optical_flow import Topology

path_image_samples = get_path_image_samples()


class TestPivNew(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_Oseen = path_image_samples / "Oseen/Images/Oseen*"
        cls.postfix = "test_optical_flow"

    @classmethod
    def tearDownClass(cls):
        paths = (cls.path_Oseen,)
        for path in paths:
            path_out = Path(str(path.parent) + "." + cls.postfix)
            if path_out.exists():
                rmtree(path_out, ignore_errors=True)

    def test_optical_flow(self):

        if error_import_cv2:
            with pytest.raises(ModuleNotFoundError):
                Topology.create_default_params()
            return

        params = Topology.create_default_params()

        params.series.path = str(self.path_Oseen)

        # params.saving.how has to be equal to 'complete' for idempotent jobs
        # (on clusters)
        params.saving.how = "recompute"
        params.saving.postfix = self.postfix

        params.filters.displacement_max = 10.0

        topology = Topology(params, logging_level="info")

        topology.make_code_graphviz(topology.path_dir_result / "topo.dot")
        topology.compute("exec_sequential")
