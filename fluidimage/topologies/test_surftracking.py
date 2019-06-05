import unittest
from pathlib import Path
from shutil import rmtree

import skimage

from fluidimage import path_image_samples
from fluidimage.topologies.surface_tracking import TopologySurfaceTracking


def rescale_intensity(tuple_image_path):
    """
    Rescale image intensities, between the specified minima and maxima,
    by using a multiplicative factor.

    ----------
    minima, maxima : float
        Sets the range to which current intensities have to be rescaled.

    """
    img, path = tuple_image_path
    # the processing can be adjusted depending on the value of the path.
    print("process file:\n" + path)
    minima = 0
    maxima = 4095
    out_range = (minima, maxima)
    img_out = skimage.exposure.rescale_intensity(img, out_range=out_range)
    return img_out, path


class TestSurfaceTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_src = path_image_samples / "SurfTracking/Images"
        cls.postfix = "test_surftracking_new"
        cls.path_out = Path(str(cls.path_src) + "." + cls.postfix)

    @classmethod
    def tearDownClass(cls):
        # path_out = Path(str(cls.path_src) + "." + cls.postfix)
        if cls.path_out.exists():
            rmtree(cls.path_out)

    def test_surftrack(self):
        params = TopologySurfaceTracking.create_default_params()

        params.images.path = str(self.path_src)
        params.images.path_ref = str(self.path_src)
        params.images.str_slice = ":4:2"
        params.images.str_slice_ref = ":3"

        params.surface_tracking.xmin = 200
        params.surface_tracking.xmax = 250
        params.surface_tracking.correct_pos = True

        params.preproc.im2im = rescale_intensity

        params.saving.how = "recompute"
        params.saving.path = str(self.path_out)
        params.saving.postfix = self.postfix

        topology = TopologySurfaceTracking(params, logging_level="info")
        topology.make_code_graphviz(topology.path_dir_result / "topo.dot")
        topology.compute(executor="exec_sequential", stop_if_error=True)
        # topology.compute(nb_max_workers=1)
        # topology.compute(sequential=True, stop_if_error=True)


if __name__ == "__main__":
    unittest.main()
