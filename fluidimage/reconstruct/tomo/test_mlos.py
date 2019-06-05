import shutil
import unittest
from pathlib import Path
from tempfile import gettempdir

import matplotlib

matplotlib.use("Agg")

from fluiddyn.io import stdout_redirected
from fluidimage import path_image_samples
from fluidimage.data_objects.tomo import ArrayTomoCV
from fluidimage.reconstruct.tomo import TomoMLOSCV


path_calib = str(path_image_samples / "TomoPIV" / "calibration" / "cam0.h5")
path_particle = str(
    path_image_samples / "TomoPIV" / "particle" / "cam0.pre" / "im00001a.tif"
)
path_output = Path(gettempdir()) / "fluidimage_test_mlos"


class TestMLOS(unittest.TestCase):
    """Test fluidimage.reconstruct.tomo.mlos module."""

    def tearDown(self):
        shutil.rmtree(path_output)

    def test(self):
        """Test classes TomoMLOSCV and ArrayTomoCV."""
        with stdout_redirected():
            tomo = TomoMLOSCV(
                path_calib,
                xlims=(-10, 10),
                ylims=(-10, 10),
                zlims=(-5, 5),
                nb_voxels=(11, 11, 5),
            )
            tomo.verify_projection()
            pix = tomo.phys2pix("cam0")
            tomo.array.init_paths(path_particle, path_output)
            tomo.reconstruct(pix, path_particle, threshold=None, save=True)

            path_result = list(path_output.glob("*"))[0]
            array = ArrayTomoCV(h5file_path=path_result)
            array.describe()
            array.load_dataset(copy=True)
            array.plot_slices(0, 1)
            array.clear()


if __name__ == "__main__":
    unittest.main()
