import unittest
from tempfile import gettempdir

import matplotlib

matplotlib.use("agg")

from fluiddyn.io import stdout_redirected
from fluidimage.reconstruct.tomo import TomoMLOSCV

from fluidimage import path_image_samples


path_calib = str(path_image_samples / "TomoPIV" / "calibration" / "cam0.h5")
path_particle = str(
    path_image_samples / "TomoPIV" / "particle" / "cam0.pre" / "im00001a.tif"
)


class TestMLOS(unittest.TestCase):
    """Test fluidimage.reconstruct.tomo.mlos module."""

    def test(self):
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
            tomo.array.init_paths(path_particle, gettempdir())
            tomo.reconstruct(pix, path_particle, threshold=None, save=False)


if __name__ == "__main__":
    unittest.main()
