import unittest

import h5py
import matplotlib.pyplot as plt
import numpy as np

from fluidimage import get_path_image_samples
from fluidimage.calibration import Calibration

plt.show = lambda: 0


pathbase = get_path_image_samples() / "Milestone"


def get_piv_field(path):
    try:
        with h5py.File(path, "r") as file:
            keyspiv = [key for key in file.keys() if key.startswith("piv")]
            keyspiv.sort()
            key = keyspiv[-1]
            X = file[key]["xs"][...]
            Y = file[key]["ys"][...]
            dx = file[key]["deltaxs_final"][...]
            dy = file[key]["deltays_final"][...]
    except Exception:
        print(path)
        raise

    return X, Y, dx, dy


class TestCalib(unittest.TestCase):
    """Test fluidimage.calibration DirectStereoReconstruction, CalibDirect."""

    def test(self):
        path_calib = pathbase / "PCO_top.xml"
        path_v = pathbase / "piv_0000a-b.h5"
        nbypix = 2160

        calib = Calibration(path_calib)

        X, Y, dx, dy = get_piv_field(path_v)

        Xphys, Yphys, Zphys, dxphys, dyphys, dzphys = calib.pix2phys_UV(
            X, Y, dx, dy, index_level=0, nbypix=nbypix
        )
        X, Y = calib.phys2pix(Xphys, Yphys, np.nanmean(Zphys))


if __name__ == "__main__":
    unittest.main()
