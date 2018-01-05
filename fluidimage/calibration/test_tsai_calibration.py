
import unittest
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from fluidimage.calibration import Calibration

plt.show = lambda: 0

here = os.path.abspath(os.path.dirname(__file__))
path_fluidimage = os.path.split(os.path.split(here)[0])[0]

pathbase = os.path.join(path_fluidimage,
                        'image_samples', 'Milestone')


def get_piv_field(path):

    try:
        with h5py.File(path, 'r') as f:
            keyspiv = [key for key in f.keys() if key.startswith('piv')]
            keyspiv.sort()
            key = keyspiv[-1]
            X = f[key]['xs'].value
            Y = f[key]['ys'].value
            dx = f[key]['deltaxs_final'].value
            dy = f[key]['deltays_final'].value
    except Exception:
        print(path)
        raise

    return X, Y, dx, dy


class TestCalib(unittest.TestCase):
    """Test fluidimage.calibration DirectStereoReconstruction, CalibDirect."""

    def test(self):
        path_calib = os.path.join(pathbase, 'PCO_top.xml')
        path_v = os.path.join(pathbase, 'piv_0000a-b.h5')
        nbypix = 2160

        calib = Calibration(path_calib)

        X, Y, dx, dy = get_piv_field(path_v)

        Xphys, Yphys, Zphys, dxphys, dyphys, dzphys = calib.pix2phys_UV(
            X, Y, dx, dy, index_level=0, nbypix=nbypix)
        X, Y = calib.phys2pix(Xphys, Yphys, np.nanmean(Zphys))


if __name__ == "__main__":
    unittest.main()
