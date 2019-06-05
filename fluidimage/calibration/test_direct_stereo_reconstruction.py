import os
import unittest

import h5py
import matplotlib.pyplot as plt

from fluidimage import path_image_samples
from fluidimage.calibration import CalibDirect, DirectStereoReconstruction
from fluidimage.calibration.util import get_plane_equation

pathbase = path_image_samples / "4th_PIV-Challenge_Case_E"


def get_piv_field(path):

    try:
        with h5py.File(path, "r") as f:
            keyspiv = [key for key in f.keys() if key.startswith("piv")]
            keyspiv.sort()
            key = keyspiv[-1]
            X = f[key]["xs"][...]
            Y = f[key]["ys"][...]
            dx = f[key]["deltaxs_final"][...]
            dy = f[key]["deltays_final"][...]
    except Exception:
        print(path)
        raise

    return X, Y, dx, dy


class TestCalib(unittest.TestCase):
    """Test fluidimage.calibration DirectStereoReconstruction, CalibDirect."""

    def test(self):

        path_cam1 = pathbase / "E_Calibration_Images" / "Camera_01"
        path_cam3 = pathbase / "E_Calibration_Images" / "Camera_03"

        path_calib1 = path_cam1 / "calib1.npy"
        path_calib3 = path_cam3 / "calib3.npy"

        nb_pixelx, nb_pixely = 1024, 1024
        nbline_x, nbline_y = 32, 32

        calib = CalibDirect(path_cam1 / "img*", (nb_pixelx, nb_pixely))
        calib.compute_interpolents()
        calib.compute_interpolents_pixel2line(nbline_x, nbline_y, test=False)

        calib.check_interp_lines(4)
        plt.close("all")
        calib.check_interp_lines_coeffs(2)
        plt.close("all")
        calib.check_interp_levels(2)
        plt.close("all")

        calib3 = CalibDirect(
            os.path.join(path_cam3, "img*"), (nb_pixelx, nb_pixely)
        )

        calib3.compute_interpolents()
        calib3.compute_interpolents_pixel2line(nbline_x, nbline_y, test=False)

        calib.save(path_calib1)
        calib3.save(path_calib3)

        postfix = ".piv"
        name = "piv_00001-00002.h5"
        path_im = pathbase / "E_Particle_Images"

        path_piv1 = path_im / ("Camera_01" + postfix) / name
        path_piv3 = path_im / ("Camera_03" + postfix) / name

        z0 = 0
        alpha = 0
        beta = 0
        a, b, c, d = get_plane_equation(z0, alpha, beta)

        Xl, Yl, dxl, dyl = get_piv_field(path_piv1)
        Xr, Yr, dxr, dyr = get_piv_field(path_piv3)

        stereo = DirectStereoReconstruction(path_calib1, path_calib3)
        X0, X1, d0cam, d1cam = stereo.project2cam(
            Xl, Yl, dxl, dyl, Xr, Yr, dxr, dyr, a, b, c, d, check=False
        )
        X, Y, Z = stereo.find_common_grid(X0, X1, a, b, c, d)

        dx, dy, dz, erx, ery, erz = stereo.reconstruction(
            X0, X1, d0cam, d1cam, a, b, c, d, X, Y, check=False
        )


# dt = 0.001
# dx, dy, dz = dx/dt, dy/dt, dz/dt
# erx, ery, erz = erx/dt, ery/dt, erz/dt

if __name__ == "__main__":
    unittest.main()
