import unittest
from shutil import rmtree

import h5py
import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage import get_path_image_samples
from fluidimage.postproc.piv import ArrayPIV, PIV2d, get_grid_pixel_from_piv_file


class TestPIV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_piv = get_path_image_samples() / "Milestone/piv_0000a-b.h5"
        cls.path_tests = cls.path_piv.parent / "tmp_tests"
        cls.path_tests.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.path_tests.exists():
            rmtree(cls.path_tests, ignore_errors=True)

    def test_get_grid(self):
        xs1d, ys1d = get_grid_pixel_from_piv_file(self.path_piv)

    def test_from_MultipassPIVResults(self):
        PIV2d.from_file(self.path_piv)

    def test_piv_objects(self):
        ny, nx = 8, 12

        xs1d = np.linspace(-1, 1, nx)
        ys1d = np.linspace(-1, 1, ny)
        xs2d, ys2d = np.meshgrid(xs1d, ys1d)
        piv0 = PIV2d(xs1d, ys1d, 0, xs2d, ys2d)

        piv0.display()
        piv0.extract(0, 0.5, 0, 0.8, phys=True)

        piv1 = (2 * piv0 + 1) / 3 + piv0

        piv1.compute_spatial_fft()
        piv1.compute_divh()
        piv1.compute_rotz()
        piv1.compute_norm()

        arr = ArrayPIV((piv0,))
        arr.append(piv1)
        arr.extend(
            (piv0 - 1, 1 + piv1 * piv0 - piv0 / piv0 - 1, 0 + piv0 * 3 + 1)
        )

        arr.set_timestep(0.1)
        assert len(arr.times) == 5
        assert arr.times.max() == 0.4

        arr.compute_temporal_fft()

        arr1 = 0 + 2 * arr * 3 + 1 - 2
        arr1 = (1 + arr1 / 2).median_filter(3).gaussian_filter(0.5)

        piv2 = arr1[0]
        arr1[1] = piv2
        repr(arr1)
        len(arr1)
        del arr1[-1]

        arr1 = arr1.truncate().extract(0, 4, 2, 7).extract_square()

        with h5py.File(self.path_piv, "r") as file:
            params = ParamContainer(hdf5_object=file["params"])

        path = self.path_tests / "piv2d.h5"
        piv2.save(path, params=params)
        piv_loaded = PIV2d.from_file(path, load_params=True)

        assert np.allclose(piv2.x, piv_loaded.x)
        assert np.allclose(piv2.vx, piv_loaded.vx)
        assert piv2.namevx == piv_loaded.namevx
        assert piv2.unitvx == piv_loaded.unitvx
        assert params == piv_loaded.params

    def test_compute_time_average(self):
        ny, nx = 8, 12

        xs1d = np.linspace(-1, 1, nx)
        ys1d = np.linspace(-1, 1, ny)
        xs2d, ys2d = np.meshgrid(xs1d, ys1d)
        piv = PIV2d(xs1d, ys1d, 0, xs2d, ys2d)
        nt = 5

        arr = ArrayPIV([piv] * nt)

        piv_mean = arr.compute_time_average()

        assert np.allclose(piv.vx, piv_mean.vx)
        assert np.allclose(piv.vy, piv_mean.vy)


if __name__ == "__main__":
    unittest.main()
