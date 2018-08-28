
import unittest

from fluidimage import path_image_samples, np

from fluidimage.postproc.piv import get_grid_pixel_from_piv_file, PIV2d, ArrayPIV


class TestPIV(unittest.TestCase):
    def test_get_grid(self):
        path_piv = path_image_samples / "Milestone" / "piv_0000a-b.h5"
        xs1d, ys1d = get_grid_pixel_from_piv_file(path_piv)

    def test_piv_objects(self):

        ny, nx = 8, 12

        xs1d = np.linspace(-1, 1, nx)
        ys1d = np.linspace(-1, 1, ny)
        xs2d, ys2d = np.meshgrid(xs1d, ys1d)
        piv0 = PIV2d(xs1d, ys1d, 0, xs2d, ys2d)
        piv1 = (2 * piv0 + 1) / 3 + piv0
        arr = ArrayPIV((piv0,))
        arr.append(piv1)
        arr.extend((piv0 - 1, piv1 - piv0, 0 + piv0 * 3 + 1))

        arr1 = 0 + 2 * arr * 3 + 1 - 2

        arr1 = (1 + arr1).median_filter(3).gaussian_filter(0.5)

        piv2 = arr1[0]
        arr1[1] = piv2
        repr(arr1)
        len(arr1)

        arr1 = arr1.truncate().extract(0, 4, 2, 7).extract_square()


if __name__ == "__main__":
    unittest.main()
