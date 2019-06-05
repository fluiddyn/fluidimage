import unittest

from numpy.testing import assert_almost_equal, assert_array_equal

from fluiddyn.io import stdout_redirected
from fluidimage import path_image_samples
from fluidimage.calibration.calib_cv import (
    CalibCV,
    SimpleCircleGrid,
    construct_object_points,
)
from fluidimage.util import imread

pathbase = path_image_samples / "TomoPIV" / "calibration"


def construct_image_points(path, origin):
    image = imread(str(path))
    # image = image.astype(np.uint8)
    # Rescale
    scale = image.max()
    image = image * (256 / scale)

    params = SimpleCircleGrid.create_default_params()
    circle_grid = SimpleCircleGrid(params)
    centers = circle_grid.detect_grid(image, origin, nx=7, ny=7, ds=50)

    return centers


class TestCalib(unittest.TestCase):
    """Test fluidimage.calibration.calib_cv module."""

    @classmethod
    def setUpClass(cls):
        path_cache = pathbase / "cam0.h5"
        # with stdout_redirected():
        cls.calib_cache = CalibCV(str(path_cache))

    def test_interpolate(self):
        """Test interpolation."""
        self.calib_cache.get_rotation(2)
        self.calib_cache.get_translate(-2)

    def test_calibrate(self):
        """Tests construct_object_points and CalibCV methods."""
        path_input = pathbase / "cam0" / "0mm_cam0.tif"
        with stdout_redirected():
            calib = CalibCV("fluidimage_test_calib_cv.h5")

        result_cache = self.calib_cache.params

        origin = (250, 250)  # Hardcoded origin
        imgpoints = [construct_image_points(path_input, origin)]
        objpoints = [construct_object_points(7, 7, 0.0, 3.0)]
        zs = [0]
        im_shape = imread(str(path_input)).shape[::-1]
        ret, mtx, dist, rvecs, tvecs = calib.calibrate(
            imgpoints, objpoints, zs, im_shape, origin, debug=True
        )

        assert_array_equal(mtx, result_cache.cam_mtx)
        assert_almost_equal(rvecs[0], result_cache.rotation[2])
        assert_almost_equal(tvecs[0], result_cache.translate[2])


if __name__ == "__main__":
    unittest.main()
