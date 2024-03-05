import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from fluidimage import get_path_image_samples
from fluidimage._opencv import error_import_cv2
from fluidimage.calibration.calib_cv import (
    CalibCV,
    SimpleCircleGrid,
    construct_object_points,
)
from fluidimage.util import imread

pathbase = get_path_image_samples() / "TomoPIV" / "calibration"
path_cache = pathbase / "cam0.h5"


def construct_image_points(path, origin):
    image = imread(str(path))
    # image = image.astype(np.uint8)
    # Rescale
    scale = image.max()
    image = image * (256 / scale)

    params = SimpleCircleGrid.create_default_params()
    circle_grid = SimpleCircleGrid(params)
    centers = circle_grid.detect_grid(
        image, origin, nx=7, ny=7, ds=50, debug=True
    )

    return centers


def test_interpolate():
    """Test interpolation."""

    if error_import_cv2:
        with pytest.raises(ModuleNotFoundError):
            CalibCV(str(path_cache))
        return

    calib_cache = CalibCV(str(path_cache))
    calib_cache.get_rotation(2)
    calib_cache.get_translate(-2)


def test_calibrate():
    """Tests construct_object_points and CalibCV methods."""

    if error_import_cv2:
        with pytest.raises(ModuleNotFoundError):
            CalibCV(str(path_cache))
        return

    calib_cache = CalibCV(str(path_cache))

    path_input = pathbase / "cam0" / "0mm_cam0.tif"
    calib = CalibCV("fluidimage_test_calib_cv.h5")

    result_cache = calib_cache.params

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
