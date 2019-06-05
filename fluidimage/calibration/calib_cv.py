"""2D/3D Calibration using OpenCV (:mod:`fluidimage.calibration.calib_cv`)
==========================================================================

.. autoclass:: ParamContainerCV
   :members:
   :private-members:

.. autoclass:: SimpleCircleGrid
   :members:
   :private-members:

.. autoclass:: CalibCV
   :members:
   :private-members:

"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from scipy.interpolate import griddata

from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage._opencv import cv2


class ParamContainerCV(ParamContainer):
    """A container to easily display OpenCV parameters"""

    def __init__(self, params_cv, tag="OpenCV"):
        super().__init__(tag)
        self._set_internal_attr("_params_cv", params_cv)

        attrs = {
            a: getattr(params_cv, a)
            for a in dir(params_cv)
            if not callable(a) and not a.startswith("__")
        }
        for key, value in attrs.items():
            self._set_attrib(key, value)

    def _as_params_cv(self):
        params_cv = self["_params_cv"]
        for key in self._get_key_attribs():
            value = self[key]
            setattr(params_cv, key, value)
        return params_cv


class SimpleCircleGrid:
    """Detect the centers of calibration target consisting of a grid of circle
    shaped points. Use it to detect the image points.

    """

    @staticmethod
    def create_default_params():
        params_cv = cv2.SimpleBlobDetector_Params()
        # Slightly nicer defaults
        params_cv.filterByColor = False
        params_cv.minArea = 0.0
        return ParamContainerCV(params_cv, "SimpleBlobDetector")

    def __init__(self, params: ParamContainerCV) -> None:
        self.params = params
        params_cv = params._as_params_cv()
        self.detector = cv2.SimpleBlobDetector_create(params_cv)

    def detect_all(self, image: np.ndarray, debug=False):
        """Detects all blobs as per parameters without any constraints.

        Parameters
        ----------
        image: array
            A calibration image of maximum intensity 255.
        debug : bool
            Plot the detected points and the bounding box

        """
        image = image.astype(np.uint8)
        keypoints = self.detector.detect(image)
        if debug:
            plt.figure()
            for k in keypoints:
                plt.scatter(*k.pt)
            plt.show()

        if len(keypoints) == 0:
            raise ValueError("No blob detected")
        return keypoints

    def detect_grid(
        self,
        image: np.ndarray,
        origin: tuple,
        nx: int,
        ny: int,
        ds: float,
        debug=False,
    ):
        """Detect a ``nx`` by ``ny`` circle grid centered around an origin.

        Parameters
        ----------
        image: array
            A calibration image of maximum intensity 255.
        origin: tuple
            Origin / Principal point location in pixel coordinates
        nx, ny : int
            Shape of the grid, i.e. number of points
        ds : float
            Grid spacing in pixel coordinates
        debug : bool
            Plot the detected points and the bounding box

        """
        keypoints = self.detect_all(image)
        # Add 1 so that the points at the edge of the bounding box are included
        w = (nx + 1) * ds
        h = (ny + 1) * ds
        originx, originy = origin
        bbox = Bbox.from_bounds(originx - w // 2, originy - h // 2, w, h)
        if debug:
            print(bbox)

        center_list = []
        for k in keypoints:
            if bbox.contains(*k.pt):
                center_list.append(k.pt)

        centers = np.array(center_list)
        xround = np.round(centers[..., 0] / ds)
        yround = np.round(centers[..., 1] / ds)
        # Sort as a grid
        ind = np.lexsort((xround, yround))
        # Add another dimension
        centers = np.array([centers[ind]], dtype=np.float32)

        if len(centers[0]) != nx * ny:
            raise AssertionError(f"Only {len(centers[0])} points were found")

        if debug:
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            ax.scatter(centers[..., 0], centers[..., 1])
            l, b, w, h = bbox.bounds
            ax.add_patch(
                Rectangle(xy=(l, b), width=w, height=h, edgecolor="r", fill=False)
            )

        return centers


def construct_object_points(nx: int, ny: int, z: float, ds: float):
    """Prepare object points in world coordinates, as flattened list of
    coordinates such as::

        (0,0,z), (1,0,z), (2,0,z) ....,(6,5,z)

    This format is expected by OpenCV's ``calibrateCamera`` function.

    Parameters
    ----------
    nx, ny : int
        Shape of the grid, i.e. number of points
    z : float
        z-location in world coordinates
    ds : float
        Grid spacing in world coordinates

    """
    objp = np.zeros((nx * ny, 3), np.float32)
    assert nx % 2 == 1
    assert ny % 2 == 1
    stepx, stepy = np.array((nx, ny), dtype=int) * 1j
    # nx, ny will refer the number of points on one side from now on
    nx = (nx - 1) // 2
    ny = (ny - 1) // 2
    objp[:, :2] = np.mgrid[-ny:ny:stepy, -nx:nx:stepx].T.reshape(-1, 2) * ds
    objp[:, 2] = z
    return objp


class CalibCV:
    """Calibrate a camera and save them as HDF5 files. Also use this to load
    saved calibrations and interpolate extrinsic parameters (rotation and
    translation) while reconstructing.

    """

    def __init__(self, path_file="cam.h5"):
        self.path_file = str(path_file)
        if os.path.exists(path_file):
            print(f"Loading {path_file}.")
            self.params = ParamContainer(path_file=self.path_file)

    def save(self, zs, ret, mtx, dist, rvecs, tvecs):
        self.params = params = ParamContainer(tag="CalibCV")
        params._set_attribs(
            {
                "class": self.__class__.__name__,
                "module": self.__module__,
                "f": np.diag(mtx)[:2],
                "C": mtx[0:2, 2],
                "cam_mtx": mtx,
                "kc": dist.T[0],
                "rotation": np.array(rvecs),
                "translate": tvecs,
                "zs": np.array(zs),
            }
        )
        path_dir = os.path.dirname(self.path_file)
        os.makedirs(path_dir, exist_ok=True)
        if os.path.exists(self.path_file):
            print(f"WARNING: {self.path_file} already exists. Skipping save.")
        else:
            params._save_as_hdf5(self.path_file)

    def rotmtx_from_rotvec(self, rot_vec):
        rot_mtx, rot_jac = cv2.Rodrigues(rot_vec)
        return rot_mtx

    def get_rotation(self, znew):
        """Linearly interpolate the rotation vector based on z location."""
        rot_vec = griddata(self.params.zs, self.params.rotation, znew)
        return rot_vec

    def get_translate(self, znew):
        """Linearly interpolate the translation vector based on z location."""
        translate = griddata(self.params.zs, self.params.translate, znew)
        return translate

    def calibrate(
        self,
        imgpoints: list,
        objpoints: list,
        zs: list,
        im_shape: tuple,
        origin=None,
        debug=False,
        flags=None,
    ):
        """Calibrate a camera based on a list of image points (in pixel
        coordinates) and object points (in world coordinates) and the z-locations
        (in world coordinates).
 
        Parameters
        ----------
        imgpoints : list of arrays
            Image points in pixel coordinates. Use `SimpleCircleGrid` to detect
            them from a single calibration image. Append such image points
            from multiple calibration images in a list.
        objpoints : list of arrays
            Object points as produced by function `construct_object_points`
            constitute a single array. Likewise append them into a list for
            multiple calibration images.
        zs : list of float
            List of z locations of the calibration targes in world coordinates.
        im_shape : tuple of int
            Image dimensions in pixels.
        origin : tuple, optional
            Origin / Principal point location in pixel coordinates
        flags : int, optional
            OpenCV specific calibration flags
        debug : bool, optional
            Return the result if true, else save it as an XML file.

        """
        # Initial guesses
        initial_mtx = np.eye(3)
        if origin is not None:
            # Origin / Principal point at center:
            initial_mtx[0:2, 2] = origin
        initial_dist = np.zeros((5, 1))

        if flags is None:
            flags = (
                cv2.CALIB_USE_INTRINSIC_GUESS
                + cv2.CALIB_FIX_K4
                + cv2.CALIB_FIX_K5
            )

        result = cv2.calibrateCamera(
            objpoints, imgpoints, im_shape, initial_mtx, initial_dist, flags=flags
        )

        if debug:
            return result
        else:
            self.save(zs, *result)
