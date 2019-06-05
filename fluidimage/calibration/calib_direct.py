"""Direct calibration (:mod:`fluidimage.calibration.calib_direct`)
==================================================================

.. autoclass:: CalibDirect
   :members:
   :private-members:

.. autoclass:: DirectStereoReconstruction
   :members:
   :private-members:

"""
import glob
import warnings
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pylab
from scipy.interpolate import (
    LinearNDInterpolator,
    RegularGridInterpolator,
    griddata,
)

from fluiddyn.util.paramcontainer import ParamContainer, tidy_container

from .util import get_base_from_normal_vector, get_number_from_string2

# from scipy.interpolate import CloughTocher2DInterpolator


try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass


class Interpolent:
    pass


def get_points(points_file):
    """Get grid points extracted from calibration images
    """
    imgpts = ParamContainer(path_file=points_file)
    tidy_container(imgpts)
    imgpts = imgpts.geometry_calib.source_calib.__dict__
    points = [x for x in imgpts.keys() if "point_" in x or "Point" in x]
    points.sort()

    # [X, Y, Z, x, y]
    results = [[], [], [], [], []]
    # X, Y and Z are coordinate in real space (physical unit)
    # x and y are the indices of the pixel in the image
    for point in points:
        numbers = get_number_from_string2(imgpts[point])

        for i, result in enumerate(results):
            result.append(numbers[i])

    results = [np.array(result) for result in results]

    # particularity for Z
    results[2] = results[2][0]

    # particularity for X, Y, Z
    for i in range(3):
        results[i] /= 100.0

    return results


class CalibDirect:
    """Class for direct Calibration
    This calibration determine the equations of optical paths for "each" pixels

    Parameters
    ----------

    glob_str_xml: None
      Path for grid points extracted from calibration images
      This files are given by UVMAT.
      example: 'Images/img*.xml'

    nb_pixels: (None, None)
      Number of pixels in the images

    pth_file: None
      Path of calibration file

    """

    def __init__(self, glob_str_xml=None, shape_img=(None, None), path_file=None):

        if path_file:
            self.load(path_file)
        else:
            self.paths_xml = glob.glob(str(glob_str_xml))
            if len(self.paths_xml) == 0:
                raise ValueError(
                    "No xml file found. \n"
                    'glob_str_xml = "{}"'.format(glob_str_xml)
                )

            self.nb_pixels_x = shape_img[1]
            self.nb_pixels_y = shape_img[0]

    def compute_interpolents(self, interpolator=LinearNDInterpolator):
        """Compute interpolents

        Create an object Interpolent (self.interp_levels) containing
        interpolents to switch from indices of pixels to physical coordinates.

        - indices_pixel2xphys
        - indices_pixel2yphys
        - phys2index_x_pixel
        - phys2index_y_pixel
        - zphys

        There are as many interpolents of each sort as numbers of planes (z
        values).

        """
        interp = Interpolent()
        interp.indices_pixel2xphys = []
        interp.indices_pixel2yphys = []
        interp.phys2index_x_pixel = []
        interp.phys2index_y_pixel = []
        interp.Z = []

        for i, path in enumerate(self.paths_xml):
            # naming convention:
            # X, Y and Z are coordinate in real space (physical unit)
            # x and y are the indices of the pixel in the image
            X, Y, Z, x, y = get_points(path)
            interp.Z.append(Z)

            interp.indices_pixel2xphys.append(interpolator((x, y), X))
            interp.indices_pixel2yphys.append(interpolator((x, y), Y))
            interp.phys2index_x_pixel.append(interpolator((X, Y), x))
            interp.phys2index_y_pixel.append(interpolator((X, Y), y))

        self.interp_levels = interp

    def compute_interpolents_pixel2line(self, nb_lines_x, nb_lines_y, test=False):
        """Compute interpolents for parameters for each optical path.

        The number of optical paths is given by nb_lines_x * nb_lines_y.

        Optical paths are defined with a point x0, y0, z0 and a vector dx, dy,
        dz.

        """

        xtmp = np.unique(np.floor(np.linspace(0, self.nb_pixels_x, nb_lines_x)))
        ytmp = np.unique(np.floor(np.linspace(0, self.nb_pixels_y, nb_lines_y)))

        x, y = np.meshgrid(xtmp, ytmp)
        x = x.transpose()
        y = y.transpose()
        V = np.zeros((x.shape[0], x.shape[1], 6))

        xtrue = []
        ytrue = []
        vtrue = []
        xfalse = []
        yfalse = []
        indi = []
        indj = []

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tmp = self.pixel2line(x[i, j], y[i, j])
                if np.isnan(tmp[0]):
                    xfalse.append(x[i, j])
                    yfalse.append(y[i, j])
                    indi.append(i)
                    indj.append(j)
                else:
                    V[i, j, :] = tmp
                    xtrue.append(x[i, j])
                    ytrue.append(y[i, j])
                    vtrue.append(tmp)

        if test:
            titles = ["X0", "Y0", "Z0", "dx", "dy", "dz"]
            for j in range(6):
                pylab.figure()
                pylab.pcolor(x, y, V[:, :, j])
                pylab.title(titles[j])
                pylab.xlabel("x (pix)")
                pylab.ylabel("y (pix)")
                pylab.colorbar()

            pylab.figure()
            pylab.pcolor(
                x, y, np.sqrt(V[:, :, 3] ** 2 + V[:, :, 4] ** 2 + V[:, :, 5] ** 2)
            )
            pylab.title("norm(d)")
            pylab.xlabel("x (pix)")
            pylab.ylabel("y (pix)")
            pylab.colorbar()
            plt.show()

        vtrue = np.array(vtrue)
        for j in range(6):
            V[indi, indj, j] = griddata(
                (xtrue, ytrue), vtrue[:, j], (xfalse, yfalse)
            )

        interp = []
        for i in range(6):
            interp.append(RegularGridInterpolator((xtmp, ytmp), V[:, :, i]))

        self.interp_lines = interp

    def pixel2line(self, indx, indy):
        """Compute parameters of the optical path for a pixel

        An optical path is defined with a point x0, y0, z0 and a vector dx, dy,
        dz.

        """
        interp = self.interp_levels
        X = []
        Y = []
        Z = interp.Z
        for i in range(len(Z)):
            X.append((interp.indices_pixel2xphys[i]((indx, indy))))
            Y.append((interp.indices_pixel2yphys[i]((indx, indy))))
        X = np.array(X)
        Y = np.array(Y)
        XYZ = np.vstack([X, Y, Z]).transpose()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            XYZ0 = np.nanmean(XYZ, 0)
        XYZ -= XYZ0
        ind = ~np.isnan(X + Y)
        XYZ = XYZ[ind, :]
        if XYZ.shape[0] > 1:
            arg = np.argsort(XYZ[:, 2])
            XYZ = XYZ[arg, :]
            u, s, v = np.linalg.svd(XYZ, full_matrices=True, compute_uv=1)
            direction = np.cross(v[-1, :], v[-2, :])
            if direction[2] < 0:
                direction = -direction
            return np.hstack([XYZ0, direction])

        else:
            return np.hstack([np.nan] * 6)

    def save(self, pth_file):
        """Save calibration
        """
        np.save(
            pth_file,
            [
                self.interp_lines,
                self.paths_xml,
                self.nb_pixels_x,
                self.nb_pixels_y,
            ],
        )

    def load(self, pth_file):
        """Load calibration
        """
        tmp = np.load(pth_file, allow_pickle=True)
        self.interp_lines = tmp[0]
        self.paths_xml = tmp[1]
        self.nb_pixels_x = tmp[2]
        self.nb_pixels_y = tmp[3]

    def intersect_with_plane(self, indx, indy, a, b, c, d):
        """Find intersection with the line associated to the pixel  indx, indy
        and a plane defined by ax + by + cz + d = 0
        """
        x0 = self.interp_lines[0]((indx, indy))
        y0 = self.interp_lines[1]((indx, indy))
        z0 = self.interp_lines[2]((indx, indy))
        dx = self.interp_lines[3]((indx, indy))
        dy = self.interp_lines[4]((indx, indy))
        dz = self.interp_lines[5]((indx, indy))
        # fmt: off
        t = -(a*x0 + b*y0 + c*z0 + d) / (a*dx + b*dy + c*dz)
        # fmt: on
        physical_coords = np.array([x0 + t * dx, y0 + t * dy, z0 + t * dz])
        return physical_coords.transpose()

    def apply_calib(self, indx, indy, dx, dy, a, b, c, d):
        """Gives the projection of the real displacement projected on each
        camera plane and then projected on the laser plane
        """
        displacements = self.intersect_with_plane(
            indx + dx / 2.0, indy + dy / 2.0, a, b, c, d
        ) - self.intersect_with_plane(
            indx - dx / 2.0, indy - dy / 2.0, a, b, c, d
        )
        return displacements

    def get_base_camera_plane(self, indx=None, indy=None):
        """Matrix of base change from camera plane to fixed plane
        """
        if indx is None:
            indx = range(self.nb_pixels_x // 2 - 20, self.nb_pixels_x // 2 + 20)
            indy = range(self.nb_pixels_y // 2 - 20, self.nb_pixels_y // 2 + 20)
            indx, indy = np.meshgrid(indx, indy)
        dx = np.nanmean(self.interp_lines[3]((indx, indy)))
        dy = np.nanmean(self.interp_lines[4]((indx, indy)))
        dz = np.nanmean(self.interp_lines[5]((indx, indy)))
        A, B = get_base_from_normal_vector(dx, dy, dz)
        return A, B

    def check_interp_levels(self, number=100):
        """Plot to check interp_levels
        """
        interp = self.interp_levels
        indx = range(0, self.nb_pixels_x, self.nb_pixels_x // number)
        indy = range(0, self.nb_pixels_y, self.nb_pixels_y // number)
        indx, indy = np.meshgrid(indx, indy)
        Z = interp.Z
        for i in range(len(Z)):
            X = interp.indices_pixel2xphys[i]((indx, indy))
            Y = interp.indices_pixel2yphys[i]((indx, indy))
            pylab.figure()
            pylab.pcolor(indx, indy, X)
            pylab.title(f"Level {i}, X")
            pylab.xlabel("x (pix)")
            pylab.ylabel("y (pix)")
            pylab.colorbar()

            pylab.figure()
            pylab.pcolor(indx, indy, Y)
            pylab.title(f"Level {i}, Y")
            pylab.xlabel("x (pix)")
            pylab.ylabel("y (pix)")
            pylab.colorbar()

        plt.show()

    def check_interp_lines(self, number=10):
        """Plot to check interp_lines
        """
        fig = pylab.figure()
        ax = Axes3D(fig)
        for i, path in enumerate(self.paths_xml):
            X, Y, Z, x, y = get_points(path)
            ax.scatter(X, Y, Z, marker=".", color="blue")

        x = range(0, self.nb_pixels_x, self.nb_pixels_x // number)
        y = range(0, self.nb_pixels_y, self.nb_pixels_y // number)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        for i in range(len(x)):
            X0 = self.interp_lines[0]((x[i], y[i]))
            Y0 = self.interp_lines[1]((x[i], y[i]))
            Z0 = self.interp_lines[2]((x[i], y[i]))
            dx = self.interp_lines[3]((x[i], y[i]))
            dy = self.interp_lines[4]((x[i], y[i]))
            dz = self.interp_lines[5]((x[i], y[i]))
            X = (np.arange(10) - 5) / 20.0 * dx + X0
            Y = (np.arange(10) - 5) / 20.0 * dy + Y0
            Z = (np.arange(10) - 5) / 20.0 * dz + Z0
            ax.plot(X, Y, Z, "r")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        plt.show()

    def check_interp_lines_coeffs(self, number=100):
        """Plot to check interp_lines coefficients
        """
        x = range(0, self.nb_pixels_x, self.nb_pixels_x // number)
        y = range(0, self.nb_pixels_y, self.nb_pixels_y // number)
        x, y = np.meshgrid(x, y)

        X0 = np.zeros(x.shape)
        Y0 = np.zeros(x.shape)
        Z0 = np.zeros(x.shape)
        dx = np.zeros(x.shape)
        dy = np.zeros(x.shape)
        dz = np.zeros(x.shape)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                X0[i, j] = self.interp_lines[0]((x[i, j], y[i, j]))
                Y0[i, j] = self.interp_lines[1]((x[i, j], y[i, j]))
                Z0[i, j] = self.interp_lines[2]((x[i, j], y[i, j]))
                dx[i, j] = self.interp_lines[3]((x[i, j], y[i, j]))
                dy[i, j] = self.interp_lines[4]((x[i, j], y[i, j]))
                dz[i, j] = self.interp_lines[5]((x[i, j], y[i, j]))

        pylab.figure()
        pylab.pcolor(x, y, X0)
        pylab.title("X0")
        pylab.xlabel("x (pix)")
        pylab.ylabel("y (pix)")
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, Y0)
        pylab.title("Y0")
        pylab.xlabel("x (pix)")
        pylab.ylabel("y (pix)")
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, Z0)
        pylab.title("Z0")
        pylab.xlabel("x (pix)")
        pylab.ylabel("y (pix)")
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, dx)
        pylab.title("dx")
        pylab.xlabel("x (pix)")
        pylab.ylabel("y (pix)")
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, dy)
        pylab.title("dy")
        pylab.xlabel("x (pix)")
        pylab.ylabel("y (pix)")
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, dz)
        pylab.title("dz")
        pylab.xlabel("x (pix)")
        pylab.ylabel("y (pix)")
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        pylab.title("norm(d)")
        pylab.xlabel("x (pix)")
        pylab.ylabel("y (pix)")
        pylab.colorbar()

        plt.show()


class DirectStereoReconstruction:
    """Class to get stereo reconstruction with direct Calibration
    This calibration determine the equations of optical paths for "each" pixels

    Parameters
    ----------

    path_file0:
       Path of the file of the first camera

    path_file1:
       Path of the file of the second camera
    """

    def __init__(self, path_file0, path_file1):
        self.calib0 = CalibDirect(path_file=path_file0)
        self.calib1 = CalibDirect(path_file=path_file1)
        # matrices from camera planes to fixed plane and inverse
        self.A0, self.B0 = self.calib0.get_base_camera_plane()
        self.A1, self.B1 = self.calib1.get_base_camera_plane()

        if np.allclose(self.A0, self.A1):
            raise ValueError("The two calibrations have to be different.")

        # M1, M2: see reconstruction function
        self.invM0 = np.linalg.inv(np.vstack([self.B0[0:2, :], self.B1[0:1, :]]))
        self.invM1 = np.linalg.inv(np.vstack([self.B0[0:1, :], self.B1[0:2, :]]))
        self.invM2 = np.linalg.inv(np.vstack([self.B0[0:2, :], self.B1[1:2, :]]))
        self.invM3 = np.linalg.inv(np.vstack([self.B0[1:2, :], self.B1[0:2, :]]))

    def project2cam(
        self,
        indx0,
        indy0,
        dx0,
        dy0,
        indx1,
        indy1,
        dx1,
        dy1,
        a,
        b,
        c,
        d,
        check=False,
    ):
        """Project displacements of each cameras dx0, dy0, dx1 and dy1
        in their respective planes.
        """

        X0 = self.calib0.intersect_with_plane(indx0, indy0, a, b, c, d)
        dX0 = self.calib0.apply_calib(indx0, indy0, dx0, dy0, a, b, c, d)

        X1 = self.calib1.intersect_with_plane(indx1, indy1, a, b, c, d)
        dX1 = self.calib1.apply_calib(indx1, indy1, dx1, dy1, a, b, c, d)

        d0cam = np.tensordot(self.B0, dX0.swapaxes(0, 1), axes=1)[
            :2, :
        ].transpose()
        d1cam = np.tensordot(self.B1, dX1.swapaxes(0, 1), axes=1)[
            :2, :
        ].transpose()

        if check:
            plt.figure()
            plt.quiver(indx0, indy0, dx0, dy0)
            plt.axis("equal")
            pylab.xlabel("x (pix)")
            pylab.ylabel("y (pix)")
            plt.title("Cam 0 in pixel")
            plt.figure()
            plt.quiver(X0[:, 0], X0[:, 1], dX0[:, 0], dX0[:, 1])
            pylab.xlabel("x (pix)")
            pylab.ylabel("y (pix)")
            plt.axis("equal")
            plt.title("Cam 0 in m")
            plt.figure()
            plt.quiver(indx1, indy1, dx1, dy1)
            pylab.xlabel("x (pix)")
            pylab.ylabel("y (pix)")
            plt.axis("equal")
            plt.title("Cam 1 in pixel")
            plt.figure()
            plt.quiver(X1[:, 0], X1[:, 1], dX1[:, 0], dX1[:, 1])
            pylab.xlabel("x (pix)")
            pylab.ylabel("y (pix)")
            plt.axis("equal")
            plt.title("Cam 1 in m")
            plt.show()

        return X0, X1, d0cam, d1cam

    def find_common_grid(self, X0, X1, a, b, c, d):
        """Find a common grid for the 2 cameras
        """
        xmin0 = np.nanmin(X0[:, 0])
        xmax0 = np.nanmax(X0[:, 0])
        ymin0 = np.nanmin(X0[:, 1])
        ymax0 = np.nanmax(X0[:, 1])
        xmin1 = np.nanmin(X1[:, 0])
        xmax1 = np.nanmax(X1[:, 0])
        ymin1 = np.nanmin(X1[:, 1])
        ymax1 = np.nanmax(X1[:, 1])

        xmin = max([xmin0, xmin1])
        xmax = min([xmax0, xmax1])
        ymin = max([ymin0, ymin1])
        ymax = min([ymax1, ymax1])

        Lx0 = xmax0 - xmin0
        Ly0 = ymax0 - ymin0
        Lx1 = xmax1 - xmin1
        Ly1 = ymax1 - ymin1

        Nx0 = sqrt(X0[~np.isnan(X0[:, 0]), 0].size) * sqrt(Lx0 / Ly0)
        Ny0 = X0[~np.isnan(X0[:, 1]), 1].size / Nx0
        Nx1 = sqrt(X1[~np.isnan(X1[:, 0]), 0].size) * sqrt(Lx1 / Ly1)
        Ny1 = X1[~np.isnan(X1[:, 1]), 1].size / Nx1

        dx0 = Lx0 / Nx0
        dy0 = Ly0 / Ny0
        dx1 = Lx1 / Nx1
        dy1 = Ly1 / Ny1

        dx = max([dx0, dx1])
        dy = max([dy0, dy1])

        x = np.linspace(xmin, xmax, int((xmax - xmin) / dx))
        y = np.linspace(ymin, ymax, int((ymax - ymin) / dy))
        x, y = np.meshgrid(x, y)
        self.grid_x = x.transpose()
        self.grid_y = y.transpose()
        self.grid_z = -(a * self.grid_x + b * self.grid_y + d) / c
        return self.grid_x, self.grid_y, self.grid_z

    def interp_on_common_grid(self, X0, X1, d0cam, d1cam, grid_x, grid_y):
        """Interpolate displacements of the 2 cameras d0cam, d1cam on the
        common grid grid_x, grid_y
        """
        # if not hasattr(self, 'grid_x'):
        #     self.find_common_grid(X0, X1, a, b, c, d)
        ind0 = (
            (~np.isnan(X0[:, 0]))
            * (~np.isnan(X0[:, 1]))
            * (~np.isnan(d0cam[:, 0]))
            * (~np.isnan(d0cam[:, 1]))
        )
        ind1 = (
            (~np.isnan(X1[:, 0]))
            * (~np.isnan(X1[:, 1]))
            * (~np.isnan(d1cam[:, 0]))
            * (~np.isnan(d1cam[:, 1]))
        )

        d0xcam = griddata(
            (X0[ind0, 0], X0[ind0, 1]), d0cam[ind0, 0], (grid_x, grid_y)
        )
        d0ycam = griddata(
            (X0[ind0, 0], X0[ind0, 1]), d0cam[ind0, 1], (grid_x, grid_y)
        )
        d1xcam = griddata(
            (X1[ind1, 0], X1[ind1, 1]), d1cam[ind1, 0], (grid_x, grid_y)
        )
        d1ycam = griddata(
            (X1[ind1, 0], X1[ind1, 1]), d1cam[ind1, 1], (grid_x, grid_y)
        )
        return d0xcam, d0ycam, d1xcam, d1ycam

    def reconstruction(
        self, X0, X1, d0cam, d1cam, a, b, c, d, grid_x, grid_y, check=False
    ):
        """Reconstruction of the 3 components of the velocity in the plane
        defined by a, b, c, d on the grid defined by grid_x, grid_y
        from the displacements of the 2 cameras d0cam, d1cam in their respective
        planes d0cam, d1cam
        """
        # reconstruction => resolution of the equation
        # MX = dcam

        d0xcam, d0ycam, d1xcam, d1ycam = self.interp_on_common_grid(
            X0, X1, d0cam, d1cam, grid_x, grid_y
        )

        # in the case where 2 vectors from different cameras are approximately
        # the same, they don't have to be used both in the computation
        n1 = np.abs(np.inner(self.A0[0], self.A1[0]))
        n2 = np.abs(np.inner(self.A0[1], self.A1[1]))
        n3 = np.abs(np.inner(self.A0[1], self.A1[0]))
        n4 = np.abs(np.inner(self.A0[0], self.A1[1]))
        # I suppose that 5deg between vectors is sufficient
        threshold = np.cos(5 * 2 * np.pi / 180.0)

        tmp = []
        dcam = np.zeros((3, d0xcam.shape[0], d0xcam.shape[1]))
        if n1 < threshold and n3 < threshold:
            dcam[0, :, :] = d0xcam
            dcam[1, :, :] = d0ycam
            dcam[2, :, :] = d1xcam
            tmp.append(np.tensordot(self.invM0, dcam, axes=([1], [0])))
        if n1 < threshold and n4 < threshold:
            dcam[0, :, :] = d0xcam
            dcam[1, :, :] = d1xcam
            dcam[2, :, :] = d1ycam
            tmp.append(np.tensordot(self.invM1, dcam, axes=([1], [0])))
        if n2 < threshold and n4 < threshold:
            dcam[0, :, :] = d0xcam
            dcam[1, :, :] = d0ycam
            dcam[2, :, :] = d1ycam
            tmp.append(np.tensordot(self.invM2, dcam, axes=([1], [0])))
        if n2 < threshold and n3 < threshold:
            dcam[0, :, :] = d0ycam
            dcam[1, :, :] = d1xcam
            dcam[2, :, :] = d1ycam
            tmp.append(np.tensordot(self.invM3, dcam, axes=([1], [0])))
        tmp = np.array(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tmp_mean = np.nanmean(tmp, 0)
        tmp_std = np.nanstd(tmp, 0)
        dX, dY, dZ = tmp_mean
        Errorx, Errory, Errorz = tmp_std

        # dXx = np.zeros(d0xcam.shape)
        # dXy = np.zeros(d0xcam.shape)
        # dXz = np.zeros(d0xcam.shape)
        # Errorx = np.zeros(d0xcam.shape)
        # Errory = np.zeros(d0xcam.shape)
        # Errorz = np.zeros(d0xcam.shape)
        # for i in range(d0xcam.shape[0]):
        #     for j in range(d0xcam.shape[1]):
        #         tmp = []
        #         if n1 < threshold and n3 < threshold:
        #             dcam = np.hstack(
        #                 [d0xcam[i, j], d0ycam[i, j], d1xcam[i, j]])
        #             tmp.append(np.dot(self.invM0, dcam))
        #         if n1 < threshold and n4 < threshold:
        #             dcam = np.hstack(
        #                 [d0xcam[i, j], d1xcam[i, j], d1ycam[i, j]])
        #             tmp.append(np.dot(self.invM1, dcam))
        #         if n2 < threshold and n4 < threshold:
        #             dcam = np.hstack(
        #                 [d0xcam[i, j], d0ycam[i, j], d1ycam[i, j]])
        #             tmp.append(np.dot(self.invM2, dcam))
        #         if n2 < threshold and n3 < threshold:
        #             dcam = np.hstack(
        #                 [d0ycam[i, j], d1xcam[i, j], d1ycam[i, j]])
        #             tmp.append(np.dot(self.invM3, dcam))
        #         tmp = np.array(tmp)

        # dXx[i, j] = np.nanmean(tmp, 0)[0]
        # Errorx[i, j] = np.nanstd(tmp, 0)[0]
        # dXy[i, j] = np.nanmean(tmp, 0)[1]
        # Errory[i, j] = np.nanstd(tmp, 0)[1]
        # dXz[i, j] = np.nanmean(tmp, 0)[2]
        # Errorz[i, j] = np.nanstd(tmp, 0)[2]

        if check:
            plt.figure()
            plt.quiver(self.grid_x, self.grid_y, d0xcam, d0ycam)
            pylab.xlabel("X")
            pylab.ylabel("Y")
            plt.axis("equal")
            plt.title("Cam 0 plane projection")
            plt.figure()
            plt.quiver(self.grid_x, self.grid_y, d1xcam, d1ycam)
            pylab.xlabel("X")
            pylab.ylabel("Y")
            plt.axis("equal")
            plt.title("Cam 1 plane projection")
            plt.figure()
            plt.quiver(self.grid_x, self.grid_y, dX, dY)
            pylab.xlabel("X")
            pylab.ylabel("Y")
            plt.axis("equal")
            plt.title("Reconstruction on laser plane")
            plt.show()

        return dX, dY, dZ, Errorx, Errory, Errorz
