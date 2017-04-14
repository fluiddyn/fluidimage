"""Direct calibration (:mod:`fluidimage.calibration.calib_direct`)
==================================================================

.. autoclass:: CalibDirect
   :members:
   :private-members:

.. autoclass:: DirectStereoReconstruction
   :members:
   :private-members:

"""
import numpy as np
import glob
import warnings
from math import sqrt

import pylab
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
# from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import matplotlib.pyplot as plt

from fluiddyn.util.paramcontainer import ParamContainer, tidy_container

from .util import get_number_from_string2, get_base_from_normal_vector


class Interpolent():
    pass


class CalibDirect():
    """Class for direct Calibration
    This calibration determine the equations of optical paths for "each" pixels

    Parameters
    ----------

    pathimg: None
      Path for grid points extracted from calibration images
      This files are given by UVMAT.
      example: 'Images/img*.xml'

    nb_pixel: (None, None)
      Number of pixels in the images

    pth_file: None
      Path of calibration file

    """
    def __init__(self, pathimg=None, nb_pixel=(None, None),
                 pth_file=None):

        if pth_file:
            self.load(pth_file)
        else:
            self.pathimg = glob.glob(pathimg)
            self.nb_pixelx = nb_pixel[0]
            self.nb_pixely = nb_pixel[1]

    def get_points(self, img):
        """ Get grid points extracted from calibration images
        """
        imgpts = ParamContainer(path_file=img)
        tidy_container(imgpts)
        imgpts = imgpts.geometry_calib.source_calib.__dict__
        pts = ([x for x in imgpts.keys() if 'point_' in x or 'Point' in x])

        # coord in real space
        X = np.array(
            [get_number_from_string2(imgpts[tmp])[0] for tmp in pts])/100.
        Y = np.array(
            [get_number_from_string2(imgpts[tmp])[1] for tmp in pts])/100.
        Z = np.array(
            [get_number_from_string2(imgpts[tmp])[2] for tmp in pts])[0]/100.
        # coord in image
        x = np.array(
            [get_number_from_string2(imgpts[tmp])[3] for tmp in pts])
        y = np.array([get_number_from_string2(imgpts[tmp])[4] for tmp in pts])
        # difference of convention with calibration done with uvmat for Y!
        return X, Y, Z, x, y

    def compute_interpolents(self, interpolator=LinearNDInterpolator):
        """ Compute interpolents (self.interp_levels) from camera coordinates to
        real coordinates for each plane z=?
        """
        imgs = self.pathimg
        interp = Interpolent()
        interp.cam2X = []
        interp.cam2Y = []
        interp.real2x = []
        interp.real2y = []
        interp.Z = []

        for i, img in enumerate(imgs):
            X, Y, Z, x, y = self.get_points(img)
            interp.Z.append(Z)

            interp.cam2X.append(interpolator((x, y), X))
            interp.cam2Y.append(interpolator((x, y), Y))
            interp.real2x.append(interpolator((X, Y), x))
            interp.real2y.append(interpolator((X, Y), y))

        self.interp_levels = interp

    def compute_interppixel2line(self, nbline,
                                 test=False):
        """ Compute interpolents for parameters for each optical path
        (number of optical path is given by nbline=(nbline_x, nbline_y)
        optical paths are defined with
        a point x0, y0, z0 and a vector dx, dy, dz
        """
        nbline_x = nbline[0]
        nbline_y = nbline[1]

        xtmp = np.unique(np.floor(np.linspace(0, self.nb_pixelx, nbline_x)))
        ytmp = np.unique(np.floor(np.linspace(0, self.nb_pixely, nbline_y)))

        x, y = np.meshgrid(xtmp, ytmp)
        x = x.transpose()
        y = y.transpose()
        V = np.zeros((x.shape[0], x.shape[1], 6))

        xtrue = []
        ytrue = []
        Vtrue = []
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
                    Vtrue.append(tmp)

        if test:
            titles = ['X0', 'Y0', 'Z0', 'dx', 'dy', 'dz']
            for j in range(6):
                pylab.figure()
                pylab.pcolor(x, y, V[:, :, j])
                pylab.title(titles[j])
                pylab.xlabel('x (pix)')
                pylab.ylabel('y (pix)')
                pylab.colorbar()

            pylab.figure()
            pylab.pcolor(x, y, np.sqrt(
                V[:, :, 3]**2+V[:, :, 4]**2+V[:, :, 5]**2))
            pylab.title('norm(d)')
            pylab.xlabel('x (pix)')
            pylab.ylabel('y (pix)')
            pylab.colorbar()
            plt.show()

        Vtrue = np.array(Vtrue)
        for j in range(6):
            V[indi, indj, j] = griddata(
                (xtrue, ytrue), Vtrue[:, j], (xfalse, yfalse))

        interp = []
        for i in range(6):
            interp.append(RegularGridInterpolator((xtmp, ytmp), V[:, :, i]))

        self.interp_lines = interp

    def pixel2line(self, indx, indy):
        """ Compute parameters of the optical path for a pixel
        optical path is defined with
        a point x0, y0, z0 and a vector dx, dy, dz
        """
        interp = self.interp_levels
        X = []
        Y = []
        Z = interp.Z
        for i in range(len(Z)):
            X.append((interp.cam2X[i]((indx, indy))))
            Y.append((interp.cam2Y[i]((indx, indy))))
        X = np.array(X)
        Y = np.array(Y)
        XYZ = np.vstack([X, Y, Z]).transpose()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            XYZ0 = np.nanmean(XYZ, 0)
        XYZ -= XYZ0
        ind = ~np.isnan(X+Y)
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
            return np.hstack([np.nan]*6)

    def save(self, pth_file):
        """ Save calibration
        """
        np.save(pth_file,
                [self.interp_lines, self.pathimg, self.nb_pixelx,
                 self.nb_pixely])

    def load(self, pth_file):
        """ Load calibration
        """
        tmp = np.load(pth_file)
        self.interp_lines = tmp[0]
        self.pathimg = tmp[1]
        self.nb_pixelx = tmp[2]
        self.nb_pixely = tmp[3]

    def intersect_with_plane(self, indx, indy, a, b, c, d):
        """ Find intersection with the line associated to the pixel  indx, indy
        and a plane defined by ax + by + cz + d =0
        """
        def get_coord(ix, iy):
            x0 = self.interp_lines[0]((ix, iy))
            y0 = self.interp_lines[1]((ix, iy))
            z0 = self.interp_lines[2]((ix, iy))
            dx = self.interp_lines[3]((ix, iy))
            dy = self.interp_lines[4]((ix, iy))
            dz = self.interp_lines[5]((ix, iy))
            t = -(a * x0 + b * y0 + c * z0 + d) / (a * dx + b * dy + c * dz)
            return np.array([x0 + t * dx, y0 + t * dy, z0 + t * dz])

        X = get_coord(indx, indy).transpose()

        return X

    def apply_calib(self, indx, indy, dx, dy, a, b, c, d):
            """ Gives the projection of the real displacement projected on each
            camera plane and then projected on the laser plane
            """
            dX = self.intersect_with_plane(
                indx+dx/2, indy+dy/2, a, b, c, d) - self.intersect_with_plane(
                    indx-dx/2, indy-dy/2, a, b, c, d)
            return dX

    def get_base_camera_plane(self, indx=None, indy=None):
        """ Matrix of base change from camera plane to fixed plane
        """
        if indx is None:
            indx = range(self.nb_pixelx//2-20, self.nb_pixelx//2+20)
            indy = range(self.nb_pixely//2-20, self.nb_pixely//2+20)
            indx, indy = np.meshgrid(indx, indy)
        dx = np.nanmean(self.interp_lines[3]((indx, indy)))
        dy = np.nanmean(self.interp_lines[4]((indx, indy)))
        dz = np.nanmean(self.interp_lines[5]((indx, indy)))
        A = get_base_from_normal_vector(dx, dy, dz)
        return A

    def check_interp_levels(self):
        """ Plot to check interp_levels
        """
        interp = self.interp_levels
        indx = range(0, self.nb_pixelx, self.nb_pixelx//100)
        indy = range(0, self.nb_pixely, self.nb_pixely//100)
        indx, indy = np.meshgrid(indx, indy)
        Z = interp.Z
        for i in range(len(Z)):
            X = interp.cam2X[i]((indx, indy))
            Y = interp.cam2Y[i]((indx, indy))
            pylab.figure()
            pylab.pcolor(indx, indy, X)
            pylab.title('Level {}, X'.format(i))
            pylab.xlabel('x (pix)')
            pylab.ylabel('y (pix)')
            pylab.colorbar()

            pylab.figure()
            pylab.pcolor(indx, indy, Y)
            pylab.title('Level {}, Y'.format(i))
            pylab.xlabel('x (pix)')
            pylab.ylabel('y (pix)')
            pylab.colorbar()

        plt.show()

    def check_interp_lines(self):
        """ Plot to check interp_lines
        """
        imgs = self.pathimg
        interp = Interpolent()
        interp.cam2X = []
        interp.cam2Y = []
        interp.real2x = []
        interp.real2y = []
        interp.Z = []
        fig = pylab.figure()
        ax = Axes3D(fig)
        for i, img in enumerate(imgs):
            X, Y, Z, x, y = self.get_points(img)
            ax.scatter(X, Y, Z, marker='.', color='blue')

        x = range(0, self.nb_pixelx, self.nb_pixelx/10)
        y = range(0, self.nb_pixely, self.nb_pixely/10)
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
            X = (np.arange(10)-5)/20. * dx + X0
            Y = (np.arange(10)-5)/20. * dy + Y0
            Z = (np.arange(10)-5)/20. * dz + Z0
            ax.plot(X, Y, Z, 'r')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        plt.show()

    def check_interp_lines_coeffs(self):
        """ Plot to check interp_lines coefficients
        """
        x = range(0, self.nb_pixelx, self.nb_pixelx/100)
        y = range(0, self.nb_pixely, self.nb_pixely/100)
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
        pylab.title('X0')
        pylab.xlabel('x (pix)')
        pylab.ylabel('y (pix)')
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, Y0)
        pylab.title('Y0')
        pylab.xlabel('x (pix)')
        pylab.ylabel('y (pix)')
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, Z0)
        pylab.title('Z0')
        pylab.xlabel('x (pix)')
        pylab.ylabel('y (pix)')
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, dx)
        pylab.title('dx')
        pylab.xlabel('x (pix)')
        pylab.ylabel('y (pix)')
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, dy)
        pylab.title('dy')
        pylab.xlabel('x (pix)')
        pylab.ylabel('y (pix)')
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, dz)
        pylab.title('dz')
        pylab.xlabel('x (pix)')
        pylab.ylabel('y (pix)')
        pylab.colorbar()
        pylab.figure()
        pylab.pcolor(x, y, np.sqrt(dx**2 + dy**2 + dz**2))
        pylab.title('norm(d)')
        pylab.xlabel('x (pix)')
        pylab.ylabel('y (pix)')
        pylab.colorbar()

        plt.show()


class DirectStereoReconstruction():
    """Class to get stereo reconstruction with direct Calibration
    This calibration determine the equations of optical paths for "each" pixels

    Parameters
    ----------

    path_file0:
       Path of the file of the first camera

    path_file1:
       Path of the file of the second camera
    """
    def __init__(self, pth_file0, pth_file1):
        self.calib0 = CalibDirect(pth_file=pth_file0)
        self.calib1 = CalibDirect(pth_file=pth_file1)
        # matrices from camera planes to fixed plane and inverse
        self.A0, self.B0 = self.calib0.get_base_camera_plane()
        self.A1, self.B1 = self.calib1.get_base_camera_plane()

        if np.allclose(self.A0, self.A1):
            raise ValueError('The two calibrations have to be different.')

        # M1, M2: see reconstruction function
        self.invM0 = np.linalg.inv(
            np.vstack([self.B0[0:2, :], self.B1[0:1, :]]))
        self.invM1 = np.linalg.inv(
            np.vstack([self.B0[0:1, :], self.B1[0:2, :]]))
        self.invM2 = np.linalg.inv(
            np.vstack([self.B0[0:2, :], self.B1[1:2, :]]))
        self.invM3 = np.linalg.inv(
            np.vstack([self.B0[1:2, :], self.B1[0:2, :]]))

    def project2cam(self, indx0, indy0, dx0, dy0, indx1, indy1, dx1, dy1,
                    a, b, c, d, check=False):
        """ Project displacements of each cameras dx0, dy0, dx1 and dy1
        in their respective planes.
        """

        X0 = self.calib0.intersect_with_plane(indx0, indy0, a, b, c, d)
        dX0 = self.calib0.apply_calib(indx0, indy0, dx0, dy0, a, b, c, d)

        X1 = self.calib1.intersect_with_plane(indx1, indy1, a, b, c, d)
        dX1 = self.calib1.apply_calib(indx1, indy1, dx1, dy1, a, b, c, d)

        d0cam = np.tensordot(
            self.B0, dX0.swapaxes(0, 1), axes=1)[:2, :].transpose()
        d1cam = np.tensordot(
            self.B1, dX1.swapaxes(0, 1), axes=1)[:2, :].transpose()

        if check:
            plt.figure()
            plt.quiver(indx0, indy0, dx0, dy0)
            plt.axis('equal')
            pylab.xlabel('x (pix)')
            pylab.ylabel('y (pix)')
            plt.title('Cam 0 in pixel')
            plt.figure()
            plt.quiver(X0[:, 0], X0[:, 1], dX0[:, 0], dX0[:, 1])
            pylab.xlabel('x (pix)')
            pylab.ylabel('y (pix)')
            plt.axis('equal')
            plt.title('Cam 0 in m')
            plt.figure()
            plt.quiver(indx1, indy1, dx1, dy1)
            pylab.xlabel('x (pix)')
            pylab.ylabel('y (pix)')
            plt.axis('equal')
            plt.title('Cam 1 in pixel')
            plt.figure()
            plt.quiver(X1[:, 0], X1[:, 1], dX1[:, 0], dX1[:, 1])
            pylab.xlabel('x (pix)')
            pylab.ylabel('y (pix)')
            plt.axis('equal')
            plt.title('Cam 1 in m')
            plt.show()

        return X0, X1, d0cam, d1cam

    def find_common_grid(self, X0, X1, a, b, c, d):
        """ Find a common grid for the 2 cameras
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

        Nx0 = sqrt(X0[~np.isnan(X0[:, 0]), 0].size) * sqrt(Lx0/Ly0)
        Ny0 = X0[~np.isnan(X0[:, 1]), 1].size / Nx0
        Nx1 = sqrt(X1[~np.isnan(X1[:, 0]), 0].size) * sqrt(Lx1/Ly1)
        Ny1 = X1[~np.isnan(X1[:, 1]), 1].size / Nx1

        dx0 = Lx0 / Nx0
        dy0 = Ly0 / Ny0
        dx1 = Lx1 / Nx1
        dy1 = Ly1 / Ny1

        dx = max([dx0, dx1])
        dy = max([dy0, dy1])

        x = np.linspace(xmin, xmax, int((xmax-xmin)/dx))
        y = np.linspace(ymin, ymax, int((ymax-ymin)/dy))
        x, y = np.meshgrid(x, y)
        self.grid_x = x.transpose()
        self.grid_y = y.transpose()
        self.grid_z = -(a*self.grid_x + b*self.grid_y+d)/c
        return self.grid_x, self.grid_y, self.grid_z

    def interp_on_common_grid(
            self, X0, X1, d0cam, d1cam, grid_x, grid_y):
        """ Interpolate displacements of the 2 cameras d0cam, d1cam on the
        common grid grid_x, grid_y
        """
        # if not hasattr(self, 'grid_x'):
        #     self.find_common_grid(X0, X1, a, b, c, d)
        ind0 = (~np.isnan(X0[:, 0])) * (~np.isnan(X0[:, 1])) *\
               (~np.isnan(d0cam[:, 0])) * (~np.isnan(d0cam[:, 1]))
        ind1 = (~np.isnan(X1[:, 0])) * (~np.isnan(X1[:, 1])) *\
               (~np.isnan(d1cam[:, 0])) * (~np.isnan(d1cam[:, 1]))

        d0xcam = griddata((X0[ind0, 0], X0[ind0, 1]), d0cam[ind0, 0],
                          (grid_x, grid_y))
        d0ycam = griddata((X0[ind0, 0], X0[ind0, 1]), d0cam[ind0, 1],
                          (grid_x, grid_y))
        d1xcam = griddata((X1[ind1, 0], X1[ind1, 1]), d1cam[ind1, 0],
                          (grid_x, grid_y))
        d1ycam = griddata((X1[ind1, 0], X1[ind1, 1]), d1cam[ind1, 1],
                          (grid_x, grid_y))
        return d0xcam, d0ycam, d1xcam, d1ycam

    def reconstruction(self, X0, X1, d0cam, d1cam, a, b, c, d, grid_x, grid_y,
                       check=False):
        """Reconstruction of the 3 components of the velocity

        In the plane defined by a, b, c, d on the grid defined by grid_x,
        grid_y from the displacements of the 2 cameras d0cam, d1cam in their
        respective planes d0cam, d1cam.

        """
        # reconstruction => resolution of the equation
        # MX = dcam

        d0xcam, d0ycam, d1xcam, d1ycam = self.interp_on_common_grid(
            X0, X1, d0cam, d1cam, grid_x, grid_y)

        # in the case where 2 vectors from different cameras are approximately
        # the same, they don't have to be used both in the computation
        n1 = np.abs(np.inner(self.A0[0], self.A1[0]))
        n2 = np.abs(np.inner(self.A0[1], self.A1[1]))
        n3 = np.abs(np.inner(self.A0[1], self.A1[0]))
        n4 = np.abs(np.inner(self.A0[0], self.A1[1]))
        # I suppose that 5deg between vectors is sufficient
        threshold = np.cos(5*2*np.pi/180.)

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
        dX = np.nanmean(tmp, 0)[0]
        Errorx = np.nanstd(tmp, 0)[0]
        dY = np.nanmean(tmp, 0)[1]
        Errory = np.nanstd(tmp, 0)[1]
        dZ = np.nanmean(tmp, 0)[2]
        Errorz = np.nanstd(tmp, 0)[2]

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
            pylab.xlabel('X')
            pylab.ylabel('Y')
            plt.axis('equal')
            plt.title('Cam 0 plane projection')
            plt.figure()
            plt.quiver(self.grid_x, self.grid_y, d1xcam, d1ycam)
            pylab.xlabel('X')
            pylab.ylabel('Y')
            plt.axis('equal')
            plt.title('Cam 1 plane projection')
            plt.figure()
            plt.quiver(self.grid_x, self.grid_y, dX, dY)
            pylab.xlabel('X')
            pylab.ylabel('Y')
            plt.axis('equal')
            plt.title('Reconstruction on laser plane')
            plt.show()

        return dX, dY, dZ, Errorx, Errory, Errorz


if __name__ == "__main__":
    def clf():
        pylab.close('all')

    nb_pixelx, nb_pixely = 1024, 1024

    nbline_x, nbline_y = 32, 32

    path_cam = ('../../image_samples/4th_PIV-Challenge_Case_E/'
                'E_Calibration_Images/Camera_0')

    pathimg = path_cam + '1/img*'
    calib = CalibDirect(pathimg, (nb_pixelx, nb_pixely))
    calib.compute_interpolents()
    calib.compute_interppixel2line((nbline_x, nbline_y), test=False)
    calib.save(path_cam + '1/calib1.npy')

    # calib.check_interp_lines_coeffs()
    # calib.check_interp_lines()
    # calib.check_interp_levels()
    pathimg = path_cam + '3/img*'
    calib3 = CalibDirect(pathimg, (nb_pixelx, nb_pixely))
    calib3.compute_interpolents()
    calib3.compute_interppixel2line((nbline_x, nbline_y), test=False)
    calib3.save(path_cam + '3/calib3.npy')

    stereo = DirectStereoReconstruction(
        path_cam + '1/calib1.npy', path_cam + '3/calib3.npy')
