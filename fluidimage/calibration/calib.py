
import re

import numpy as np
import glob
import os
import pylab
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RegularGridInterpolator

from fluiddyn.util.paramcontainer import ParamContainer, tidy_container

def get_number_from_string(string):
    return map(float, re.findall(r"[-+]?\d*\.\d+|\d+", string))

class Interpolent():
    pass

class ParamCalibration(ParamContainer):

    def __init__(self, tag=None, attribs=None,
                 path_file=None, elemxml=None, hdf5_object=None):

        if path_file is None:
            super(ParamCalibration, self).__init__(
                tag=tag, attribs=attribs,
                elemxml=elemxml, hdf5_object=hdf5_object)
            return

        super(ParamCalibration, self).__init__(tag='calib')

        calib_uvmat = ParamContainer(path_file=path_file)
        tidy_container(calib_uvmat)

        calib_uvmat = calib_uvmat['geometry_calib']

        f = map(
            float, np.asarray(get_number_from_string(calib_uvmat.fx_fy)))
        C = np.asarray(get_number_from_string(calib_uvmat.cx__cy))
        kc = np.asarray(calib_uvmat.kc)
        T = np.asarray(get_number_from_string(calib_uvmat.tx__ty__tz))

        R = []
        for i in range(3):
            R = np.hstack([
                R, get_number_from_string(calib_uvmat['r_{}'.format(i+1)])])

        omc = np.asarray(get_number_from_string(calib_uvmat['omc']))

        self._set_attribs(
            {'f': f, 'C': C, 'kc': kc, 'T': T, 'R': R, 'omc': omc})

        if calib_uvmat.nb_slice is not None:

            nb_slice = np.asarray(calib_uvmat['nb_slice'])
            zslice_coord = np.zeros([nb_slice, 3])
            
            if calib_uvmat.nb_slice == 1:
                zslice_coord[:] = get_number_from_string(
                    calib_uvmat['slice_coord'])
                if hasattr(calib_uvmat, 'slice_angle') and calib_uvmat['slice_angle'] is not None:
                    slice_angle = np.zeros([nb_slice, 3])
                    slice_angle[:] = get_number_from_string(
                        calib_uvmat['slice_angle'])
                else:
                    slice_angle = [0, 0, 0]
            else:
                for i in range(nb_slice):
                    zslice_coord[i][:] = get_number_from_string(
                        calib_uvmat['slice_coord_{}'.format(i+1)])

                if hasattr(calib_uvmat, 'slice_angle_1') and calib_uvmat['slice_angle_1'] is not None:
                    slice_angle = np.zeros([nb_slice, 3])
                    for i in range(nb_slice):
                        slice_angle[i][:] = get_number_from_string(
                            calib_uvmat['slice_angle_{}'.format(i+1)])
                else:
                    slice_angle = [0, 0, 0]

            self._set_child('slices', attribs={
                'nb_slice': nb_slice,
                'zslice_coord': zslice_coord,
                'slice_angle': slice_angle})

        if hasattr(calib_uvmat, 'refraction_index'):
            self._set_attrib('refraction_index', calib_uvmat.refraction_index)

        if hasattr(calib_uvmat, 'interface_coord'):
            self._set_attrib(
                'interface_coord',
                get_number_from_string(calib_uvmat['interface_coord']))


class Calibration(object):
    def __init__(self, path_file):
        self.path_file=path_file;
        self.params = ParamCalibration(path_file=path_file)

    def pix2phys_UV(self, X, Y, dx, dy, index_level, nbypix, angle=True):
        """Apply Tsai Calibration to the field

        Notes
        -----

        The displacement is 3 component BUT it corresponds to the real 3
        component displacement projected on the corresponding plane indexed by
        index_level.

        """

        Xphys, Yphys, Zphys = self.pix2phys(X, Y, index_level=index_level, nbypix=nbypix, angle=True)
        dxb, dyb, dzb = self.pix2phys(X + dx/2.0, Y + dy/2.0, index_level, nbypix=nbypix, angle=True)
        dxa, dya, dza = self.pix2phys(X - dx/2.0, Y - dy/2.0, index_level, nbypix=nbypix, angle=True)
        dxphys = dxb - dxa
        dyphys = dyb - dya
        dzphys = dzb - dza
        return Xphys, Yphys, Zphys, dxphys, dyphys, dzphys

    def pix2phys(self, X, Y, index_level, nbypix, angle=True):
        params = ParamCalibration(path_file=self.path_file)
        Y = nbypix-Y # difference of convention with calibration done with uvmat!       
        # determine position of Z0
        testangle = 0
        if hasattr(params.slices, 'slice_angle') and \
           np.any(params.slices.slice_angle[index_level] !=
                  np.asarray([0, 0, 0])) and angle:
            testangle = 1
            om = np.linalg.norm(params.slices.slice_angle[index_level])
            axis_rot = params.slices.slice_angle[index_level] / om
            cos_om = np.cos(np.pi*om/180.0)
            sin_om = np.sin(np.pi*om/180.0)
            coeff = axis_rot[2]*(1-cos_om)
            norm_plane = np.zeros(3)
            norm_plane[0] = axis_rot[0]*coeff+axis_rot[1]*sin_om
            norm_plane[1] = axis_rot[1]*coeff-axis_rot[0]*sin_om
            norm_plane[2] = axis_rot[2]*coeff+cos_om
            Z0 = np.dot(
                norm_plane,
                (params.slices.zslice_coord[index_level]))/norm_plane[2]
        else:
            Z0 = params.slices.zslice_coord[index_level][2]
        Z0virt = Z0
        if hasattr(params, 'interface_coord') and \
           hasattr(params, 'refraction_index'):
            H = params.interface_coord[2]
            if H > Z0:
                Z0virt = H-(H-Z0)/params.refraction_index
                # corrected z (virtual object)
                test_refraction = 1

        if hasattr(params, 'f') is False:
            params.f = np.asarray([1, 1])
        if hasattr(params, 'T') is False:
            params.T = np.asarray([0, 0, 1])
        if hasattr(params, 'C') is False:
            params.C = np.asarray([0, 0])
        if hasattr(params, 'kc') is False:
            params.kc = 0

        if hasattr(params, 'R'):
            R = params.R
            #R[0]= params.R[4]
            #R[1]= params.R[3]
            #R[2]= params.R[5]
            #R[4]= params.R[0]
            #R[5]= params.R[2]
            #R[6]= params.R[7]
            #R[7]= params.R[6]

            #R[1]= params.R[3]
            #R[3]= params.R[1]
            #R[2]= params.R[6]
            #R[6]= params.R[2]
            #R[5]= params.R[5]
            #R[7]= params.R[7]
            
            if testangle:
                a = -norm_plane[0]/norm_plane[2]
                b = -norm_plane[1]/norm_plane[2]
                if test_refraction:
                    a /= params.refraction_index
                    b /= params.refraction_index

                    R[0] += a*R[2]
                    R[1] += b*R[2]
                    R[3] += a*R[5]
                    R[4] += b*R[5]
                    R[6] += a*R[8]
                    R[7] += b*R[8]

            Tx = params.T[0]
            Ty = params.T[1]
            Tz = params.T[2]
            Dx = R[4]*R[6]-R[3]*R[7]
            Dy = R[0]*R[7]-R[1]*R[6]
            D0 = R[1]*R[3]-R[0]*R[4]
            Z11 = R[5]*R[7]-R[4]*R[8]
            Z12 = R[1]*R[8]-R[2]*R[7]
            Z21 = R[3]*R[8]-R[5]*R[6]
            Z22 = R[2]*R[6]-R[0]*R[8]
            Zx0 = R[2]*R[4]-R[1]*R[5]
            Zy0 = R[0]*R[5]-R[2]*R[3]

            A11 = R[7]*Ty-R[4]*Tz+Z11*Z0virt
            A12 = R[1]*Tz-R[7]*Tx+Z12*Z0virt
            A21 = -R[6]*Ty+R[3]*Tz+Z21*Z0virt
            A22 = -R[0]*Tz+R[6]*Tx+Z22*Z0virt


            X0 = (R[4]*Tx-R[1]*Ty+Zx0*Z0virt)
            Y0 = (-R[3]*Tx+R[0]*Ty+Zy0*Z0virt)

            Xd = (X-params.C[0])/params.f[0]  # sensor coordinates
            Yd = (Y-params.C[1])/params.f[1]
            dist_fact = 1 + params.kc*(Xd*Xd+Yd*Yd)
            Xu = Xd/dist_fact  # undistorted sensor coordinates
            Yu = Yd/dist_fact
            denom = Dx*Xu+Dy*Yu+D0
            Xphys = (A11*Xu+A12*Yu+X0)/denom  # world coordinates
            Yphys = (A21*Xu+A22*Yu+Y0)/denom
            if testangle:
                Zphys = Z0+a*Xphys+b*Yphys
            else:
                Zphys = Z0
        else:
            Xphys = -params.T[0]+X/params.f[0]
            Yphys = -params.T[1]+Y/params.f[1]
            Zphys = X*0
        Xphys /= 100  # cm to m
        Yphys /= 100  # cm to m
        Zphys /= 100  # cm to m

        return Xphys, Yphys, Zphys

    def phys2pix(self, Xphys, Yphys, Zphys=0):
        params = self.params
        Xphys *= 100  # m to cm
        Yphys *= 100  # m to cm
        Zphys *= 100  # m to cm

        if hasattr(params, 'f') is False:
            params.f = np.asarray([1, 1])
        if hasattr(params, 'T') is False:
            params.T = np.asarray([0, 0, 1])

        # general case
        if hasattr(params, 'R'):
            R = params.R
            if hasattr(params, 'interface_coord') and \
               hasattr(params, 'refraction_index'):
                H = params.interface_coord[2]
                if H > Zphys:
                    Zphys = H-(H-Zphys)/params.refraction_index
                    # corrected z (virtual object)

            # camera coordinates
            xc = R[0]*Xphys+R[1]*Yphys+R[2]*Zphys+params.T[0]
            yc = R[3]*Xphys+R[4]*Yphys+R[5]*Zphys+params.T[1]
            zc = R[6]*Xphys+R[7]*Yphys+R[8]*Zphys+params.T[2]

            # undistorted image coordinates
            Xu = xc/zc
            Yu = yc/zc

            # radial quadratic correction factor
            if hasattr(params, 'kc') is False:
                r2 = 1  # no quadratic distortion
            else:
                r2 = 1 + params.kc*(Xu*Xu+Yu*Yu)

            # pixel coordinates
            if hasattr(params, 'C') is False:
                params.C = np.asarray([0, 0])  # default value

            X = params.f[0]*Xu*r2 + params.C[0]
            Y = params.f[1]*Yu*r2 + params.C[1]

        # case 'rescale'
        else:
            X = params.f[0]*(Xphys+params.T[0])
            Y = params.f[1]*(Yphys+params.T[1])

        return X, Y

    def get_coeff(Calib, X, Y, x, y, z):
        # compute A~ coefficients 
        R = self.R
        T_z = self.T[2]
        T = R[6] * x + R[7]*y+R[8] * z + T_z;

        A[:, :, 0, 0] = (R[0] - R[6] * X) / T
        A[:, :, 0, 1] = (R[1] - R[7] * X) / T
        A[:, :, 0, 2] = (R[2] - R[8] * X) / T
        A[:, :, 1, 0] = (R[3] - R[6] * Y) / T
        A[:, :, 1, 1] = (R[4] - R[7] * Y) / T
        A[:, :, 1, 2] = (R[5] - R[8] * Y) / T
        return A

    def ud2u(Xd, Yd, Ud, Vd):
        #convert image coordinates to view angles, after removal of  quadratic distorsion
        # input in pixel, output in radians
        X1d = Xd-Ud/2;
        X2d = Xd+Ud/2;
        Y1d = Yd-Vd/2;
        Y2d = Yd+Vd/2;

        X1 = (X1d - self.C[0]) /  self.f[0] * \
             (1 + self.kc * self.f[0]**(-2)  * (X1d - self.C[0])**2 + \
              self.kc * self.f[1]**(-2) * (Y1d - self.C[1])**2 )**(-1)
        X1 = (X2d - self.C[0]) /  self.f[0] * \
             (1 + self.kc * self.f[0]**(-2)  * (X2d - self.C[0])**2 + \
              self.kc * self.f[1]**(-2) * (Y2d - self.C[1])**2 )**(-1)
        X1 = (Y1d - self.C[1]) /  self.f[1] * \
             (1 + self.kc * self.f[0]**(-2)  * (X1d - self.C[0])**2 + \
              self.kc * self.f[1]**(-2) * (Y1d - self.C[1])**2 )**(-1)
        X1 = (Y2d - self.C[1]) /  self.f[1] * \
             (1 + self.kc * self.f[0]**(-2)  * (X2d - self.C[0])**2 + \
              self.kc * self.f[1]**(-2) * (Y2d - self.C[1])**2 )**(-1)

        U=X2-X1
        V=Y2-Y1
        X=X1+U/2
        Y=Y1+V/2
        return U, V, X, Y


class StereoReconstruction(Calibration):
    def __init__(self, path_file1, path_file2):
        self.field1 = Calibration(path_file1)
        self.field2 = Calibration(path_file2)

    def shift2z(xmid, ymid, u, v):
        z=0;
        error=0;
        
        # first image
        R = self.field1.R
        T = self.field1.T
        x_a = xmid - u/2
        y_a = ymid - v/2 
        z_a = R[6] * x_a + R[7] * y_a + T[0,2]
        Xa = (R[0] * x_a + R[1] * y_a + T[0, 0]) / z_a
        Ya = (R[3] * x_a + R[4] * y_a + T[0,1]) / z_a

        A_1_1=R[0] - R[6] * Xa;
        A_1_2=R[1] - R[7] * Xa;
        A_1_3=R[2] - R[8] * Xa;
        A_2_1=R[3] - R[6] * Ya;
        A_2_2=R[4] - R[7] * Ya;
        A_2_3=R[5] - R[8] * Ya;
        Det = A_1_1 * A_2_2 - A_1_2 * A_2_1;
        Dxa = (A_1_2 * A_2_3 - A_2_2 * A_1_3) / Det;
        Dya = (A_2_1 * A_1_3 - A_1_1 * A_2_3) / Det;

        #second image
        #loading shift angle

        R = self.field2.R
        T = self.field2.T
        x_b = xmid - u/2
        y_b = ymid - v/2 
        z_b = R[6] * x_b + R[7] * y_b + T[0,2]
        Xb = (R[0] * x_b + R[1] * y_b + T[0, 0]) / z_b
        Yb = (R[3] * x_b + R[4] * y_b + T[0,1]) / z_b

        B_1_1=R[0] - R[6] * Xb
        B_1_2=R[1] - R[7] * Xb
        B_1_3=R[2] - R[8] * Xb
        B_2_1=R[3] - R[6] * Yb
        B_2_2=R[4] - R[7] * Yb
        B_2_3=R[5] - R[8] * Yb
        Det = B_1_1 * B_2_2 - B_1_2 * B_2_1
        Dxb = (B_1_2 * B_2_3 - B_2_2 * B_1_3) / Det
        Dyb = (B_2_1 * B_1_3 - B_1_1 * B_2_3) / Det

        # result
        Den = (Dxb - Dxa) * (Dxb - Dxa) + (Dyb - Dya) * (Dyb - Dya);
        error=abs(((Dyb - Dya) *(-u) - (Dxb - Dxa) * (-v))) / Den;
        z=((Dxb - Dxa) * (-u) + (Dyb - Dya) * (-v)) /Den;

        xnew[0,:] = Dxa * z + x_a;
        xnew[1,:] = Dxb * z + x_b;
        ynew[0,:] = Dya * z + y_a;
        ynew[1,:] = Dyb * z + y_b;
        Xphy=mean(xnew,0);
        Yphy=mean(ynew,0); 
        
        return z, Xphy, Yphy, error

    def stereo_reconstruction(X1, Y1, U1, V1, Xa, Ya, X2, Y2, U2, V2, Xb, Yb):
        # initialisatiion des matrices
        # xI=ObjectData.RangeX(1):ObjectData.DX:ObjectData.RangeX(2);
        # yI=ObjectData.RangeY(1):ObjectData.DY:ObjectData.RangeY(2);
        # XI, YI = np.meshgrid(xI, yI);
        # ZI = ??
        
        U=zeros(size(XI,1),size(XI,2));
        V=zeros(size(XI,1),size(XI,2));
        W=zeros(size(XI,1),size(XI,2));
        
        Ua = np.griddata(X1, Y1, U1, Xa, Ya)
        Va = np.griddata(X1,Y1,V1,Xa,Ya);
        Ua, Va, Xa, Ya = self.field1.Ud2U(Xa,Ya,Ua,Va)
        A = self.field1.get_coeff(Xa,Ya,XI,YI,ZI)
    
        Ub=griddata(X2,Y2,U2,Xb,Yb);
        Vb=griddata(X2,Y2,V2,Xb,Yb);
        Ub, Vb, Xb, Yb = self.field2.Ud2U(Xb, Yb, Ub, Vb)
        B = self.field2.get_coeff(Xb, Yb, XI, YI, ZI)
    
        S = ones(size(XI,0), size(XI,1), 3);
        D = np.ones(size(XI,0), size(XI,1), 3, 3);
    
        S[:, :, 0] = A[:, :, 0, 0] * Ua + A[: ,: ,1 ,0] * Va + B[:, :, 0, 0] *\
                     Ub + B[:, :, 1, 0] * Vb
        S[:, :, 1] = A[:, :, 0, 1] * Ua + A[:, : ,1, 1] * Va + B[:, :, 0, 1] * \
                     Ub + B[:, :, 1, 1] * Vb
        S[:, :, 2] = A[:, :, 0, 2] * Ua + A[:, :, 1, 2] * Va + B[:, :, 0, 2] * \
                     Ub + B[:, :, 1, 2] * Vb
        D[:, :, 0, 0] = A[:, :, 0, 0] * A[:, :, 0, 0] + A[:, :, 1, 0] * \
                        A[:, :, 1, 0] + B[:, :, 0, 0] * B[:, :, 0, 0] + \
                        B[:, :, 1, 0] * B[:, :, 1, 0]
        D[:, :, 0, 1] = A[:, :, 0, 0] * A[:, :, 0, 1] + A[:, :, 1, 0] * \
                        A[:, :, 1, 1] + B[:, :, 0, 0] * B[:, :, 0, 1] + \
                        B[:, :, 1, 0] * B[:, :, 1, 1]
        D[:, :, 0, 2] = A[:, :, 0, 0] * A[:, :, 0, 2] + A[:, :, 1, 0] * \
                        A[:, :, 1, 2] + B[:, :, 0, 0] * B[:, :, 0, 2] +\
                        B[:, :, 1, 0] * B[:, :, 1, 2]
        D[:, :, 1, 0] = A[:, :, 0, 1] *A[:, :, 0, 0] + A[:, :, 1, 1] *\
                        A[:, : ,1 ,0] + B[:, :, 0, 1] * B[:, :, 0, 0] +\
                        B[:, :, 1, 1] * B[:, :, 1, 0]
        D[:, :, 1, 1] = A[:, :, 0, 1] * A[:, :, 0, 1] + A[:, :, 1, 1] *\
                        A[:, :, 1, 1] + B[:, :, 0, 1] * B[: ,: ,0 ,1] +\
                        B[:, :, 1, 1] * B[:, :, 1, 1]
        
        D[:, :, 1, 2] = A[:, :, 0, 1] *A[:, :, 0, 2] + A[:, :, 1, 1] *\
                        A[:, :, 1, 2] + B[:, :, 0, 1] * B[:, :, 0, 2] + \
                        B[:, :, 1, 1] * B[:, :, 1, 2]
        D[:, :, 2, 0] = A[:, :, 0, 2] * A[:, :, 0, 0] + A[:, :, 1, 2]* \
                        A[:, :, 1, 0] + B[:, :, 0, 2] * B[:, :, 0, 0] + \
                        B[:, :, 1, 2] * B[:, :, 1, 0]
        D[:, :, 2, 1] = A[:, :, 0, 2] * A[:, :, 0, 1] + A[:, :, 1, 2] * \
                        A[:, :, 1, 1] + B[:, :, 0, 2] * B[:, :, 0, 1] + \
                        B[:, :, 1, 2] * B[:, :, 1, 1]
        D[:, :, 2, 2] = A[:, :, 0, 2]  * A[:, :, 0, 2] + A[:, :, 1, 2] * \
                        A[:, :, 0, 2] + B[:, :, 0, 2] * B[:, :, 0, 2] +\
                        B[:, :, 1, 2] * B[:, :, 1, 2]
        
        for indj in range(np.size(XI)[0]):
            for indi in range(np.size(XI)[1]):
                dxyz, resid,rank, s = np.linalg.lstsq(
                    np.squeeze(D[indj, indi, :, :])*1000,
                    np.squeeze(S[indj, indi, :])*1000)
                # U(indj,indi)=dxyz[0]
                # V(indj,indi)=dxyz[1]
                # W(indj,indi)=dxyz[2]

        Error = zeros(np.size(XI)[0], np.size(XI)[1],4);
        Error[:, :, 0] = A[:, :, 0, 0] * U + A[:, :, 0, 1] * \
                         V + A[:, :, 0, 2] * W - Ua
        Error[:, :, 1] = A[:, :, 1, 0] * U + A[: ,: ,1 ,1] *\
                         V + A[:, :, 1, 2] * W - Va
        Error[:, :, 2] = B[:, :, 0, 0] * U + B[:, :, 0, 1] *\
                         V + B[:, :, 0, 2] * W - Ub
        Error[:, :, 3] = B[:, :, 1, 0] * U + B[:, :, 1, 1] *\
                         V + B[:, :, 1, 2] * W - Vb

        Error=0.5*sqrt(sum(Error**2, 3))
        return xI, yI, U, V, W, ZI, error


class CalibDirect():
    def __init__(self, pathimg, nb_pixely):
        pass
    
    def compute_interpolents(self, interpolator = CloughTocher2DInterpolator):

        imgs = glob.glob(os.path.join(pathimg, 'img*.xml'))
        interp = Interpolent()
        interp.cam2X = []
        interp.cam2Y = []
        interp.real2x = []
        interp.real2y = []
        interp.Z = []
    
        for i, img in enumerate(imgs):
            imgpts = ParamContainer(path_file=img)
            tidy_container(imgpts)
            imgpts = imgpts.geometry_calib.source_calib.__dict__
            pts = ([x for x in imgpts.keys() if 'point_' in x or 'Point' in x])
        
            # coord in real space
            X = np.array([get_number_from_string(imgpts[tmp])[0] for tmp in pts])/100.
            Y = np.array([get_number_from_string(imgpts[tmp])[1] for tmp in pts])/100.
            interp.Z.append(get_number_from_string(imgpts[pts[0]])[2]/100.)
            # coord in image
            x = np.array([get_number_from_string(imgpts[tmp])[3] for tmp in pts])
            y = nb_pixely - \
                np.array([get_number_from_string(imgpts[tmp])[4] for tmp in pts])
            cam = np.vstack([x,y]).transpose()
            real = np.vstack([X,Y]).transpose()

            interp.cam2X.append(interpolator(cam, X))
            interp.cam2Y.append(interpolator(cam, Y))
            interp.real2x.append(interpolator(real, x))
            interp.real2y.append(interpolator(real, y))
        self.interp_levels = interp

    def compute_interppixel2line(self, nbpix_x, nbpix_y, nbline_x, nbline_y):
        x = np.unique(np.floor(np.linspace(0, nbpix_x, nbline_x)))
        y = np.unique(np.floor(np.linspace(0, nbpix_y, nbline_y)))

        X, Y = np.meshgrid(x, y)
        V = np.zeros((X.shape[0], X.shape[1], 6))

        Xtrue = []
        Ytrue = []
        Vtrue = []
        Xfalse = []
        Yfalse = []
        indi = []
        indj = []

        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                tmp = self.pixel2line(X[i,j], Y[i,j])
                if np.isnan(tmp[0]):
                    Xfalse.append(X[i,j])
                    Yfalse.append(Y[i,j])
                    indi.append(i)
                    indj.append(j)
                else:
                    V[i, j, :] = tmp
                    Xtrue.append(X[i,j])
                    Ytrue.append(Y[i,j])
                    Vtrue.append(tmp)
                    
        # for j in range(6):
        #     indfalse = np.where(isnan(V[:, :, j]))
        #     indtrue = np.where(isnan(V[:, :, j]) == False)
        #     V[indfalse[0], indfalse[1], j]  = griddata((X[indfalse], Y[indfalse]),
        #                     V[indtrue[0], indtrue[1], j],
        #                     (X[indtrue], Y[indtrue]))
            
        Vtrue = np.array(Vtrue)
        for j in range(6):
            V[indi, indj, j] = griddata((Xtrue, Ytrue),
                                                  Vtrue[:, j],
                                                  (Xfalse, Yfalse))

        interp = []
        for i in range(6):
            interp.append(RegularGridInterpolator((x, y), V[:, :, i]))

        self.interp_lines = interp                                   

    def pixel2line(self, indx, indy):
        interp = self.interp_levels
        X = []
        Y = []
        Z = interp.Z
        for i in range(len(Z)):
            X.append(float(interp.cam2X[i]((indx, indy))))
            Y.append(float(interp.cam2Y[i]((indx, indy))))
        X = np.array(X)
        Y = np.array(Y)
        XYZ = np.vstack([X, Y, Z]).transpose()
        XYZ0 = np.nanmean(XYZ, 0)
        XYZ -= XYZ0
    
        ind = np.isnan(X+Y) == False
        XYZ = XYZ[ind, :]
        if XYZ.shape[0] > 1:
            u, s, v = np.linalg.svd(XYZ, full_matrices=True, compute_uv=1)
            direction=np.cross(v[:,-1],v[:,-2])
            return np.hstack([XYZ0, direction])
        else:
            return np.hstack([np.nan]*6)

    def check(self):
        imgs = glob.glob(os.path.join(pathimg, 'img*.xml'))
        interp = Interpolent()
        interp.cam2X = []
        interp.cam2Y = []
        interp.real2x = []
        interp.real2y = []
        interp.Z = []
        fig = pylab.figure()
        ax = Axes3D(fig)
        for i, img in enumerate(imgs):
            imgpts = ParamContainer(path_file=img)
            tidy_container(imgpts)
            imgpts = imgpts.geometry_calib.source_calib.__dict__
            pts = ([x for x in imgpts.keys() if 'point_' in x or 'Point' in x])
        
            # coord in real space
            X = np.array([get_number_from_string(imgpts[tmp])[0] for tmp in pts])/100.
            Y = np.array([get_number_from_string(imgpts[tmp])[1] for tmp in pts])/100.
            Z = np.array([get_number_from_string(imgpts[tmp])[2] for tmp in pts])/100.
            #ax.scatter(X,Y,Z, marker='+')

        x = range(500, 800, 10)
        y = range(500, 800, 10)
        x, y =  np.meshgrid(x,y)
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
            ax.plot(X,Y,Z)
        pylab.show()

    def check_interp_lines_coeffs(self):
        x = range(500, 800, 10)
        y = range(500, 800, 10)
        x, y =  np.meshgrid(x,y)
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
         
        fig = pylab.figure()
        pylab.pcolor(x, y, X0)
        pylab.colorbar()
        fig.show()
        fig = pylab.figure()
        pylab.pcolor(x, y, Y0)
        pylab.colorbar()
        fig.show()
        fig = pylab.figure()
        pylab.pcolor(x, y, Z0)
        pylab.colorbar()
        fig.show()
        fig = pylab.figure()
        pylab.pcolor(x, y, dx)
        pylab.colorbar()
        fig.show()
        fig = pylab.figure()
        pylab.pcolor(x, y, dy)
        pylab.colorbar()
        fig.show()
        fig = pylab.figure()
        pylab.pcolor(x, y, dz)
        pylab.colorbar()
        fig.show()
        fig = pylab.figure()
        pylab.pcolor(x, y, np.sqrt(dx**2 + dy**2 + dz**2))
        pylab.colorbar()
        fig.show()

if __name__ == "__main__":
    def clf():
        pylab.close('all')
    pathimg = '/.fsdyn_people/campagne8a/project/16BICOUCHE/Antoine/0_Ref_mika/Dalsa1'
    nb_pixely = 1024
    calib = CalibDirect(pathimg, nb_pixely)
    calib.compute_interpolents()
    nbpix_x, nbpix_y, nbline_x, nbline_y= 1024, 1024, 512, 512
    test = calib.compute_interppixel2line(nbpix_x, nbpix_y, nbline_x, nbline_y)
    calib.check_interp_lines_coeffs()

    


