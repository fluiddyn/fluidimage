
import re

import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer, tidy_container


def get_number_from_string(string):
    return map(float, re.findall(r"[-+]?\d*\.\d+|\d+", string))


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

            for i in range(nb_slice):
                zslice_coord[i][:] = get_number_from_string(
                    calib_uvmat['slice_coord_{}'.format(i+1)])

            if calib_uvmat['slice_angle_1'] is not None:
                slice_angle = np.zeros([nb_slice, 3])
                for i in range(nb_slice):
                    slice_angle[i][:] = get_number_from_string(
                        calib_uvmat['slice_angle_{}'.format(i+1)])

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
        self.params = ParamCalibration(path_file=path_file)

    def pix2phys_UV(self, X, Y, dx, dy, index_level=0):
        """Apply Tsai Calibration to the field

        Notes
        -----

        The displacement is 3 component BUT it corresponds to the real 3
        component displacement projected on the corresponding plane indexed by
        index_level.

        """

        Xphys, Yphys, Zphys = self.pix2phys(X, Y, index_level)

        dxb, dyb, dzb = self.pix2phys(X + dx/2.0, Y + dy/2.0, index_level)
        dxa, dya, dza = self.pix2phys(X - dx/2.0, Y - dy/2.0, index_level)
        dxphys = dxb - dxa
        dyphys = dyb - dya
        dzphys = dzb - dza
        return Xphys, Yphys, Zphys, dxphys, dyphys, dzphys

    def pix2phys(self, X, Y, index_level=0):
        params = self.params

        # determine position of Z0
        testangle = 0
        if hasattr(params.slices, 'slice_angle') and \
           np.any(params.slices.slice_angle[index_level] !=
                  np.asarray([0, 0, 0])):
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
            params.C = np.asarray([0, 0, 1])
        if hasattr(params, 'kc') is False:
            params.kc = 0

        if hasattr(params, 'R'):
            R = params.R

            if testangle:
                a = -norm_plane[0]/norm_plane[2]
                b = -norm_plane[1]/norm_plane[2]
                if test_refraction:
                    a /= params.refraction_index
                    b /= params.refraction_index

                R[0] = R[0]+a*R[2]
                R[1] = R[1]+b*R[2]
                R[3] = R[3]+a*R[5]
                R[4] = R[4]+b*R[5]
                R[6] = R[6]+a*R[8]
                R[7] = R[7]+b*R[8]

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

            #     X0=Params.fx_fy(1)*(R(5)*Tx-R(2)*Ty+Zx0*Z0virt)
            #     Y0=Params.fx_fy(2)*(-R(4)*Tx+R(1)*Ty+Zy0*Z0virt)
            X0 = (R[4]*Tx-R[1]*Ty+Zx0*Z0virt)
            Y0 = (-R[3]*Tx+R[0]*Ty+Zy0*Z0virt)
            # px to camera:
            #     Xd=dpx*(X-Params.Cx_Cy(1)) % sensor coordinates
            #     Yd=(Y-Params.Cx_Cy(2))
            Xd = (X-params.C[0])/params.f[0]  # sensor coordinates
            Yd = (Y-params.C[1])/params.f[1]
            dist_fact = 1 + params.kc*(Xd*Xd+Yd*Yd)
            # /(f*f) %distortion factor
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
