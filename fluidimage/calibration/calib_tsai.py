import copy

import numpy as np

from .util import make_params_calibration


class Calibration:
    def __init__(self, path_file):
        self.path_file = path_file
        self.params = make_params_calibration(path_file)

    def pix2phys_UV(self, X, Y, dx, dy, index_level, nbypix, angle=True):
        """Apply Tsai Calibration to the field

        Notes
        -----

        The displacement is 3 component BUT it corresponds to the real 3
        component displacement projected on the corresponding plane indexed by
        index_level.

        """

        Xphys, Yphys, Zphys = self.pix2phys(
            X, Y, index_level=index_level, nbypix=nbypix, angle=angle
        )
        dxb, dyb, dzb = self.pix2phys(
            X + dx / 2.0, Y + dy / 2.0, index_level, nbypix=nbypix, angle=angle
        )
        dxa, dya, dza = self.pix2phys(
            X - dx / 2.0, Y - dy / 2.0, index_level, nbypix=nbypix, angle=angle
        )
        dxphys = dxb - dxa
        dyphys = dyb - dya
        dzphys = dzb - dza
        return Xphys, Yphys, Zphys, dxphys, dyphys, dzphys

    def pix2phys(self, X, Y, index_level, nbypix, angle=True):
        params = copy.deepcopy(self.params)
        # difference of convention with calibration done with uvmat!
        Y = nbypix - Y
        # determine position of Z0
        testangle = 0
        if (
            hasattr(params.slices, "slice_angle")
            and np.any(
                params.slices.slice_angle[index_level] != np.asarray([0, 0, 0])
            )
            and angle
        ):
            testangle = 1
            om = np.linalg.norm(params.slices.slice_angle[index_level])
            axis_rot = params.slices.slice_angle[index_level] / om
            cos_om = np.cos(np.pi * om / 180.0)
            sin_om = np.sin(np.pi * om / 180.0)
            coeff = axis_rot[2] * (1 - cos_om)
            norm_plane = np.zeros(3)
            norm_plane[0] = axis_rot[0] * coeff + axis_rot[1] * sin_om
            norm_plane[1] = axis_rot[1] * coeff - axis_rot[0] * sin_om
            norm_plane[2] = axis_rot[2] * coeff + cos_om
            Z0 = (
                np.dot(norm_plane, (params.slices.zslice_coord[index_level]))
                / norm_plane[2]
            )
        else:
            Z0 = params.slices.zslice_coord[index_level][2]
        Z0virt = Z0
        if hasattr(params, "interface_coord") and hasattr(
            params, "refraction_index"
        ):
            H = params.interface_coord[2]
            if H > Z0:
                Z0virt = H - (H - Z0) / params.refraction_index
                # corrected z (virtual object)
                test_refraction = 1

        if hasattr(params, "f") is False:
            params.f = np.asarray([1, 1])
        if hasattr(params, "T") is False:
            params.T = np.asarray([0, 0, 1])
        if hasattr(params, "C") is False:
            params.C = np.asarray([0, 0])
        if hasattr(params, "kc") is False:
            params.kc = 0

        if hasattr(params, "R"):
            R = copy.deepcopy(params.R)
            # R[0]= params.R[4]
            # R[1]= params.R[3]
            # R[2]= params.R[5]
            # R[4]= params.R[0]
            # R[5]= params.R[2]
            # R[6]= params.R[7]
            # R[7]= params.R[6]

            # R[1]= params.R[3]
            # R[3]= params.R[1]
            # R[2]= params.R[6]
            # R[6]= params.R[2]
            # R[5]= params.R[7]
            # R[7]= params.R[5]

            # R[1] = -R[1]
            # R[3] = -R[3]

            if testangle:
                a = -norm_plane[0] / norm_plane[2]
                b = -norm_plane[1] / norm_plane[2]
                if test_refraction:
                    atmp = a / params.refraction_index
                    btmp = b / params.refraction_index

                    R[0] += atmp * R[2]
                    R[1] += btmp * R[2]
                    R[3] += atmp * R[5]
                    R[4] += btmp * R[5]
                    R[6] += atmp * R[8]
                    R[7] += btmp * R[8]

            Tx = params.T[0]
            Ty = params.T[1]
            Tz = params.T[2]
            Dx = R[4] * R[6] - R[3] * R[7]
            Dy = R[0] * R[7] - R[1] * R[6]
            D0 = R[1] * R[3] - R[0] * R[4]
            Z11 = R[5] * R[7] - R[4] * R[8]
            Z12 = R[1] * R[8] - R[2] * R[7]
            Z21 = R[3] * R[8] - R[5] * R[6]
            Z22 = R[2] * R[6] - R[0] * R[8]
            Zx0 = R[2] * R[4] - R[1] * R[5]
            Zy0 = R[0] * R[5] - R[2] * R[3]

            A11 = R[7] * Ty - R[4] * Tz + Z11 * Z0virt
            A12 = R[1] * Tz - R[7] * Tx + Z12 * Z0virt
            A21 = -R[6] * Ty + R[3] * Tz + Z21 * Z0virt
            A22 = -R[0] * Tz + R[6] * Tx + Z22 * Z0virt

            X0 = R[4] * Tx - R[1] * Ty + Zx0 * Z0virt
            Y0 = -R[3] * Tx + R[0] * Ty + Zy0 * Z0virt

            Xd = (X - params.C[0]) / params.f[0]  # sensor coordinates
            Yd = (Y - params.C[1]) / params.f[1]
            dist_fact = 1 + params.kc * (Xd * Xd + Yd * Yd)
            Xu = Xd / dist_fact  # undistorted sensor coordinates
            Yu = Yd / dist_fact
            denom = Dx * Xu + Dy * Yu + D0
            Xphys = (A11 * Xu + A12 * Yu + X0) / denom  # world coordinates
            Yphys = (A21 * Xu + A22 * Yu + Y0) / denom
            if testangle:
                Zphys = Z0 + a * Xphys + b * Yphys
            else:
                Zphys = Z0
        else:
            Xphys = -params.T[0] + X / params.f[0]
            Yphys = -params.T[1] + Y / params.f[1]
            Zphys = X * 0
        Xphys /= 100  # cm to m
        Yphys /= 100  # cm to m
        Zphys /= 100  # cm to m

        return Xphys, Yphys, Zphys

    def phys2pix(self, Xphys, Yphys, Zphys=0):
        params = copy.deepcopy(self.params)
        Xphys *= 100  # m to cm
        Yphys *= 100  # m to cm
        Zphys *= 100  # m to cm

        if hasattr(params, "f") is False:
            params.f = np.asarray([1, 1])
        if hasattr(params, "T") is False:
            params.T = np.asarray([0, 0, 1])

        # general case
        if hasattr(params, "R"):
            R = params.R
            if hasattr(params, "interface_coord") and hasattr(
                params, "refraction_index"
            ):
                H = params.interface_coord[2]
                if H > Zphys:
                    Zphys = H - (H - Zphys) / params.refraction_index
            # corrected z (virtual object)

            # camera coordinates
            xc = R[0] * Xphys + R[1] * Yphys + R[2] * Zphys + params.T[0]
            yc = R[3] * Xphys + R[4] * Yphys + R[5] * Zphys + params.T[1]
            zc = R[6] * Xphys + R[7] * Yphys + R[8] * Zphys + params.T[2]

            # undistorted image coordinates
            Xu = xc / zc
            Yu = yc / zc

            # radial quadratic correction factor
            if hasattr(params, "kc") is False:
                r2 = 1  # no quadratic distortion
            else:
                r2 = 1 + params.kc * (Xu * Xu + Yu * Yu)

            # pixel coordinates
            if hasattr(params, "C") is False:
                params.C = np.asarray([0, 0])  # default value

            X = params.f[0] * Xu * r2 + params.C[0]
            Y = params.f[1] * Yu * r2 + params.C[1]

        # case 'rescale'
        else:
            X = params.f[0] * (Xphys + params.T[0])
            Y = params.f[1] * (Yphys + params.T[1])

        return X, Y
