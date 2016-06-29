# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:56:57 2016

@author: campagne8a
"""
import numpy as np


def pix2phys_UV(calib, X, Y, dx, dy, Zindex):
    """
        apply Tsai Calibration to the field

        Be careful:
        The displacement is 3 component BUT
        it corresponds to the real 3 component displacement projected on the
        corresponding plane indexed by Zindex.
    """
    Xphys, Yphys, Zphys = pix2phys(calib, X, Y, Zindex)

    dxb, dyb, dzb = pix2phys(calib, X + dx/2.0, Y + dy/2.0, Zindex)
    dxa, dya, dza = pix2phys(calib, X - dx/2.0, Y - dy/2.0, Zindex)
    dxphys = dxb - dxa
    dyphys = dyb - dya
    dzphys = dzb - dza
    return Xphys, Yphys, Zphys, dxphys, dyphys, dzphys


def pix2phys(calib, X, Y, Zindex):

    # determine position of Z0
    testangle = 0
    if hasattr(calib.slices, 'sliceAngle') and \
       np.any(calib.slices.sliceAngle[Zindex] != np.asarray([0, 0, 0])):
        testangle = 1
        om = np.linalg.norm(calib.slices.sliceAngle[Zindex])
        axis_rot = calib.slices.sliceAngle[Zindex] / om
        cos_om = np.cos(np.pi*om/180.0)
        sin_om = np.sin(np.pi*om/180.0)
        coeff = axis_rot[2]*(1-cos_om)
        norm_plane = np.zeros(3)
        norm_plane[0] = axis_rot[0]*coeff+axis_rot[1]*sin_om
        norm_plane[1] = axis_rot[1]*coeff-axis_rot[0]*sin_om
        norm_plane[2] = axis_rot[2]*coeff+cos_om
        Z0 = np.dot(norm_plane,
                    (calib.slices.zsliceCoord[Zindex]))/norm_plane[2]
    else:
        Z0 = calib.slices.zsliceCoord[Zindex][2]
    Z0virt = Z0
    if hasattr(calib,'interface_coord') and hasattr(calib,'refraction_index'):
        H=calib.interface_coord[2]
        if H>Z0:
            Z0virt=H-(H-Z0)/calib.refraction_index #corrected z (virtual object)
            test_refraction=1;

    
    if hasattr(calib, 'f') is False:
        calib.f = np.asarray([1, 1])
    if hasattr(calib, 'T') is False:
        calib.T = np.asarray([0, 0, 1])
    if hasattr(calib, 'C') is False:
        calib.C = np.asarray([0, 0, 1])
    if hasattr(calib, 'kc') is False:
        calib.kc = 0

    if hasattr(calib, 'R'):
        R = calib.R

        if testangle:
            a = -norm_plane[0]/norm_plane[2]
            b = -norm_plane[1]/norm_plane[2]
            if test_refraction:
                a /= calib.refraction_index;
                b /= calib.refraction_index;

            R[0] = R[0]+a*R[2]
            R[1] = R[1]+b*R[2]
            R[3] = R[3]+a*R[5]
            R[4] = R[4]+b*R[5]
            R[6] = R[6]+a*R[8]
            R[7] = R[7]+b*R[8]

        Tx = calib.T[0]
        Ty = calib.T[1]
        Tz = calib.T[2]
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

        #     X0=Calib.fx_fy(1)*(R(5)*Tx-R(2)*Ty+Zx0*Z0virt)
        #     Y0=Calib.fx_fy(2)*(-R(4)*Tx+R(1)*Ty+Zy0*Z0virt)
        X0 = (R[4]*Tx-R[1]*Ty+Zx0*Z0virt)
        Y0 = (-R[3]*Tx+R[0]*Ty+Zy0*Z0virt)
        # px to camera:
        #     Xd=dpx*(X-Calib.Cx_Cy(1)) % sensor coordinates
        #     Yd=(Y-Calib.Cx_Cy(2))
        Xd = (X-calib.C[0])/calib.f[0]  # sensor coordinates
        Yd = (Y-calib.C[1])/calib.f[1]
        dist_fact = 1 + calib.kc*(Xd*Xd+Yd*Yd)  # /(f*f) %distortion factor
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
        Xphys = -calib.T[0]+X/calib.f[0]
        Yphys = -calib.T[1]+Y/calib.f[1]
        Zphys = X*0
    Xphys /= 100  # cm to m
    Yphys /= 100  # cm to m
    Zphys /= 100  # cm to m

    return Xphys, Yphys, Zphys


def phys2pix(calib, Xphys, Yphys, Zphys=0):
    Xphys *= 100  # m to cm
    Yphys *= 100  # m to cm
    Zphys *= 100  # m to cm

    if hasattr(calib, 'f') is False:
        calib.f = np.asarray([1, 1])
    if hasattr(calib, 'T') is False:
        calib.T = np.asarray([0, 0, 1])

    # general case
    if hasattr(calib, 'R'):
        R = calib.R
        if hasattr(calib,'interface_coord') and hasattr(calib,'refraction_index'):
            H=calib.interface_coord[2];
            if H>Zphys:
                Zphys=H-(H-Zphys)/calib.refraction_index; #%corrected z (virtual object)
                
        #%camera coordinates
        xc=R[0]*Xphys+R[1]*Yphys+R[2]*Zphys+calib.T[0]
        yc=R[3]*Xphys+R[4]*Yphys+R[5]*Zphys+calib.T[1]
        zc=R[6]*Xphys+R[7]*Yphys+R[8]*Zphys+calib.T[2]
    
        #%undistorted image coordinates
        Xu=xc/zc;
        Yu=yc/zc;
    
        #%radial quadratic correction factor
        if hasattr(calib, 'kc') is False:
            r2 = 1  # no quadratic distortion
        else:
            r2 = 1 + calib.kc*(Xu*Xu+Yu*Yu)

        # pixel coordinates
        if hasattr(calib, 'C') is False:
            calib.C = np.asarray([0, 0])  # default value

        X = calib.f[0]*Xu*r2 + calib.C[0]
        Y = calib.f[1]*Yu*r2 + calib.C[1]

    # case 'rescale'
    else:
        X = calib.f[0]*(Xphys+calib.T[0])
        Y = calib.f[1]*(Yphys+calib.T[1])

    return X, Y
 
    
