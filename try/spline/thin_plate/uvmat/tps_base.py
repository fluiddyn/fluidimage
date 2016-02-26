"""Try to directly translate uvmat code on thin plate spline
============================================================

First, we just write false Python, so it won't work at all.


%=======================================================================
% Copyright 2008-2014, LEGI UMR 5519 / CNRS UJF G-INP, Grenoble, France
%   http://www.legi.grenoble-inp.fr
%   Joel.Sommeria - Joel.Sommeria (A) legi.cnrs.fr
%
%     This file is part of the toolbox UVMAT.
%
%     UVMAT is free software; you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published
%     by the Free Software Foundation; either version 2 of the license,
%     or (at your option) any later version.
%
%     UVMAT is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License (see LICENSE.txt) for more details.
%=======================================================================


"""
import numpy as np
from numpy import log


def tps_coeff(centers, U, smoothing_coef):
    """Calculate the thin plate spline (tps) coefficients

    (ref fasshauer@iit.edu MATH 590 ? Chapter 19 32)

    This interpolation/smoothing minimises a linear combination of the
    squared curvature and squared difference form the initial data.

    This function calculates the weight coefficients U_tps of the N
    sites where data are known.

    Interpolated data are then obtained as the matrix product `EM @
    U_tps` where the matrix `EM` is obtained by the function
    `tps_eval`.  The spatial derivatives are obtained as `EMDX @
    U_tps` and `EMDY @ U_tps`, where `EMDX` and `EMDY` are obtained
    from the function `tps_eval_dxy`.  For big data sets, a splitting
    in subdomains is needed, see functions `set_subdomains` and
    `tps_coeff_field`.

    INPUT:

    centers: N x nb_dim matrix representing the positions of the N centers,
    sources of the tps (nb_dim = space dimension)

    U: N x 1 column vector representing the values of the considered
    scalar measured at the centres centers

    smoothing_coef: smoothing parameter. The result is smoother for larger
    smoothing_coef.

    OUTPUT:

    U_smooth: values of the quantity U at the N centres after smoothing

    U_tps: tps weights of the centres and columns of the linear

    RELATED FUNCTIONS:
    tps_eval, tps_eval_dxy
    tps_coeff_field, set_subdomains, filter_tps, calc_field

    """
    N, nb_dim = centers.shape
    U = np.vstack([U, np.zeros([nb_dim + 1, 1])])
    EM = tps_eval(centers, centers)
    smoothing_mat = smoothing_coef * np.eye(N, N)
    smoothing_mat = np.hstack([smoothing_mat, np.zeros([N, nb_dim + 1])])
    PM = np.hstack([np.ones([N, 1]), centers])
    IM = np.vstack([EM + smoothing_mat,
                    np.hstack([PM.T, np.zeros([nb_dim + 1, nb_dim + 1])])])
    U_tps, r, r2, r3 = np.linalg.lstsq(IM, U)
    U_smooth = np.dot(EM, U_tps)
    return U_smooth, U_tps


def tps_eval(dsites, centers):
    """calculate the thin plate spline (tps) interpolation at a set of points

    INPUT:

    dsites: M * s matrix representing the postions of the M
    'observation' sites, with s the space dimension

    centers: N * s matrix representing the postions of the N centers,
    sources of the tps,


    OUTPUT:

    EM: M * (N+s) matrix representing the contributions at the M sites

    from unit sources located at each of the N centers, + (s+1) columns
    representing the contribution of the linear gradient part.

    use : U_interp = EM * U_tps


    RELATED FUNCTIONS:
    tps_coeff, tps_eval_dxy
    tps_coeff_field, set_subdomains, filter_tps, calc_field

    """
    M, s = dsites.shape
    N, s2 = centers.shape
    assert s == s2
    EM = np.zeros([M, N])
    for d in range(s):
        Dsites, Centers = np.meshgrid(
            dsites[:, d], centers[:, d], indexing='ij')
        EM = EM + (Dsites - Centers) ** 2

    nb_p = np.where(EM != 0)
    EM[nb_p] = EM[nb_p] * log(EM[nb_p]) / 2
    EM = np.hstack([EM, np.ones([M, 1]), dsites])
    return EM


def tps_eval_T(dsites, centers):
    """calculate the thin plate spline (tps) interpolation at a set of points

    INPUT:

    dsites: M * s matrix representing the postions of the M
    'observation' sites, with s the space dimension

    centers: N * s matrix representing the postions of the N centers,
    sources of the tps,


    OUTPUT:

    EM: M * (N+s) matrix representing the contributions at the M sites

    from unit sources located at each of the N centers, + (s+1) columns
    representing the contribution of the linear gradient part.

    use : U_interp = EM * U_tps


    RELATED FUNCTIONS:
    tps_coeff, tps_eval_dxy
    tps_coeff_field, set_subdomains, filter_tps, calc_field

    """
    M, s = dsites.shape
    N, s2 = centers.shape
    assert s == s2
    EM = np.zeros([N, M])
    for d in range(s):
        Dsites, Centers = np.meshgrid(
            dsites[:, d], centers[:, d])
        EM = EM + (Dsites - Centers) ** 2

    nb_p = np.where(EM != 0)
    EM[nb_p] = EM[nb_p] * log(EM[nb_p]) / 2
    EM = np.vstack([EM, np.ones([M]), dsites.T])
    return EM



def tps_eval_dxy(dsites, centers):
    """Calculate the derivatives of thin plate spline (tps) interpolation
    at a set of points (limited to the 2D case)

    OUTPUT:

    DMX: Mx(N+3) matrix representing the contributions to the X
    derivatives at the M sites from unit sources located at each of
    the N centers, + 3 columns representing the contribution of the
    linear gradient part.

    DMY: idem for Y derivatives

    INPUT:

    dsites: M x s matrix of interpolation site coordinates (s=space
    dimension=2 here)

    centers: N x s matrix of centre coordinates (initial data)

    RELATED FUNCTIONS:
    tps_coeff, tps_eval
    tps_coeff_field, set_subdomains, filter_tps, calc_field

    """
    M, s = dsites.shape
    N, s2 = centers.shape
    assert s == s2
    Dsites, Centers = np.meshgrid(dsites[:, 0], centers[:, 0], indexing='ij')
    DX = Dsites - Centers
    Dsites, Centers = np.meshgrid(dsites[:, 1], centers[:, 1], indexing='ij')
    DY = Dsites - Centers
    DM = DX * DX + DY * DY
    DM[DM != 0] = log(DM[DM != 0]) + 1
    DMX = np.hstack([
        DX * DM,
        np.zeros([M, 1]),
        np.ones([M, 1]),
        np.zeros([M, 1])])
    DMY = np.hstack([
        DY * DM,
        np.zeros([M, 1]),
        np.zeros([M, 1]),
        np.ones([M, 1])])
    return (DMX, DMY)


if __name__ == '__main__':
    x = 2 * np.pi * np.random.rand(100)
    y = 2 * np.pi * np.random.rand(100)
    x = x.reshape([x.size, 1])
    y = y.reshape([y.size, 1])
    centers = np.hstack([x, y])
    EM = tps_eval(centers, centers)
    U = np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2)
    smoothing_coef = 0
    U_smooth, U_tps = tps_coeff(centers, U, smoothing_coef)
