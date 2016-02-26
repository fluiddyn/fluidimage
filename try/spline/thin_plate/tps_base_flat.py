"""Thin plate spline
====================

Translated and adapted from UVmat code (Joel.Sommeria - Joel.Sommeria
(A) legi.cnrs.fr)

This interpolation/smoothing (ref fasshauer@iit.edu MATH 590 ?
Chapter 19 32)) minimises a linear combination of the squared
curvature and squared difference form the initial data.

Interpolated data can be obtained as the matrix product `dot(U_tps,
EM)` where the matrix `EM` is obtained by the function `compute_tps_matrix`.
The spatial derivatives are obtained as `dot(U_tps, EMDX)` and
`dot(U_tps, EMDY)`, where `EMDX` and `EMDY` are obtained from the
function `compute_tps_matrix_dxy`.

"""
import numpy as np


def compute_tps_coeff(centers, U, smoothing_coef):
    """Calculate the thin plate spline (tps) coefficients


    INPUT:

    centers: nb_dim x N matrix representing the positions of the N centers,
    sources of the tps (nb_dim = space dimension)

    U: N array representing the values of the considered
    scalar measured at the centres `centers`.

    smoothing_coef: smoothing parameter. The result is smoother for larger
    smoothing_coef.

    OUTPUT:

    U_smooth: values of the quantity U at the N centres after smoothing

    U_tps: tps weights of the centres and columns of the linear

    """
    nb_dim, N = centers.shape
    U = np.hstack([U, np.zeros(nb_dim + 1)])
    U = U.reshape([U.size, 1])
    EM = compute_tps_matrix(centers, centers).T
    smoothing_mat = smoothing_coef * np.eye(N, N)
    smoothing_mat = np.hstack([smoothing_mat, np.zeros([N, nb_dim + 1])])
    PM = np.hstack([np.ones([N, 1]), centers.T])
    IM = np.vstack([EM + smoothing_mat,
                    np.hstack([PM.T, np.zeros([nb_dim + 1, nb_dim + 1])])])
    U_tps, r, r2, r3 = np.linalg.lstsq(IM, U)
    U_smooth = np.dot(EM, U_tps)
    return U_smooth.ravel(), U_tps.ravel()


def compute_tps_matrix(dsites, centers):
    """calculate the thin plate spline (tps) interpolation at a set of points

    INPUT:

    dsites: M * nb_dim matrix representing the postions of the M
    'observation' sites, with nb_dim the space dimension

    centers: N * nb_dim matrix representing the postions of the N centers,
    sources of the tps,

    OUTPUT:

    EM: (N+nb_dim) * M matrix representing the contributions at the M sites

    from unit sources located at each of the N centers, + (nb_dim+1) columns
    representing the contribution of the linear gradient part.

    use : U_interp = np.dot(U_tps, EM)

    """
    s, M = dsites.shape
    s2, N = centers.shape
    assert s == s2
    EM = np.zeros([N, M])
    for d in range(s):
        Dsites, Centers = np.meshgrid(dsites[d], centers[d])
        EM = EM + (Dsites - Centers) ** 2

    nb_p = np.where(EM != 0)
    EM[nb_p] = EM[nb_p] * np.log(EM[nb_p]) / 2
    EM = np.vstack([EM, np.ones(M), dsites])
    return EM


def compute_tps_matrices_dxy(dsites, centers):
    """Calculate the derivatives of thin plate spline (tps) interpolation
    at a set of points (limited to the 2D case)

    OUTPUT:

    DMX: M x (N+3) matrix representing the contributions to the X
    derivatives at the M sites from unit sources located at each of
    the N centers, + 3 columns representing the contribution of the
    linear gradient part.

    DMY: idem for Y derivatives

    INPUT:

    dsites: M x nb_dim matrix of interpolation site coordinates
    (nb_dim = space dimension = 2 here)

    centers: N x nb_dim matrix of centre coordinates (initial data)

    """
    s, M = dsites.shape
    s2, N = centers.shape
    assert s == s2
    Dsites, Centers = np.meshgrid(dsites[0], centers[0])
    DX = Dsites - Centers
    Dsites, Centers = np.meshgrid(dsites[1], centers[1])
    DY = Dsites - Centers
    DM = DX * DX + DY * DY
    DM[DM != 0] = np.log(DM[DM != 0]) + 1
    DMX = np.vstack([DX * DM, np.zeros(M), np.ones(M), np.zeros(M)])
    DMY = np.vstack([DY * DM, np.zeros(M), np.zeros(M), np.ones(M)])
    return DMX, DMY
