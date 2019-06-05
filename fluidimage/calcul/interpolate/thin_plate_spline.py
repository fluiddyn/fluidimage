"""Thin plate spline
====================

Translated and adapted from UVmat code (Joel Sommeria, LEGI, CNRS).

This interpolation/smoothing (Duchon, 1976; NguyenDuc and Sommeria,
1988) minimises a linear combination of the squared curvature and
squared difference from the initial data.

We first need to compute tps coefficients ``U_tps`` (function
``compute_tps_coeff``). Interpolated data can then be obtained as the
matrix product ``dot(U_tps, EM)`` where the matrix ``EM`` is obtained
by the function ``compute_tps_matrix``.  The spatial derivatives are
obtained as ``dot(U_tps, EMDX)`` and ``dot(U_tps, EMDY)``, where
``EMDX`` and ``EMDY`` are obtained from the function
``compute_tps_matrix_dxy``. A helper class is also provided.

.. autofunction:: compute_tps_coeff

.. autoclass:: ThinPlateSpline
   :members:

.. autofunction:: compute_tps_matrix_numpy

.. autofunction:: compute_tps_matrices_dxy


"""

import numpy as np

from transonic import Transonic, boost

ts = Transonic()

A = "float64[][]"


@boost
def compute_tps_matrix_pythran(new_pos: A, centers: A):
    """calculate the thin plate spline (tps) interpolation at a set of points

    Parameters
    ----------

    new_pos: np.array
        ``[nb_dim, M]`` array representing the postions of the M
        'observation' sites, with nb_dim the space dimension.

    centers: np.array
        ``[nb_dim, N]`` array representing the postions of the N centers,
        sources of the tps.

    Returns
    -------

    EM : np.array
        ``[(N+nb_dim), M]`` matrix representing the contributions at the M sites.

        From unit sources located at each of the N centers, +
        (nb_dim+1) columns representing the contribution of the linear
        gradient part.

    Notes
    -----

    >>> U_interp = np.dot(U_tps, EM)

    """

    d, nb_new_pos = new_pos.shape
    d2, nb_centers = centers.shape
    assert d == d2

    EM = np.zeros((nb_centers, nb_new_pos))

    # # pythran 0.9.2 does not know np.meshgrid
    # for ind_d in range(s):
    #     Dsites, Centers = np.meshgrid(dsites[ind_d], centers[ind_d])
    #     EM += (Dsites - Centers)**2

    for ind_d in range(d):
        for ic, center in enumerate(centers[ind_d]):
            for inp, npos in enumerate(new_pos[ind_d]):
                EM[ic, inp] += (npos - center) ** 2

    nb_p = np.where(EM != 0)
    EM[nb_p] = EM[nb_p] * np.log(EM[nb_p]) / 2
    EM_ret = np.vstack([EM, np.ones(nb_new_pos), new_pos])
    return EM_ret


def compute_tps_matrix_numpy(new_pos: A, centers: A):
    """calculate the thin plate spline (tps) interpolation at a set of points

    Parameters
    ----------

    new_pos: np.array
        ``[nb_dim, M]`` array representing the postions of the M
        'observation' sites, with nb_dim the space dimension.

    centers: np.array
        ``[nb_dim, N]`` array representing the postions of the N centers,
        sources of the tps.

    Returns
    -------

    EM : np.array

        ``[(N+nb_dim), M]`` matrix representing the contributions at the M
        sites.

        From unit sources located at each of the N centers, +
        (nb_dim+1) columns representing the contribution of the linear
        gradient part.

    Notes
    -----

    >>> U_interp = np.dot(U_tps, EM)

    """

    d, nb_new_pos = new_pos.shape
    d2, nb_centers = centers.shape
    assert d == d2

    EM = np.zeros((nb_centers, nb_new_pos))

    for ind_d in range(d):
        Dsites, Centers = np.meshgrid(new_pos[ind_d], centers[ind_d])
        EM += (Dsites - Centers) ** 2

    nb_p = np.where(EM != 0)
    EM[nb_p] = EM[nb_p] * np.log(EM[nb_p]) / 2
    EM_ret = np.vstack([EM, np.ones(nb_new_pos), new_pos])
    return EM_ret


if ts.is_compiled:

    def compute_tps_matrix(newcenters, centers):
        return compute_tps_matrix_pythran(
            newcenters.astype(np.float64), centers.astype(np.float64)
        )


else:
    print("Warning: function compute_tps_matrix_numpy not pythranized.")
    compute_tps_matrix = compute_tps_matrix_numpy


def compute_tps_coeff(centers, U, smoothing_coef):
    """Calculate the thin plate spline (tps) coefficients

    Parameters
    ----------

    centers : np.array
        ``[nb_dim,  N]`` array representing the positions of the N centers,
        sources of the TPS (nb_dim = space dimension).

    U : np.array
        ``[N]`` array representing the values of the considered
        scalar measured at the centres ``centers``.

    smoothing_coef : float
        Smoothing parameter. The result is smoother for larger smoothing_coef.

    Returns
    -------

    U_smooth : np.array
         Values of the quantity U at the N centres after smoothing.

    U_tps : np.array
         TPS weights of the centres and columns of the linear.

    """
    nb_dim, N = centers.shape
    U = np.hstack([U, np.zeros(nb_dim + 1)])
    U = U.reshape([U.size, 1])
    try:
        EM = compute_tps_matrix(centers, centers).T
    except TypeError as e:
        print(centers.dtype, centers.shape)
        raise e

    smoothing_mat = smoothing_coef * np.eye(N, N)
    smoothing_mat = np.hstack([smoothing_mat, np.zeros([N, nb_dim + 1])])
    PM = np.hstack([np.ones([N, 1]), centers.T])
    IM = np.vstack(
        [
            EM + smoothing_mat,
            np.hstack([PM.T, np.zeros([nb_dim + 1, nb_dim + 1])]),
        ]
    )

    # print('det(IM)', np.linalg.det(IM))
    # print('cond(IM)', np.linalg.cond(IM))

    # U_tps, r, r2, r3 = np.linalg.lstsq(IM, U)
    U_tps = np.linalg.solve(IM, U)

    U_smooth = np.dot(EM, U_tps)
    return U_smooth.ravel(), U_tps.ravel()


def compute_tps_matrices_dxy(dsites, centers):
    """Calculate the derivatives of thin plate spline (tps) interpolation
    at a set of points (limited to the 2D case)

    Parameters
    ----------

    dsites : np.array
        ``[nb_dim,  M]`` array of interpolation site coordinates
        (nb_dim = space dimension = 2, here).

    centers : np.array
        ``[nb_dim,  N]`` array of centre coordinates (initial data).

    Returns
    -------

    DMX : np.array
        ``[(N+3),  M]`` array representing the contributions to the X
        derivatives at the M sites from unit sources located at each
        of the N centers, + 3 columns representing the contribution of
        the linear gradient part.

    DMY : np.array
        idem for Y derivatives.

    """
    s, M = dsites.shape
    s2, N = centers.shape
    assert s == s2
    Dsites, Centers = np.meshgrid(dsites[1], centers[1])
    DX = Dsites - Centers
    Dsites, Centers = np.meshgrid(dsites[0], centers[0])
    DY = Dsites - Centers
    DM = DX * DX + DY * DY
    DM[DM != 0] = np.log(DM[DM != 0]) + 1
    DMX = np.vstack([DX * DM, np.zeros(M), np.ones(M), np.zeros(M)])
    DMY = np.vstack([DY * DM, np.zeros(M), np.zeros(M), np.ones(M)])
    return DMX, DMY


class ThinPlateSpline:
    """Helper class for thin plate interpolation."""

    _compute_tps_matrix = compute_tps_matrix

    def __init__(self, new_positions, centers):
        self.EM = type(self)._compute_tps_matrix(new_positions, centers)
        self.DMX, self.DMY = compute_tps_matrices_dxy(new_positions, centers)

    def compute_field(self, U_tps):
        """Compute the interpolated field."""
        return np.dot(U_tps, self.EM)

    def compute_gradient(self, U_tps):
        """Compute the gradient (dx_U, dy_U)"""
        return np.dot(U_tps, self.DMX), np.dot(U_tps, self.DMY)


class ThinPlateSplineNumpy(ThinPlateSpline):
    pass
    # _compute_tps_matrix = compute_tps_matrix_numpy
