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
from __future__ import print_function

import numpy as np

from . import tps_pythran


def compute_tps_coeff_subdom(
    centers,
    U,
    smoothing_coef,
    subdom_size,
    new_positions,
    threshold=None,
    percent_buffer_area=0.25,
):

    max_coord = np.max(centers, 1)
    min_coord = np.min(centers, 1)
    range_coord = max_coord - min_coord
    aspect_ratio = range_coord[0] / range_coord[1]

    nb_subdom = centers[1, :].size / subdom_size
    nb_subdomx = int(np.max(np.floor(np.sqrt(nb_subdom / aspect_ratio)), 0))
    nb_subdomy = int(np.max(np.floor(np.sqrt(nb_subdom * aspect_ratio)), 0))
    nb_subdom = nb_subdomx * nb_subdomy

    x_dom = np.arange(min_coord[1], max_coord[1], range_coord[1] / nb_subdomx)
    x_dom = np.unique(np.append(x_dom, max_coord[1]))
    y_dom = np.arange(min_coord[0], max_coord[0], range_coord[0] / nb_subdomy)
    y_dom = np.unique(np.append(y_dom, max_coord[0]))

    buffer_area_x = (
        x_dom * 0 + range_coord[1] / (nb_subdomx) * percent_buffer_area
    )
    # buffer_area_x[0], buffer_area_x[-1] = 0, 0
    buffer_area_y = (
        y_dom * 0 + range_coord[0] / (nb_subdomy) * percent_buffer_area
    )
    # buffer_area_y[0], buffer_area_y[-1] = 0, 0

    ind_subdom = np.zeros([nb_subdom, 2])
    ind_v_subdom = []
    ind_new_positions_subdom = []

    count = 0
    for i in range(nb_subdomy):
        for j in range(nb_subdomx):
            ind_subdom[count, :] = [i, j]

            ind_v_subdom.append(
                np.argwhere(
                    (centers[1, :] >= x_dom[j] - buffer_area_x[j])
                    & (centers[1, :] < x_dom[j + 1] + buffer_area_x[j + 1])
                    & (centers[0, :] >= y_dom[i] - buffer_area_y[i])
                    & (centers[0, :] < y_dom[i + 1] + buffer_area_y[i + 1])
                ).flatten()
            )

            ind_new_positions_subdom.append(
                np.argwhere(
                    (new_positions[1, :] >= x_dom[j] - buffer_area_x[j])
                    & (new_positions[1, :] < x_dom[j + 1] + buffer_area_x[j + 1])
                    & (new_positions[0, :] >= y_dom[i] - buffer_area_y[i])
                    & (new_positions[0, :] < y_dom[i + 1] + buffer_area_y[i + 1])
                ).flatten()
            )

            count += 1

    U_eval = np.zeros(new_positions[1].shape)
    nb_tps = np.zeros(new_positions[1].shape)

    U_tps = [None] * nb_subdom
    U_smooth = [None] * nb_subdom

    for i in range(nb_subdom):
        centerstemp = np.vstack(
            [centers[1][ind_v_subdom[i]], centers[0][ind_v_subdom[i]]]
        )
        Utemp = U[ind_v_subdom[i]]
        U_smooth[i], U_tps[i] = compute_tps_coeff_iter(
            centerstemp, Utemp, smoothing_coef, threshold
        )

        centers_newposition_temp = np.vstack(
            [
                new_positions[1][ind_new_positions_subdom[i]],
                new_positions[0][ind_new_positions_subdom[i]],
            ]
        )

        EM = compute_tps_matrix(centers_newposition_temp, centerstemp)

        U_eval[ind_new_positions_subdom[i]] = np.dot(U_tps[i], EM)
        nb_tps[ind_new_positions_subdom[i]] += 1.0

    U_eval /= nb_tps

    return U_smooth, U_tps, x_dom, y_dom, buffer_area_x, buffer_area_y


def compute_tps_coeff_iter(centers, U, smoothing_coef, threshold=None):
    """ Compute the thin plate spline (tps) coefficients removing erratic
    vectors
    It computes iteratively "compute_tps_coeff", compares the tps result
    to the initial data and remove it if difference is larger than the given
    threshold

    """
    U_smooth, U_tps = compute_tps_coeff(centers, U, smoothing_coef)

    if threshold is not None:
        Udiff = np.sqrt((U_smooth - U) ** 2)
        ind_erratic_vector = np.argwhere(Udiff > threshold)

        count = 1
        while ind_erratic_vector.size != 0:
            U[ind_erratic_vector] = U_smooth[ind_erratic_vector]
            U_smooth, U_tps = compute_tps_coeff(centers, U, smoothing_coef)

            Udiff = np.sqrt((U_smooth - U) ** 2)
            ind_erratic_vector = np.argwhere(Udiff > threshold)
            count += 1

            if count > 10:
                print("tps stopped after 10 iterations")
                break

    if count > 1:
        print("tps done after ", count, " attempt(s)")
    return U_smooth, U_tps


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


def compute_tps_matrix_numpy(dsites, centers):
    """calculate the thin plate spline (tps) interpolation at a set of points

    Parameters
    ----------

    dsites: np.array
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
    s, M = dsites.shape
    s2, N = centers.shape
    assert s == s2
    EM = np.zeros([N, M])
    for d in range(s):
        Dsites, Centers = np.meshgrid(dsites[d], centers[d])
        EM += (Dsites - Centers) ** 2

    nb_p = np.where(EM != 0)
    EM[nb_p] = EM[nb_p] * np.log(EM[nb_p]) / 2
    EM = np.vstack([EM, np.ones(M), dsites])
    return EM


if hasattr(tps_pythran, "__pythran__"):

    def compute_tps_matrix(newcenters, centers):
        return tps_pythran.compute_tps_matrix(
            newcenters.astype(np.float64), centers.astype(np.float64)
        )


else:
    print("Warning: function compute_tps_matrix_numpy not pythranized.")
    compute_tps_matrix = compute_tps_matrix_numpy


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


class ThinPlateSpline(object):
    """Helper class for thin plate interpolation."""

    def __init__(self, new_positions, centers):
        self.EM = compute_tps_matrix(new_positions, centers)
        self.DMX, self.DMY = compute_tps_matrices_dxy(new_positions, centers)

    def compute_field(self, U_tps):
        """Compute the interpolated field."""
        return np.dot(U_tps, self.EM)

    def compute_gradient(self, U_tps):
        """Compute the gradient (dx_U, dy_U)"""
        return np.dot(U_tps, self.DMX), np.dot(U_tps, self.DMY)
