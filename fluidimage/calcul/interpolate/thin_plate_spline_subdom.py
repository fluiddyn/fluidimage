"""Thin plate spline with subdomains
====================================

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

.. autoclass:: ThinPlateSplineSubdom
   :members:

"""

from logging import debug

import numpy as np

from .thin_plate_spline import compute_tps_coeff, compute_tps_matrix


class ThinPlateSplineSubdom:
    """Helper class for thin plate interpolation."""

    def __init__(
        self,
        centers,
        subdom_size,
        smoothing_coef,
        threshold=None,
        percent_buffer_area=0.2,
    ):

        self.centers = centers
        self.subdom_size = subdom_size
        self.smoothing_coef = smoothing_coef
        self.threshold = threshold
        self.compute_indices(percent_buffer_area)

    def compute_indices(self, percent_buffer_area=0.25):
        xs = self.centers[1]
        ys = self.centers[0]
        max_coord = np.max(self.centers, 1)
        min_coord = np.min(self.centers, 1)
        range_coord = max_coord - min_coord
        aspect_ratio = range_coord[0] / range_coord[1]

        nb_subdom = xs.size / self.subdom_size
        nb_subdomx = int(np.floor(np.sqrt(nb_subdom / aspect_ratio)))
        nb_subdomx = nb_subdomx or 1
        nb_subdomy = int(np.ceil(np.sqrt(nb_subdom * aspect_ratio)))
        nb_subdomy = nb_subdomy or 1

        debug(f"nb_subdomx: {nb_subdomx} ; nb_subdomy: {nb_subdomy}")

        nb_subdom = nb_subdomx * nb_subdomy

        self.nb_subdomx = nb_subdomx
        self.nb_subdomy = nb_subdomy

        x_dom = np.linspace(min_coord[1], max_coord[1], nb_subdomx + 1)
        y_dom = np.linspace(min_coord[0], max_coord[0], nb_subdomy + 1)

        buffer_area_x = (
            range_coord[1]
            / (nb_subdomx)
            * percent_buffer_area
            * np.ones_like(x_dom)
        )
        buffer_area_y = (
            range_coord[0]
            / (nb_subdomy)
            * percent_buffer_area
            * np.ones_like(y_dom)
        )

        self.x_dom = x_dom
        self.y_dom = y_dom
        self.buffer_area_x = buffer_area_x
        self.buffer_area_y = buffer_area_y

        ind_subdom = np.zeros([nb_subdom, 2])
        ind_v_subdom = []

        i_subdom = 0
        for i in range(nb_subdomx):
            for j in range(nb_subdomy):
                ind_subdom[i_subdom, :] = [i, j]

                ind_v_subdom.append(
                    np.where(
                        (xs >= x_dom[i] - buffer_area_x[i])
                        & (xs < x_dom[i + 1] + buffer_area_x[i + 1])
                        & (ys >= y_dom[j] - buffer_area_y[j])
                        & (ys < y_dom[j + 1] + buffer_area_y[j + 1])
                    )[0]
                )

                i_subdom += 1
        self.ind_v_subdom = ind_v_subdom
        self.nb_subdom = nb_subdom

    def compute_tps_coeff_subdom(self, U):

        U_smooth = [None] * self.nb_subdom
        U_tps = [None] * self.nb_subdom

        for i in range(self.nb_subdom):

            centers_tmp = self.centers[:, self.ind_v_subdom[i]]

            U_tmp = U[self.ind_v_subdom[i]]
            U_smooth[i], U_tps[i] = self.compute_tps_coeff_iter(
                centers_tmp, U_tmp
            )

        U_smooth_tmp = np.zeros(self.centers[1].shape)
        nb_tps = np.zeros(self.centers[1].shape, dtype=int)

        for i in range(self.nb_subdom):
            U_smooth_tmp[self.ind_v_subdom[i]] += U_smooth[i]
            nb_tps[self.ind_v_subdom[i]] += 1

        U_smooth_tmp /= nb_tps

        return U_smooth_tmp, U_tps

    def init_with_new_positions(self, new_positions):
        npos = self.new_positions = new_positions

        ind_new_positions_subdom = []

        x_dom = self.x_dom
        y_dom = self.y_dom
        buffer_area_x = self.buffer_area_x
        buffer_area_y = self.buffer_area_y

        i_subdom = 0
        for i in range(self.nb_subdomx):
            for j in range(self.nb_subdomy):
                ind_new_positions_subdom.append(
                    np.where(
                        (npos[1] >= x_dom[i] - buffer_area_x[i])
                        & (npos[1] < x_dom[i + 1] + buffer_area_x[i + 1])
                        & (npos[0] >= y_dom[j] - buffer_area_y[j])
                        & (npos[0] < y_dom[j + 1] + buffer_area_y[j + 1])
                    )[0]
                )

                i_subdom += 1

        self.ind_new_positions_subdom = ind_new_positions_subdom
        self._init_EM_subdom()

    def _init_EM_subdom(self):

        EM = [None] * self.nb_subdom

        for i in range(self.nb_subdom):
            centers_tmp = self.centers[:, self.ind_v_subdom[i]]
            new_positions_tmp = self.new_positions[
                :, self.ind_new_positions_subdom[i]
            ]
            EM[i] = compute_tps_matrix(new_positions_tmp, centers_tmp)

        self.EM = EM

    def compute_eval(self, U_tps):

        U_eval = np.zeros(self.new_positions[1].shape)
        nb_tps = np.zeros(self.new_positions[1].shape, dtype=int)

        for i in range(self.nb_subdom):
            U_eval[self.ind_new_positions_subdom[i]] += np.dot(
                U_tps[i], self.EM[i]
            )
            nb_tps[self.ind_new_positions_subdom[i]] += 1

        U_eval /= nb_tps

        return U_eval

    def compute_tps_coeff_iter(self, centers, U):
        """Compute the thin plate spline (tps) coefficients removing erratic
        vectors

        It computes iteratively "compute_tps_coeff", compares the tps
        result to the initial data and remove it if difference is
        larger than the given threshold

        """
        U_smooth, U_tps = compute_tps_coeff(centers, U, self.smoothing_coef)
        if self.threshold is not None:
            Udiff = np.sqrt((U_smooth - U) ** 2)
            ind_erratic_vector = np.argwhere(Udiff > self.threshold)

            count = 1
            while ind_erratic_vector.size != 0:
                U[ind_erratic_vector] = U_smooth[ind_erratic_vector]
                U_smooth, U_tps = compute_tps_coeff(
                    centers, U, self.smoothing_coef
                )

                Udiff = np.sqrt((U_smooth - U) ** 2)
                ind_erratic_vector = np.argwhere(Udiff > self.threshold)
                count += 1

                if count > 10:
                    print("tps stopped after 10 iterations")
                    break

        if count > 1:
            print("tps done after ", count, " attempt(s)")
        return U_smooth, U_tps
