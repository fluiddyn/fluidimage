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
from typing import List

import numpy as np
from transonic import Array

from .thin_plate_spline import compute_tps_coeff, compute_tps_matrix


class ThinPlateSplineSubdom:
    """Helper class for thin plate interpolation."""

    new_positions: "np.int64[:,:]"
    ind_new_positions_subdom: List["np.int64[:]"]
    norm_coefs: "float[:]"

    def __init__(
        self,
        centers,
        subdom_size,
        smoothing_coef,
        threshold=None,
        percent_buffer_area=20,
    ):
        self.centers = centers
        self.subdom_size = subdom_size
        self.smoothing_coef = smoothing_coef
        self.threshold = threshold
        self.compute_indices(percent_buffer_area)

    def compute_indices(self, percent_buffer_area):
        # note: `centers = np.vstack([ys, xs])`
        xs = self.centers[1]
        ys = self.centers[0]
        x_min = xs.min()
        y_min = ys.min()
        x_max = xs.max()
        y_max = ys.max()
        range_x = x_max - x_min
        range_y = y_max - y_min
        aspect_ratio = range_y / range_x

        nb_subdom = xs.size / self.subdom_size
        nb_subdomx = int(np.floor(np.sqrt(nb_subdom / aspect_ratio)))
        nb_subdomx = nb_subdomx or 1
        nb_subdomy = int(np.ceil(np.sqrt(nb_subdom * aspect_ratio)))
        nb_subdomy = nb_subdomy or 1

        debug(f"nb_subdomx: {nb_subdomx} ; nb_subdomy: {nb_subdomy}")

        nb_subdom = nb_subdomx * nb_subdomy

        self.nb_subdomx = nb_subdomx
        self.nb_subdomy = nb_subdomy

        x_dom = np.linspace(x_min, x_max, nb_subdomx + 1)
        y_dom = np.linspace(y_min, y_max, nb_subdomy + 1)

        coef_buffer = percent_buffer_area / 100
        buffer_length_x = coef_buffer * range_x / nb_subdomx
        buffer_length_y = coef_buffer * range_y / nb_subdomy

        self.xmin_limits = x_dom[:-1] - buffer_length_x
        self.xmax_limits = x_dom[1:] + buffer_length_x

        self.ymin_limits = y_dom[:-1] - buffer_length_y
        self.ymax_limits = y_dom[1:] + buffer_length_y

        ind_v_subdom = []

        for iy in range(nb_subdomy):
            for ix in range(nb_subdomx):

                ind_v_subdom.append(
                    np.where(
                        (xs >= self.xmin_limits[ix])
                        & (xs < self.xmax_limits[ix])
                        & (ys >= self.ymin_limits[iy])
                        & (ys < self.ymin_limits[iy])
                    )[0]
                )

        self.ind_v_subdom = ind_v_subdom
        self.nb_subdom = nb_subdom

    def compute_tps_coeff_subdom(self, U):
        U_smooth = [None] * self.nb_subdom
        U_tps = [None] * self.nb_subdom
        summaries = [None] * self.nb_subdom

        for i in range(self.nb_subdom):
            centers_tmp = self.centers[:, self.ind_v_subdom[i]]
            U_tmp = U[self.ind_v_subdom[i]]
            U_smooth[i], U_tps[i], summaries[i] = self.compute_tps_coeff_iter(
                centers_tmp, U_tmp
            )

        U_smooth_tmp = np.zeros(self.centers[1].shape)
        nb_tps = np.zeros(self.centers[1].shape, dtype=int)
        summary = {"nb_fixed_vectors": [], "max(Udiff)": [], "nb_iterations": []}

        for i in range(self.nb_subdom):
            U_smooth_tmp[self.ind_v_subdom[i]] += U_smooth[i]
            nb_tps[self.ind_v_subdom[i]] += 1
            for key in ("nb_fixed_vectors", "max(Udiff)", "nb_iterations"):
                summary[key].append(summaries[i][key])

        summary["nb_fixed_vectors_tot"] = sum(summary["nb_fixed_vectors"])
        U_smooth_tmp /= nb_tps

        return U_smooth_tmp, U_tps, summary

    def init_with_new_positions(self, new_positions):
        """Initialize with the new positions

        Parameters
        ----------

        new_positions: 2d array of int64
          new_positions[1] and new_positions[0] correspond to the x and y values, respectively.

        """
        self.new_positions = new_positions
        xs = new_positions[1]
        ys = new_positions[0]

        ind_new_positions_subdom = []

        for iy in range(self.nb_subdomy):
            for ix in range(self.nb_subdomx):
                ind_new_positions_subdom.append(
                    np.where(
                        (xs >= self.xmin_limits[ix])
                        & (xs < self.xmax_limits[ix])
                        & (ys >= self.ymin_limits[iy])
                        & (ys < self.ymin_limits[iy])
                    )[0]
                )

        self.ind_new_positions_subdom = ind_new_positions_subdom

        self.norm_coefs = np.zeros(self.new_positions.shape[1])
        for i_subdom in range(self.nb_subdom):
            # TODO: replace 1 by an appropriate function of
            # ind_new_positions_subdom[i_subdom]
            # + save another list of 1d array like ind_new_positions_subdom
            # containing the norm coefficients for each subdomain
            self.norm_coefs[ind_new_positions_subdom[i_subdom]] += 1

        EM = [None] * self.nb_subdom

        for i in range(self.nb_subdom):
            centers_tmp = self.centers[:, self.ind_v_subdom[i]]
            new_positions_tmp = new_positions[:, ind_new_positions_subdom[i]]
            EM[i] = compute_tps_matrix(new_positions_tmp, centers_tmp)

        self.EM = EM

    def interpolate(self, U_tps):
        U_eval = np.zeros(self.new_positions.shape[1])

        for i in range(self.nb_subdom):
            U_eval[self.ind_new_positions_subdom[i]] += np.dot(
                U_tps[i], self.EM[i]
            )

        U_eval /= self.norm_coefs

        return U_eval

    def compute_tps_coeff_iter(self, centers, values: Array[np.float64, "1d"]):
        """Compute the thin plate spline (tps) coefficients removing erratic
        vectors

        It computes iteratively "compute_tps_coeff", compares the tps
        result to the initial data and remove it if difference is
        larger than the given threshold

        """
        summary = {"nb_fixed_vectors": 0}

        # normalization as UVmat so that the effect of the filter do not depends
        # too much on the size of the domains
        smoothing_coef = self.smoothing_coef * values.size / 1000

        U_smooth, U_tps = compute_tps_coeff(centers, values, smoothing_coef)
        count = 1
        if self.threshold is not None:
            differences = np.sqrt((U_smooth - values) ** 2)
            ind_erratic_vector = np.argwhere(differences > self.threshold)

            summary["max(Udiff)"] = max(differences)

            nb_fixed_vectors = 0
            while ind_erratic_vector.size != 0:
                nb_fixed_vectors += ind_erratic_vector.size
                values[ind_erratic_vector] = U_smooth[ind_erratic_vector]
                U_smooth, U_tps = compute_tps_coeff(
                    centers, values, smoothing_coef
                )

                differences = np.sqrt((U_smooth - values) ** 2)
                ind_erratic_vector = np.argwhere(differences > self.threshold)
                count += 1

                if count > 10:
                    print(
                        "iterative tps interp.: stopped because maximum number "
                        "of iteration (10) was reached. "
                        "params.multipass.threshold_tps might be too small."
                    )
                    break

            summary["nb_fixed_vectors"] = nb_fixed_vectors
        summary["nb_iterations"] = count
        return U_smooth, U_tps, summary
