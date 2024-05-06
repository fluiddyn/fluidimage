"""Thin plate spline with subdomains
====================================

Translated and adapted from UVmat code (Joel Sommeria, LEGI, CNRS).

This interpolation/smoothing (Duchon, 1976; NguyenDuc and Sommeria,
1988) minimises a linear combination of the squared curvature and
squared difference from the initial data.

.. autoclass:: ThinPlateSplineSubdom
   :members:

"""

from logging import debug
from math import sqrt
from typing import List

import numpy as np
from transonic import Array

from .thin_plate_spline import compute_tps_matrix, compute_tps_weights


class ThinPlateSplineSubdom:
    """Helper class for thin plate interpolation."""

    num_centers: int
    tps_matrices: List["float[:,:]"]
    norm_coefs: "float[:]"
    norm_coefs_domains: List["float[:]"]

    num_new_positions: int
    ind_new_positions_domains: List["np.int64[:]"]
    norm_coefs_new_pos: "float[:]"
    norm_coefs_new_pos_domains: List["float[:]"]

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
        self.threshold = threshold

        # note: `centers = np.vstack([ys, xs])`
        xs = self.centers[1]
        ys = self.centers[0]
        self.num_centers = xs.size
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        range_x = x_max - x_min
        range_y = y_max - y_min
        aspect_ratio = range_y / range_x

        nb_subdom = xs.size / self.subdom_size
        nb_subdomx = int(np.floor(sqrt(nb_subdom / aspect_ratio)))
        nb_subdomx = nb_subdomx or 1
        nb_subdomy = int(np.ceil(sqrt(nb_subdom * aspect_ratio)))
        nb_subdomy = nb_subdomy or 1

        debug(f"nb_subdomx: {nb_subdomx} ; nb_subdomy: {nb_subdomy}")

        self.nb_subdomx = nb_subdomx
        self.nb_subdomy = nb_subdomy
        self.nb_subdom = nb_subdomx * nb_subdomy

        x_dom = np.linspace(x_min, x_max, nb_subdomx + 1)
        y_dom = np.linspace(y_min, y_max, nb_subdomy + 1)

        # normalization as UVmat so that the effect of the filter do not depends
        # too much on the size of the domains
        num_vectors_per_subdom = xs.size / self.nb_subdom
        self.smoothing_coef = smoothing_coef * sqrt(
            (x_dom[1] - x_dom[0]) * (y_dom[1] - y_dom[0]) / num_vectors_per_subdom
        )

        coef_buffer = percent_buffer_area / 100
        buffer_length_x = coef_buffer * range_x / nb_subdomx
        buffer_length_y = coef_buffer * range_y / nb_subdomy

        self.limits_min_x = x_dom[:-1] - buffer_length_x
        self.limits_max_x = x_dom[1:] + buffer_length_x

        self.limits_min_y = y_dom[:-1] - buffer_length_y
        self.limits_max_y = y_dom[1:] + buffer_length_y

        self.xmax_domains = np.empty(self.nb_subdom)
        self.ymax_domains = np.empty(self.nb_subdom)
        self.xc_domains = np.empty(self.nb_subdom)
        self.yc_domains = np.empty(self.nb_subdom)
        self.indices_domains = []

        i_dom = 0
        for iy in range(nb_subdomy):
            for ix in range(nb_subdomx):
                xmin = self.limits_min_x[ix]
                xmax = self.limits_max_x[ix]
                ymin = self.limits_min_y[iy]
                ymax = self.limits_max_y[iy]

                self.indices_domains.append(
                    np.where(
                        (xs >= xmin) & (xs < xmax) & (ys >= ymin) & (ys < ymax)
                    )[0]
                )

                self.xmax_domains[i_dom] = xmax
                self.ymax_domains[i_dom] = ymax
                self.xc_domains[i_dom] = 0.5 * (xmin + xmax)
                self.yc_domains[i_dom] = 0.5 * (ymin + ymax)
                i_dom += 1

        self.norm_coefs = np.zeros(self.num_centers)
        self.norm_coefs_domains = []
        for i_dom in range(self.nb_subdom):
            indices_domain = self.indices_domains[i_dom]
            xs_domain = xs[indices_domain]
            ys_domain = ys[indices_domain]
            coefs = self._compute_coef_norm(xs_domain, ys_domain, i_dom)
            self.norm_coefs_domains.append(coefs)
            self.norm_coefs[indices_domain] += coefs

    def compute_tps_weights_subdom(self, values):
        """Compute the TPS weights for all subdomains"""
        smoothed_field_domains = [None] * self.nb_subdom
        weights_domains = [None] * self.nb_subdom
        summaries = [None] * self.nb_subdom

        for idx in range(self.nb_subdom):
            centers_domain = self.centers[:, self.indices_domains[idx]]
            values_domain = values[self.indices_domains[idx]]
            (
                smoothed_field_domains[idx],
                weights_domains[idx],
                summaries[idx],
            ) = self.compute_tps_weights_iter(
                centers_domain, values_domain, self.smoothing_coef
            )

        smoothed_field = np.zeros(self.num_centers)
        summary = {"nb_fixed_vectors": [], "max(Udiff)": [], "nb_iterations": []}

        for idx in range(self.nb_subdom):
            indices_domain = self.indices_domains[idx]
            smoothed_field[indices_domain] += (
                self.norm_coefs_domains[idx] * smoothed_field_domains[idx]
            )
            for key in ("nb_fixed_vectors", "max(Udiff)", "nb_iterations"):
                summary[key].append(summaries[idx][key])

        summary["nb_fixed_vectors_tot"] = sum(summary["nb_fixed_vectors"])
        smoothed_field /= self.norm_coefs

        return smoothed_field, weights_domains, summary

    def init_with_new_positions(self, new_positions):
        """Initialize with the new positions

        Parameters
        ----------

        new_positions: 2d array of int64
          new_positions[1] and new_positions[0] correspond to the x and y values, respectively.

        """
        xs = new_positions[1]
        ys = new_positions[0]
        self.num_new_positions = xs.size

        self.ind_new_positions_domains = ind_new_positions_domains = []
        for iy in range(self.nb_subdomy):
            for ix in range(self.nb_subdomx):
                ind_new_positions_domains.append(
                    np.where(
                        (xs >= self.limits_min_x[ix])
                        & (xs < self.limits_max_x[ix])
                        & (ys >= self.limits_min_y[iy])
                        & (ys < self.limits_max_y[iy])
                    )[0]
                )

        self.norm_coefs_new_pos = np.zeros(self.num_new_positions)
        self.norm_coefs_new_pos_domains = []

        for i_domain in range(self.nb_subdom):
            indices_domain = ind_new_positions_domains[i_domain]
            xs_domain = xs[indices_domain]
            ys_domain = ys[indices_domain]
            coefs = self._compute_coef_norm(xs_domain, ys_domain, i_domain)
            self.norm_coefs_new_pos_domains.append(coefs)
            self.norm_coefs_new_pos[indices_domain] += coefs

        self.tps_matrices = [None] * self.nb_subdom
        for i_domain in range(self.nb_subdom):
            centers_tmp = self.centers[:, self.indices_domains[i_domain]]
            new_positions_tmp = new_positions[
                :, ind_new_positions_domains[i_domain]
            ]
            self.tps_matrices[i_domain] = compute_tps_matrix(
                new_positions_tmp, centers_tmp
            )

    def _compute_coef_norm(self, xs_domain, ys_domain, i_domain):

        x_center_domain = self.xc_domains[i_domain]
        y_center_domain = self.yc_domains[i_domain]

        x_max_domain = self.xmax_domains[i_domain]
        y_max_domain = self.ymax_domains[i_domain]

        dx_max = x_max_domain - x_center_domain
        dy_max = y_max_domain - y_center_domain

        dx2_normed = (xs_domain - x_center_domain) ** 2 / dx_max**2
        dy2_normed = (ys_domain - y_center_domain) ** 2 / dy_max**2

        return (1 - dx2_normed) * (1 - dy2_normed) + 0.001

    def interpolate(self, weights_domains):
        """Interpolate on new positions"""
        values = np.zeros(self.num_new_positions)
        for i_domain in range(self.nb_subdom):
            values[
                self.ind_new_positions_domains[i_domain]
            ] += self.norm_coefs_new_pos_domains[i_domain] * np.dot(
                weights_domains[i_domain], self.tps_matrices[i_domain]
            )
        values /= self.norm_coefs_new_pos
        return values

    def compute_tps_weights_iter(
        self, centers, values: Array[np.float64, "1d"], smoothing_coef
    ):
        """Compute the thin plate spline (tps) coefficients removing erratic
        vectors

        It computes iteratively "compute_tps_weights", compares the tps
        result to the initial data and remove it if difference is
        larger than the given threshold

        """
        summary = {"nb_fixed_vectors": 0}
        smoothed_values, tps_weights = compute_tps_weights(
            centers, values, smoothing_coef
        )
        count = 1
        if self.threshold is not None:
            differences = np.sqrt((smoothed_values - values) ** 2)
            ind_erratic_vector = np.argwhere(differences > self.threshold)

            summary["max(Udiff)"] = max(differences)

            nb_fixed_vectors = 0
            while ind_erratic_vector.size != 0:
                nb_fixed_vectors += ind_erratic_vector.size
                values[ind_erratic_vector] = smoothed_values[ind_erratic_vector]
                smoothed_values, tps_weights = compute_tps_weights(
                    centers, values, smoothing_coef
                )

                differences = np.sqrt((smoothed_values - values) ** 2)
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
        return smoothed_values, tps_weights, summary
