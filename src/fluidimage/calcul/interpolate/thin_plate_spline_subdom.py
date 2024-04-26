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
from typing import List

import numpy as np
from transonic import Array

from .thin_plate_spline import compute_tps_matrix, compute_tps_weights


class ThinPlateSplineSubdom:
    """Helper class for thin plate interpolation."""

    ind_new_positions_domains: List["np.int64[:]"]
    norm_coefs: "float[:]"
    num_new_positions: int
    num_centers: int
    tps_matrices: List["float[:,:]"]

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
        nb_subdomx = int(np.floor(np.sqrt(nb_subdom / aspect_ratio)))
        nb_subdomx = nb_subdomx or 1
        nb_subdomy = int(np.ceil(np.sqrt(nb_subdom * aspect_ratio)))
        nb_subdomy = nb_subdomy or 1

        debug(f"nb_subdomx: {nb_subdomx} ; nb_subdomy: {nb_subdomy}")

        self.nb_subdomx = nb_subdomx
        self.nb_subdomy = nb_subdomy
        self.nb_subdom = nb_subdomx * nb_subdomy

        x_dom = np.linspace(x_min, x_max, nb_subdomx + 1)
        y_dom = np.linspace(y_min, y_max, nb_subdomy + 1)

        # normalization as UVmat so that the effect of the filter do not depends
        # too much on the size of the domains
        self.smoothing_coef = (
            smoothing_coef * (x_dom[1] - x_dom[0]) * (y_dom[1] - y_dom[0]) / 1000
        )

        coef_buffer = percent_buffer_area / 100
        buffer_length_x = coef_buffer * range_x / nb_subdomx
        buffer_length_y = coef_buffer * range_y / nb_subdomy

        self.xmin_limits = x_dom[:-1] - buffer_length_x
        self.xmax_limits = x_dom[1:] + buffer_length_x

        self.ymin_limits = y_dom[:-1] - buffer_length_y
        self.ymax_limits = y_dom[1:] + buffer_length_y

        self.indices_domains = []
        for iy in range(nb_subdomy):
            for ix in range(nb_subdomx):
                self.indices_domains.append(
                    np.where(
                        (xs >= self.xmin_limits[ix])
                        & (xs < self.xmax_limits[ix])
                        & (ys >= self.ymin_limits[iy])
                        & (ys < self.ymin_limits[iy])
                    )[0]
                )

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
        nb_tps = np.zeros(self.num_centers, dtype=int)
        summary = {"nb_fixed_vectors": [], "max(Udiff)": [], "nb_iterations": []}

        for idx in range(self.nb_subdom):
            smoothed_field[self.indices_domains[idx]] += smoothed_field_domains[
                idx
            ]
            nb_tps[self.indices_domains[idx]] += 1
            for key in ("nb_fixed_vectors", "max(Udiff)", "nb_iterations"):
                summary[key].append(summaries[idx][key])

        summary["nb_fixed_vectors_tot"] = sum(summary["nb_fixed_vectors"])
        smoothed_field /= nb_tps

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
                        (xs >= self.xmin_limits[ix])
                        & (xs < self.xmax_limits[ix])
                        & (ys >= self.ymin_limits[iy])
                        & (ys < self.ymin_limits[iy])
                    )[0]
                )

        self.norm_coefs = np.zeros(self.num_new_positions)
        for i_domain in range(self.nb_subdom):
            # TODO: replace 1 by an appropriate function of
            # ind_new_positions_domains[i_subdom]
            # + save another list of 1d array like ind_new_positions_domains
            # containing the norm coefficients for each subdomain
            self.norm_coefs[ind_new_positions_domains[i_domain]] += 1

        self.tps_matrices = [None] * self.nb_subdom
        for i_domain in range(self.nb_subdom):
            centers_tmp = self.centers[:, self.indices_domains[i_domain]]
            new_positions_tmp = new_positions[
                :, ind_new_positions_domains[i_domain]
            ]
            self.tps_matrices[i_domain] = compute_tps_matrix(
                new_positions_tmp, centers_tmp
            )

    def interpolate(self, tps_weights_domains):
        """Interpolate on new positions"""
        values = np.zeros(self.num_new_positions)
        for i_domain in range(self.nb_subdom):
            values[self.ind_new_positions_domains[i_domain]] += np.dot(
                tps_weights_domains[i_domain], self.tps_matrices[i_domain]
            )
        values /= self.norm_coefs
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
