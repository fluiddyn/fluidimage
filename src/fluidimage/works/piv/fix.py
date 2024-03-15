"""Fix work
===========

.. autoclass:: WorkFIX
   :members:
   :private-members:

"""

import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage.calcul.errors import PIVError

from ...calcul.smooth_clean import smooth_clean
from .. import BaseWork


class WorkFIX(BaseWork):
    """Fix a displacement vector field."""

    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag="params")
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params, tag="fix"):
        params._set_child(
            tag,
            attribs={
                "correl_min": 0.2,
                "threshold_diff_neighbour": 10,
                "displacement_max": None,
            },
        )

        params.fix._set_doc(
            """
Parameters indicating how are detected and processed false vectors.

correl_min : 0.2

    Vectors associated with correlation smaller than correl_min are considered
    as false vectors.

threshold_diff_neighbour : 10

    Vectors for which the difference with the average vectors is larger than
    `threshold_diff_neighbour` are considered as false vectors.

displacement_max : None

    Vectors larger than `displacement_max` are considered as false vectors.

"""
        )

    def __init__(self, params_fix, piv_work):
        self.params_fix = params_fix
        self.piv_work = piv_work

    def calcul(self, piv_results):
        # print('piv_results = ', piv_results)
        deltaxs_wrong = {}
        deltays_wrong = {}

        deltaxs = piv_results.deltaxs
        deltays = piv_results.deltays
        for ierr in piv_results.errors.keys():
            deltaxs_wrong[ierr] = deltaxs[ierr]
            deltays_wrong[ierr] = deltays[ierr]
            deltaxs[ierr] = np.nan
            deltays[ierr] = np.nan

        def put_to_nan(inds, explanation):
            for ind in inds:
                ind = int(ind)

                if ind not in deltaxs_wrong:
                    deltaxs_wrong[ind] = deltaxs[ind]
                    deltays_wrong[ind] = deltays[ind]

                    deltaxs[ind] = np.nan
                    deltays[ind] = np.nan
                    piv_results.errors[ind] = explanation
                else:
                    piv_results.errors[ind] += " + " + explanation

        # condition correl < correl_min
        inds = (piv_results.correls_max < self.params_fix.correl_min).nonzero()[0]
        put_to_nan(inds, "correl < correl_min")

        # condition delta2 < displacement_max2
        if self.params_fix.displacement_max:
            displacement_max2 = self.params_fix.displacement_max**2
            delta2s = deltaxs**2 + deltays**2
            with np.errstate(invalid="ignore"):
                inds = (delta2s > displacement_max2).nonzero()[0]
            put_to_nan(inds, "delta2 < displacement_max2")

        if self.params_fix.threshold_diff_neighbour is not None:
            threshold = self.params_fix.threshold_diff_neighbour
            ixvecs = self.piv_work.ixvecs
            iyvecs = self.piv_work.iyvecs
            xs = piv_results.xs
            ys = piv_results.ys

            ixvecs, iyvecs = self.piv_work._xyoriginalimage_from_xymasked(
                ixvecs, iyvecs
            )

            dxs_smooth, dys_smooth = smooth_clean(
                xs, ys, deltaxs, deltays, iyvecs, ixvecs, threshold
            )
            piv_results.dxs_smooth_clean = dxs_smooth
            piv_results.dys_smooth_clean = dys_smooth

            differences = np.sqrt(
                (dxs_smooth - deltaxs) ** 2 + (dys_smooth - deltays) ** 2
            )

            with np.errstate(invalid="ignore"):
                inds = (differences > threshold).nonzero()[0]

            put_to_nan(inds, "diff neighbour too large")

            for ivec in inds:
                piv_results.errors[ivec] += " (diff = {:.2f})".format(
                    differences[ivec]
                )

            piv_results.deltaxs_wrong = deltaxs_wrong
            piv_results.deltays_wrong = deltays_wrong

        if self.piv_work.params.piv0.nb_peaks_to_search < 2:
            return piv_results

        # condition check 2nd peak
        ratio_correl_peaks = 0.6
        nb_bad_peaks_replaced = 0
        secondary_peaks = piv_results.secondary_peaks
        for ivec, other_peaks in enumerate(secondary_peaks):
            if other_peaks is None or len(other_peaks) == 0:
                continue

            correl0 = piv_results.correls_max[ivec]

            try:
                dx_input = piv_results.deltaxs_input[ivec]
                dy_input = piv_results.deltays_input[ivec]
            except AttributeError:
                # first pass
                dx_input = 0
                dy_input = 0

            other_peaks_good = []
            for dx, dy, corr in other_peaks:
                if (
                    corr / correl0 > ratio_correl_peaks
                    and corr > self.params_fix.correl_min
                ):
                    dx += dx_input
                    dy += dy_input
                    other_peaks_good.append((dx, dy, corr))

            if len(other_peaks_good) == 0:
                continue

            diff_neighbours = np.empty(len(other_peaks_good) + 1)
            diff_neighbours[0] = differences[ivec]

            for i, (dx, dy, corr) in enumerate(other_peaks_good):
                diff_neighbours[i + 1] = np.sqrt(
                    (dxs_smooth[ivec] - dx) ** 2 + (dys_smooth[ivec] - dy) ** 2
                )

            argmin = diff_neighbours.argmin()

            if argmin != 0:
                dx, dy, corr = other_peaks_good[argmin - 1]

                if (
                    self.params_fix.threshold_diff_neighbour is not None
                    and diff_neighbours[argmin]
                    > self.params_fix.threshold_diff_neighbour
                ):
                    continue

                if (
                    self.params_fix.displacement_max
                    and dx**2 + dy**2 > self.params_fix.displacement_max
                ):
                    continue

                # try to apply subpix
                correl = piv_results.correls[ivec]
                dx -= dx_input
                dy -= dy_input
                try:
                    dx, dy = self.piv_work.correl.apply_subpix(dx, dy, correl)
                except PIVError:
                    continue
                dx += dx_input
                dy += dy_input
                nb_bad_peaks_replaced += 1
                # saved the replaced vector
                try:
                    replaced_vectors = piv_results.replaced_vectors
                except AttributeError:
                    replaced_vectors = piv_results.replaced_vectors = {}

                try:
                    old_dx = piv_results.deltaxs_wrong[ivec]
                    old_dy = piv_results.deltays_wrong[ivec]
                except KeyError:
                    old_dx = piv_results.deltaxs[ivec]
                    old_dy = piv_results.deltays[ivec]

                replaced_vectors[ivec] = (
                    old_dx,
                    old_dy,
                    piv_results.correls_max[ivec],
                )

                # replace!
                piv_results.deltaxs[ivec] = dx
                piv_results.deltays[ivec] = dy
                piv_results.correls_max[ivec] = corr

                if ivec in piv_results.errors:
                    del (
                        piv_results.deltaxs_wrong[ivec],
                        piv_results.deltays_wrong[ivec],
                        piv_results.errors[ivec],
                    )

        if nb_bad_peaks_replaced > 0:
            print(
                f"Secondary peaks: {nb_bad_peaks_replaced} bad peak(s) replaced"
            )

        return piv_results
