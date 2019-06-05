"""Fix work
===========

.. autoclass:: WorkFIX
   :members:
   :private-members:

"""

import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer

from .. import BaseWork
from ...calcul.smooth_clean import smooth_clean


class WorkFIX(BaseWork):
    """Fix a displacement vector field.

    """

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

    def __init__(self, params, piv_work):
        self.params = params
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
        inds = (piv_results.correls_max < self.params.correl_min).nonzero()[0]
        put_to_nan(inds, "correl < correl_min")

        # condition delta2 < displacement_max2
        if self.params.displacement_max:
            displacement_max2 = self.params.displacement_max ** 2
            delta2s = deltaxs ** 2 + deltays ** 2
            with np.errstate(invalid="ignore"):
                inds = (delta2s > displacement_max2).nonzero()[0]
            put_to_nan(inds, "delta2 < displacement_max2")

        # warning condition neighbour not implemented...
        if self.params.threshold_diff_neighbour is not None:
            threshold = self.params.threshold_diff_neighbour
            ixvecs = self.piv_work.ixvecs
            iyvecs = self.piv_work.iyvecs
            xs = piv_results.xs
            ys = piv_results.ys

            # import ipdb; ipdb.set_trace()
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

            for ind in inds:
                piv_results.errors[ind] += " (diff = {:.2f})".format(
                    differences[ind]
                )

            piv_results.deltaxs_wrong = deltaxs_wrong
            piv_results.deltays_wrong = deltays_wrong

        # condition check 2d peak
        ratio_correl_peaks = 0.85

        secondary_peaks = piv_results.secondary_peaks
        for ind, other_peaks in enumerate(secondary_peaks):
            if other_peaks is None or len(other_peaks) == 0:
                continue

            correl0 = piv_results.correls_max[ind]

            other_peaks_good = []
            for (dx, dy, corr) in other_peaks:
                if (
                    corr / correl0 > ratio_correl_peaks
                    and corr > self.params.correl_min
                ):
                    other_peaks_good.append((dx, dy, corr))

            if len(other_peaks_good) == 0:
                continue

            diff_neighbours = np.empty(len(other_peaks_good) + 1)
            diff_neighbours[0] = differences[ind]

            for i, (dx, dy, corr) in enumerate(other_peaks_good):
                diff_neighbours[i + 1] = np.sqrt(
                    (dxs_smooth[ind] - dx) ** 2 + (dys_smooth[ind] - dy) ** 2
                )

            argmin = diff_neighbours.argmin()

            if argmin != 0:
                print("replace peak")
                dx, dy, corr = other_peaks_good[argmin - 1]
                other_peaks_good[argmin - 1] = (
                    piv_results.deltaxs[ind],
                    piv_results.deltays[ind],
                    piv_results.correls_max[ind],
                )
                piv_results.secondary_peaks[ind] = other_peaks_good

                piv_results.deltaxs[ind] = dx
                piv_results.deltays[ind] = dy
                piv_results.correls_max[ind] = corr

                if ind in piv_results.errors:
                    del (
                        piv_results.deltaxs_wrong[ind],
                        piv_results.deltays_wrong[ind],
                    )
                    print("!!!!! TO DO ind in piv_results.errors !!!!!")

        return piv_results
