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
        params = ParamContainer(tag='params')
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params, tag='fix'):

        params._set_child(tag, attribs={
            'correl_min': 0.2,
            'threshold_diff_neighbour': 10,
            'displacement_max': None})

    def __init__(self, params, piv_work):
        self.params = params
        self.piv_work = piv_work

    def calcul(self, piv_results):

        deltaxs_wrong = {}
        deltays_wrong = {}

        deltaxs = piv_results.deltaxs
        deltays = piv_results.deltays

        for ierr in piv_results.errors.keys():
            deltaxs[ierr] = np.nan
            deltays[ierr] = np.nan

        def put_to_nan(inds, explanation):
            for ind in inds:
                ind = int(ind)

                deltaxs_wrong[ind] = deltaxs[ind]
                deltays_wrong[ind] = deltays[ind]

                deltaxs[ind] = np.nan
                deltays[ind] = np.nan
                try:
                    piv_results.errors[ind] += ' + ' + explanation
                except KeyError:
                    piv_results.errors[ind] = explanation

        # condition correl < correl_min
        inds = (piv_results.correls_max < self.params.correl_min).nonzero()[0]
        put_to_nan(inds, 'correl < correl_min')

        # condition delta2 < displacement_max2
        if self.params.displacement_max:
            displacement_max2 = self.params.displacement_max**2
            delta2s = deltaxs**2 + deltays**2
            with np.errstate(invalid='ignore'):
                inds = (delta2s > displacement_max2).nonzero()[0]
            put_to_nan(inds, 'delta2 < displacement_max2')

        # warning condition neighbour not implemented...
        if self.params.threshold_diff_neighbour is not None:
            threshold = self.params.threshold_diff_neighbour
            ixvecs = self.piv_work.ixvecs
            iyvecs = self.piv_work.iyvecs
            xs = piv_results.xs
            ys = piv_results.ys

            dxs, dys = smooth_clean(
                xs, ys, deltaxs, deltays, iyvecs, ixvecs, threshold)

            piv_results.dxs_smooth_clean = dxs
            piv_results.dys_smooth_clean = dys

            with np.errstate(invalid='ignore'):
                inds = (abs(dxs - deltaxs) +
                        abs(dys - deltays) > threshold).nonzero()[0]

            # from fluiddyn.debug import ipydebug
            # import matplotlib.pylab as plb
            # plb.ion()
            # ipydebug()

            put_to_nan(inds, 'diff neighbour too large')

        piv_results.deltaxs_wrong = deltaxs_wrong
        piv_results.deltays_wrong = deltays_wrong

        return piv_results
