"""Fix work
===========

.. autoclass:: WorkFIX
   :members:
   :private-members:

"""

import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer

from .. import BaseWork


class WorkFIX(BaseWork):
    """Fix a displacement vector field.

    .. todo::

       Calculus default delta_max! Actual default value (4) is bad!

    """

    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag='params')
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params, tag='fix'):

        params._set_child(tag, attribs={
            'correl_min': 0.4,
            # 'delta_diff': 0.1,
            'delta_max': None,
            'remove_error_vec': True})

    def __init__(self, params):
        self.params = params

    def calcul(self, piv_results):

        deltaxs = piv_results.deltaxs
        deltays = piv_results.deltays

        for ierr in piv_results.errors.keys():
            deltaxs[ierr] = np.nan
            deltays[ierr] = np.nan

        def put_to_nan(inds, explanation):
            for ind in inds:
                ind = int(ind)
                deltaxs[ind] = np.nan
                deltays[ind] = np.nan
                try:
                    piv_results.errors[ind] += ' + ' + explanation
                except KeyError:
                    piv_results.errors[ind] = explanation

        # condition correl < correl_min
        inds = (piv_results.correls_max < self.params.correl_min).nonzero()[0]
        put_to_nan(inds, 'correl < correl_min')

        # condition delta2 < delta_max2
	if self.params.delta_max:
            delta_max2 = self.params.delta_max**2
            delta2s = deltaxs**2 + deltays**2
            inds = (delta2s > delta_max2).nonzero()[0]
            put_to_nan(inds, 'delta2 < delta_max2')

        # warning condition neighbour not implemented...

        return piv_results
