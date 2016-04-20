"""Piv work and subworks
========================

.. todo::

   - better multipass


.. autoclass:: WorkPIV
   :members:
   :private-members:

"""

from __future__ import print_function

from fluiddyn.util.paramcontainer import ParamContainer

from .. import BaseWork

from .fix import WorkFIX
from .singlepass import FirstWorkPIV, WorkPIVFromDisplacement

from ...data_objects.piv import MultipassPIVResults


class WorkPIV(BaseWork):
    """Main work for PIV with multipass."""

    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag='params')
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params):
        FirstWorkPIV._complete_params_with_default(params)
        WorkFIX._complete_params_with_default(params)

        params._set_child(
            'multipass',
            attribs={'number': 0,
                     'use_tps': True})

    def __init__(self, params=None):
        self.params = params
        self.work_piv0 = FirstWorkPIV(params)
        self.work_fix0 = WorkFIX(params.fix)

        if params.multipass.number > 0:
            self.work_piv1 = WorkPIVFromDisplacement(params)
            self.work_fix1 = WorkFIX(params.fix)

    def calcul(self, couple):

        piv_result = self.work_piv0.calcul(couple)
        piv_result = self.work_fix0.calcul(piv_result)

        results = MultipassPIVResults()
        results.append(piv_result)

        if self.params.multipass.number > 0:
            piv_result1 = self.work_piv1.calcul(piv_result)
            piv_result1 = self.work_fix1.calcul(piv_result)
            self.work_piv1.apply_interp(piv_result1)
            results.append(piv_result1)

        return results

    def _prepare_with_image(self, im):
        self.work_piv0._prepare_with_image(im)
        if self.params.multipass.number > 0:
            self.work_piv1._prepare_with_image(im)
