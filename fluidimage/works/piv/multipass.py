"""Multipass PIV
================

.. autoclass:: WorkPIV
   :members:
   :private-members:

"""

from __future__ import print_function

from copy import copy

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
            attribs={'number': 1,
                     'coeff_zoom': 2,
                     'use_tps': 'last',
                     'subdom_size': 1000,
                     'smoothing_coef': 0.5})

        params.multipass._set_doc(
            """Multipass PIV parameters:

`coeff_zoom` can be an integer or a iterable of size `number - 1`.
""")

    def __init__(self, params=None):
        self.params = params

        self.works_piv = []
        self.works_fix = []

        self.works_piv.append(FirstWorkPIV(params))
        self.works_fix.append(WorkFIX(params.fix))

        coeff_zoom = params.multipass.coeff_zoom

        if isinstance(coeff_zoom, int):
            coeffs_zoom = [coeff_zoom] * (params.multipass.number - 1)
        elif len(coeff_zoom) == params.multipass.number - 1:
            coeffs_zoom = coeff_zoom
        else:
            raise ValueError(
                'params.multipass.coeff_zoom has to be an integer or '
                'an iterable of length params.multipass.number - 1')

        shape_crop_im0 = copy(params.piv0.shape_crop_im0)
        shape_crop_im1 = copy(params.piv0.shape_crop_im1)
        if shape_crop_im1 is None:
            shape_crop_im1 = shape_crop_im0

        if isinstance(shape_crop_im0, int):
            shape_crop_im0 = (shape_crop_im0, shape_crop_im0)
        elif not(isinstance(shape_crop_im0, tuple) and
                 len(shape_crop_im0) == 2):
            raise NotImplementedError(
                'For now, shape_crop_im0 has to be one or two integer!')
        if isinstance(shape_crop_im1, int):
            shape_crop_im1 = (shape_crop_im1, shape_crop_im1)
        elif not(isinstance(shape_crop_im1, tuple) and
                 len(shape_crop_im1) == 2):
            raise NotImplementedError(
                'For now, shape_crop_im1 has to be one or two integer!')

        for i in range(1, params.multipass.number):

            shape_crop_im0 = (copy(shape_crop_im0[0]/coeffs_zoom[i-1]),
                              copy(shape_crop_im0[1]/coeffs_zoom[i-1]))
            shape_crop_im1 = (copy(shape_crop_im1[0]/coeffs_zoom[i-1]),
                              copy(shape_crop_im1[1]/coeffs_zoom[i-1]))

            self.works_piv.append(
                WorkPIVFromDisplacement(
                    params, index_pass=i, shape_crop_im0=shape_crop_im0,
                    shape_crop_im1=shape_crop_im1))
            self.works_fix.append(WorkFIX(params.fix))

    def calcul(self, couple):

        results = MultipassPIVResults()

        # just for simplicity
        piv_result = couple

        for i, work_piv in enumerate(self.works_piv):
            work_fix = self.works_fix[i]
            piv_result = work_piv.calcul(piv_result)
            piv_result = work_fix.calcul(piv_result)
            results.append(piv_result)

        work_piv.apply_interp(piv_result, last=True)

        return results

    def _prepare_with_image(self, im):
        for work_piv in self.works_piv:
            work_piv._prepare_with_image(im)
