"""Multipass PIV
================

.. autoclass:: WorkPIV
   :members:
   :private-members:

"""

from __future__ import print_function

# import os
# import sys
from copy import copy

from fluiddyn.util.paramcontainer import ParamContainer

from .. import BaseWork
# from ...data_objects.piv import get_name_piv

from .fix import WorkFIX
from .singlepass import FirstWorkPIV, WorkPIVFromDisplacement, InterpError

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
                     'subdom_size': 200,
                     'smoothing_coef': 0.5,
                     'threshold_tps': 1.})

        params.multipass._set_doc(
            """Multipass PIV parameters:

number : int (default 1)
    Number of PIV passes.

coeff_zoom : integer or iterable of size `number - 1`.

    Coefficient defining the size of the interrogation windows for the passes 1
            to `number - 1` (always defined comparing the passes `i-1`).

use_tps : bool or 'last'

    If it is True, the interpolation is done using the Thin Plate Spline method
    (computationnally heavy but sometimes interesting). If it is 'last', the
    TPS method is used only for the last pass.

subdom_size : int

    Number of vectors in the subdomains used for the TPS method.

smoothing_coef : float

    Coefficient used for the TPS method. The result is smoother for larger
    smoothing_coef.

threshold_tps :  float

    Allowed difference of displacement (in pixels) between smoothed and input
    field for TPS filter.

""")

    def __init__(self, params=None):
        if params.piv0.method_subpix == "2d_gaussian2" and \
           params.piv0.nsubpix is not None:
            raise ValueError(
                "For the subpixel method '2d_gaussian2', "
                "params.piv0.nsubpix has to be equal to None.")
        self.params = params

        self.works_piv = []
        self.works_fix = []

        work_piv = FirstWorkPIV(params)
        self.works_piv.append(work_piv)
        self.works_fix.append(WorkFIX(params.fix, work_piv))

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

            work_piv = WorkPIVFromDisplacement(
                params, index_pass=i, shape_crop_im0=shape_crop_im0,
                shape_crop_im1=shape_crop_im1)
            self.works_piv.append(work_piv)
            self.works_fix.append(WorkFIX(params.fix, work_piv))

    def calcul(self, couple):

        # if hasattr(couple, 'serie'):
        #     serie = couple.serie
        # else:
        #     serie = couple
        # name = get_name_piv(serie)[:-3]
        # sys.stdout = open('log_' + name + '_' + str(os.getpid()), 'w')

        results = MultipassPIVResults()

        # just for simplicity
        piv_result = couple

        for i, work_piv in enumerate(self.works_piv):
            work_fix = self.works_fix[i]
            piv_result = work_piv.calcul(piv_result)
            piv_result = work_fix.calcul(piv_result)
            results.append(piv_result)

        try:
            work_piv.apply_interp(piv_result, last=True)
        except InterpError as e:
            print('Warning: InterpError at the end of the last piv pass:', e)

        return results

    def _prepare_with_image(self, im):
        for work_piv in self.works_piv:
            work_piv._prepare_with_image(im)
