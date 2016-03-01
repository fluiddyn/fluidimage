"""Piv work and subworks
========================

To do:

- better multipass

- as in UVmat: better patch "thin-plate spline" (?). Add variables as
  in UVmat (NbCenter, Coord_tps, SubRange, U_tps, V_tps)

- detect and save multipeaks. Add variables:

  * deltaxs_2ndpeak {ivec: float32}
  * deltays_2ndpeak {ivec: float32}
  * correl_2ndpeak {ivec: float32}

"""


from __future__ import print_function

from copy import deepcopy

import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles

from ..data_objects.piv import (
    ArrayCouple, HeavyPIVResults, MultipassPIVResults)
from ..calcul.correl import PIVError, correlation_classes
from ..works import BaseWork

from ..calcul.interpolate.thin_plate_spline import \
    compute_tps_coeff, ThinPlateSpline


class BaseWorkPIV(BaseWork):

    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag='params')
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params):
        pass

    def __init__(self, params=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params

        shape_crop_im0 = params.piv0.shape_crop_im0
        overlap = params.piv0.grid.overlap

        self.shape_crop_im0 = shape_crop_im0
        if isinstance(shape_crop_im0, int):
            n_interrogation_window = shape_crop_im0
        else:
            raise NotImplementedError(
                'For now, shape_crop_im0 has to be an integer!')

        niw = self.n_interrogation_window = n_interrogation_window
        self.overlap = overlap

        self.niwo2 = niw/2

        self._init_correl()

    def _init_correl(self):
        niw = self.n_interrogation_window
        try:
            correl_cls = correlation_classes[self.params.piv0.method_correl]
        except KeyError:
            raise ValueError(
                'params.piv0.method_correl should be in ' +
                str(correlation_classes.keys()))

        self.correl = correl_cls(im0_shape=(niw, niw),
                                 method_subpix=self.params.piv0.method_subpix)

    def _prepare_with_image(self, im):

        len_y, len_x = im.shape
        niw = self.n_interrogation_window
        step = niw - int(np.round(self.overlap*niw))

        ixvecs = np.arange(0, len_x, step, dtype=int)
        iyvecs = np.arange(0, len_y, step, dtype=int)

        ixvecs, iyvecs = np.meshgrid(ixvecs, iyvecs)

        self.ixvecs_grid = ixvecs.flatten()
        self.iyvecs_grid = iyvecs.flatten()

    def calcul(self, couple):
        if isinstance(couple, SerieOfArraysFromFiles):
            couple = ArrayCouple(serie=couple)

        if not isinstance(couple, ArrayCouple):
            raise ValueError

        im0, im1 = couple.get_arrays()
        if not hasattr(self, 'ixvecs_grid'):
            self._prepare_with_image(im0)

        deltaxs, deltays, xs, ys, correls_max, correls, errors = \
            self._loop_vectors(im0, im1)

        result = HeavyPIVResults(
            deltaxs, deltays, xs, ys, errors,
            correls_max=correls_max, correls=correls,
            couple=deepcopy(couple), params=self.params)

        return result

    def _pad_images(self, im0, im1):
        npad = self.npad = self.niwo2 + 10
        tmp = [(npad, npad), (npad, npad)]
        im0pad = np.pad(im0 - im0.min(), tmp, 'constant')
        im1pad = np.pad(im1 - im1.min(), tmp, 'constant')
        return im0pad, im1pad

    def calcul_indices_vec(self, deltaxs_approx=None, deltays_approx=None):

        xs = self.ixvecs_grid
        ys = self.iyvecs_grid

        ixs0_pad = self.ixvecs_grid + self.npad
        iys0_pad = self.iyvecs_grid + self.npad
        ixs1_pad = ixs0_pad
        iys1_pad = iys0_pad

        return xs, ys, ixs0_pad, iys0_pad, ixs1_pad, iys1_pad

    def _loop_vectors(self, im0, im1,
                      deltaxs_approx=None, deltays_approx=None):

        im0pad, im1pad = self._pad_images(im0, im1)

        xs, ys, ixs0_pad, iys0_pad, ixs1_pad, iys1_pad = \
            self.calcul_indices_vec(deltaxs_approx=deltaxs_approx,
                                    deltays_approx=deltays_approx)

        nb_vec = len(xs)

        correls = [None]*nb_vec
        errors = {}
        deltaxs = np.empty(xs.shape, dtype='float32')
        deltays = np.empty_like(deltaxs)
        correls_max = np.empty_like(deltaxs)

        for ivec in range(nb_vec):

            ixvec0 = ixs0_pad[ivec]
            iyvec0 = iys0_pad[ivec]
            ixvec1 = ixs1_pad[ivec]
            iyvec1 = iys1_pad[ivec]

            im0crop = self._crop_im(ixvec0, iyvec0, im0pad)
            im1crop = self._crop_im(ixvec1, iyvec1, im1pad)

            correl, coef_norm = self.correl(im0crop, im1crop)
            correls[ivec] = correl
            try:
                deltay, deltax, correl_max = \
                    self.correl.compute_displacement_from_correl(
                        correl, coef_norm=coef_norm)
            except PIVError as e:
                errors[ivec] = e.explanation
                try:
                    deltay, deltax, correl_max = \
                        e.results_compute_displacement_from_correl
                except AttributeError:
                    deltaxs = np.nan
                    deltays = np.nan
                    correl_max = np.nan

            deltaxs[ivec] = deltax
            deltays[ivec] = deltay
            correls_max[ivec] = correl_max

        if deltaxs_approx is not None:
            deltaxs += deltaxs_approx
            deltays += deltays_approx

        return deltaxs, deltays, xs, ys, correls_max, correls, errors

    def _crop_im(self, ixvec, iyvec, im):
        niwo2 = self.niwo2
        subim = im[iyvec - niwo2:iyvec + niwo2,
                   ixvec - niwo2:ixvec + niwo2]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()


class FirstWorkPIV(BaseWorkPIV):

    @classmethod
    def _complete_params_with_default(cls, params):
        params._set_child('piv0', attribs={
            'shape_crop_im0': 48,
            'shape_crop_im1': None,
            'delta_max': None,
            'delta_mean': None,
            'method_correl': 'fftw',
            'method_subpix': 'centroid'})

        params.piv0._set_child('grid', attribs={
            'overlap': 0.5,
            'from': 'overlap'})

        params._set_child('mask', attribs={})


class WorkPIVFromDisplacement(BaseWorkPIV):

    def __init__(self, params=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params

        shape_crop_im0 = params.piv0.shape_crop_im0
        overlap = params.piv0.grid.overlap

        self.shape_crop_im0 = shape_crop_im0
        if isinstance(shape_crop_im0, int):
            n_interrogation_window = shape_crop_im0
        else:
            raise NotImplementedError(
                'For now, shape_crop_im0 has to be an integer!')

        niw = self.n_interrogation_window = n_interrogation_window/2
        self.overlap = overlap

        self.niwo2 = niw/2

        self._init_correl()

    def calcul(self, piv_results):

        if not isinstance(piv_results, HeavyPIVResults):
            raise ValueError

        couple = piv_results.couple

        im0, im1 = couple.get_arrays()
        if not hasattr(self, 'ixvecs_grid'):
            self._prepare_with_image(im0)

        # for the interpolation
        selection = ~np.isnan(piv_results.deltaxs)

        xs = piv_results.xs[selection]
        ys = piv_results.ys[selection]

        deltaxs = piv_results.deltaxs[selection]
        deltays = piv_results.deltays[selection]

        coef = 0.5
        centers = np.vstack([xs, ys])
        piv_results.deltaxs_smooth, piv_results.deltax_tps = \
            compute_tps_coeff(centers, deltaxs, coef)
        piv_results.deltays_smooth, piv_results.deltay_tps = \
            compute_tps_coeff(centers, deltays, coef)

        new_positions = np.vstack([self.ixvecs_grid, self.iyvecs_grid])

        tps = ThinPlateSpline(new_positions, centers)

        deltaxs_approx = np.round(tps.compute_field(
            piv_results.deltax_tps)).astype('int32')
        deltays_approx = np.round(tps.compute_field(
            piv_results.deltay_tps)).astype('int32')

        deltaxs, deltays, xs, ys, correls_max, correls, errors = \
            self._loop_vectors(im0, im1,
                               deltaxs_approx=deltaxs_approx,
                               deltays_approx=deltays_approx)

        result = HeavyPIVResults(
            deltaxs, deltays, xs, ys, errors,
            correls_max=correls_max, correls=correls,
            couple=deepcopy(couple), params=self.params)

        return result

    def calcul_indices_vec(self, deltaxs_approx=None, deltays_approx=None):

        ixs0 = self.ixvecs_grid - deltaxs_approx//2
        iys0 = self.iyvecs_grid - deltays_approx//2
        ixs1 = ixs0 + deltaxs_approx
        iys1 = iys0 + deltays_approx

        xs = (ixs0 + ixs1) / 2.
        ys = (iys0 + iys1) / 2.

        ixs0_pad = ixs0 + self.npad
        iys0_pad = iys0 + self.npad
        ixs1_pad = ixs1 + self.npad
        iys1_pad = iys1 + self.npad

        return xs, ys, ixs0_pad, iys0_pad, ixs1_pad, iys1_pad


class WorkFIX(BaseWork):

    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag='params')
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params, tag='fix'):

        params._set_child(tag, attribs={
            'correl_min': 0.4,
            'delta_diff': 0.1,
            'delta_max': 4,
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
        delta_max2 = self.params.delta_max**2
        delta2s = deltaxs**2 + deltays**2
        inds = (delta2s > delta_max2).nonzero()[0]
        put_to_nan(inds, 'delta2 < delta_max2')

        # warning condition neighbour not implemented...

        return piv_results


class WorkPIV(BaseWork):

    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag='params')
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params):
        FirstWorkPIV._complete_params_with_default(params)
        WorkFIX._complete_params_with_default(params)

        params._set_child('multipass', attribs={'number': 0})

    def __init__(self, params=None):
        self.params = params
        self.work_piv0 = FirstWorkPIV(params)
        self.work_fix0 = WorkFIX(params.fix)

        if params.multipass.number > 0:
            self.work_piv1 = WorkPIVFromDisplacement(params)

    def calcul(self, couple):

        piv_result = self.work_piv0.calcul(couple)
        piv_result = self.work_fix0.calcul(piv_result)

        results = MultipassPIVResults()
        results.append(piv_result)

        if self.params.multipass.number > 0:
            piv_result1 = self.work_piv1.calcul(piv_result)
            results.append(piv_result1)

        return results

    def _prepare_with_image(self, im):
        self.work_piv0._prepare_with_image(im)
        if self.params.multipass.number > 0:
            self.work_piv1._prepare_with_image(im)
