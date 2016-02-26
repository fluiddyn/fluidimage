"""Piv work and subworks
========================

To do:

- improve NoPeakError (different classes with parameters, as in "fix" of UVmat)

- nearly as in UVmat? Variables to use (?):

  * nb_vec (int)
  * x (nb_vec)
  * y (nb_vec)
  * deltax (nb_vec)
  * deltay (nb_vec)
  * correl_peak (nb_vec)
  * warning {ivec: str}
  * error {ivec: str}

- as in UVmat: civ1, fix1, patch1, ...

- as in UVmat: multipass

- as in UVmat: patch "thin-plate spline" (?). Add variables as in UVmat
  (NbCenter, Coord_tps, SubRange, U_tps, V_tps)

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

from ..data_objects.piv import ArrayCouple, HeavyPIVResults
from ..calcul.correl import PIVError, correlation_classes
from ..works import BaseWork


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

        try:
            correl_cls = correlation_classes[params.piv0.method_correl]
        except KeyError:
            raise ValueError(
                'params.piv0.method_correl should be in ' +
                str(correlation_classes.keys()))

        self.correl = correl_cls(im0_shape=(niw, niw),
                                 method_subpix=params.piv0.method_subpix)

    def _prepare_with_image(self, im):

        len_y, len_x = im.shape
        niw = self.n_interrogation_window
        step = niw - int(np.round(self.overlap*niw))

        inds_x_vec = np.arange(0, len_x, step, dtype=int)
        inds_y_vec = np.arange(0, len_y, step, dtype=int)

        inds_x_vec, inds_y_vec = np.meshgrid(inds_x_vec, inds_y_vec)

        self.inds_x_vec = inds_x_vec.flatten()
        self.inds_y_vec = inds_y_vec.flatten()

    def calcul(self, couple):
        if isinstance(couple, SerieOfArraysFromFiles):
            couple = ArrayCouple(serie=couple)

        if not isinstance(couple, ArrayCouple):
            raise ValueError

        im0, im1 = couple.get_arrays()
        if not hasattr(self, 'inds_x_vec'):
            self._prepare_with_image(im0)

        niwo2 = self.niwo2
        tmp = [(niwo2, niwo2), (niwo2, niwo2)]
        im0pad = np.pad(im0 - im0.min(), tmp, 'constant')
        im1pad = np.pad(im1 - im1.min(), tmp, 'constant')

        deltaxs, deltays, correls, correls_max, errors = \
            self._loop_vectors(im0pad, im1pad)

        result = HeavyPIVResults(
            deltaxs, deltays, self.inds_x_vec, self.inds_y_vec,
            errors, correls_max=correls_max, correls=correls,
            couple=deepcopy(couple), params=self.params)

        return result

    def _loop_vectors(self, im0, im1):

        correls = []
        errors = {}
        deltaxs = np.empty(self.inds_y_vec.shape)
        deltays = np.empty_like(deltaxs)
        correls_max = np.empty_like(deltaxs)

        inds_x_vec_pad = self.inds_x_vec + self.niwo2
        inds_y_vec_pad = self.inds_y_vec + self.niwo2

        for ivec, iyvec in enumerate(inds_y_vec_pad):
            ixvec = inds_x_vec_pad[ivec]

            im0crop, im1crop = self._crop_subimages(ixvec, iyvec, im0, im1)
            correl, coef_norm = self.correl(im0crop, im1crop)
            correls.append(correl)
            try:
                deltay, deltax, correl_max = \
                    self.correl.compute_displacement_from_correl(
                        correl, coef_norm=coef_norm)
                deltaxs[ivec] = deltax
                deltays[ivec] = deltay
                correls_max[ivec] = correl_max
            except PIVError as e:
                try:
                    deltay, deltax, correl_max = \
                        e.results_compute_displacement_from_correl
                    errors[(ivec)] = e.explanation
                    deltaxs[ivec] = deltax
                    deltays[ivec] = deltay
                    correls_max[ivec] = correl_max
                except AttributeError:
                    errors[(ivec)] = e.explanation
                    deltaxs[ivec] = np.nan
                    deltays[ivec] = np.nan

        return deltaxs, deltays, correls, correls_max, errors

    def _crop_subimages(self, ixvec, iyvec, im0, im1):
        raise NotImplementedError


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

    def _crop_im(self, ixvec, iyvec, im):
        niwo2 = self.niwo2
        subim = im[iyvec - niwo2:iyvec + niwo2,
                   ixvec - niwo2:ixvec + niwo2]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def _crop_subimages(self, ixvec, iyvec, im0, im1):
        return (self._crop_im(ixvec, iyvec, im0),
                self._crop_im(ixvec, iyvec, im1))


class WorkPIVFromDisplacement(BaseWorkPIV):
    def _crop_im0(self, ixvec, iyvec, im):
        niwo2 = self.niwo2
        subim = im[iyvec - niwo2:iyvec + niwo2,
                   ixvec - niwo2:ixvec + niwo2]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def _crop_im1(self, ixvec, iyvec, im):
        niwo2 = self.niwo2
        subim = im[iyvec - niwo2:iyvec + niwo2,
                   ixvec - niwo2:ixvec + niwo2]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def _crop_subimages(self, ixvec, iyvec, im0, im1):
        return (self._crop_im0(ixvec, iyvec, im0),
                self._crop_im1(ixvec, iyvec, im1))


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

        deltaxs = piv_results.deltaxs  # .copy()
        deltays = piv_results.deltays  # .copy()

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

        params._set_child('multipass', attribs={})

    def __init__(self, params=None):
        self.work_piv0 = FirstWorkPIV(params)
        self.work_fix0 = WorkFIX(params.fix)

    def calcul(self, couple):

        piv_result = self.work_piv0.calcul(couple)
        piv_result = self.work_fix0.calcul(piv_result)
        return piv_result
