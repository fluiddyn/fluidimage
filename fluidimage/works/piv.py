"""Piv work and subworks
========================

To do:

- reorganize (WorkPIV should not be a FirstWorkPIV)

- use different methods for the correlations (with parameters)

- use parameter to chose the method for subpix

- improve NoPeakError (different classes with parameters, as in "fix" of UVmat)

- as in UVmat? Variables to keep: xs, ys, deltaxs, deltays,
  correl_peak, deltaxs_2ndpeak, deltays_2ndpeak, correl_2ndpeak,
  no_peak_errors... (?)

- as in UVmat: civ1, fix1, patch1, ... (?)

- detect and save multipeaks


"""


from __future__ import print_function

from copy import deepcopy

import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles

from ..data_objects.piv import ArrayCouple, HeavyPIVResults
from ..calcul.correl import (NoPeakError, CorrelFFTW)
from ..works import BaseWork


class BaseWorkPIV(BaseWork):

    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag='params')
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params):
        params._set_child('piv', attribs={
            'n_interrogation_window': 48,
            'overlap': 0.5})

    def __init__(self, params=None,
                 n_interrogation_window=None, overlap=None):

        if n_interrogation_window is None:
            n_interrogation_window = params.piv.n_interrogation_window

        if overlap is None:
            overlap = params.piv.overlap

        niw = self.n_interrogation_window = n_interrogation_window
        self.overlap = overlap

        self.niwo2 = niw/2

        self.correl = CorrelFFTW(im0_shape=(niw, niw))

    def _prepare_with_image(self, im):

        len_y, len_x = im.shape
        niw = self.n_interrogation_window
        overlap = int(np.round(self.overlap*niw))

        self.inds_x_vec = np.arange(0, len_x, overlap, dtype=int)
        self.inds_y_vec = np.arange(0, len_y, overlap, dtype=int)

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

        deltaxs, deltays, correls = self._loop_vectors(im0pad, im1pad)

        result = HeavyPIVResults(
            deltaxs, deltays, self.inds_x_vec, self.inds_y_vec,
            correls, deepcopy(couple))

        return result

    def _loop_vectors(self, im0, im1):

        correls = []
        deltaxs = np.empty([self.inds_y_vec.size, self.inds_x_vec.size])
        deltays = np.empty_like(deltaxs)

        inds_x_vec_pad = self.inds_x_vec + self.niwo2
        inds_y_vec_pad = self.inds_y_vec + self.niwo2

        iy = -1
        for iyvec in inds_y_vec_pad:
            iy += 1
            ix = -1
            for ixvec in inds_x_vec_pad:
                ix += 1
                im0crop, im1crop = self._crop_subimages(ixvec, iyvec, im0, im1)
                correl = self.correl(im0crop, im1crop)
                correls.append(correl)
                try:
                    deltax, deltay = \
                        self.correl.compute_displacement_from_correl(
                            correl, method='centroid')
                    deltaxs[iy, ix] = deltax
                    deltays[iy, ix] = deltay
                except NoPeakError as e:
                    print(e)
                    deltaxs[iy, ix] = np.nan
                    deltays[iy, ix] = np.nan

        return deltaxs, deltays, correls

    def _crop_subimages(self, ixvec, iyvec, im0, im1):
        raise NotImplementedError


class FirstWorkPIV(BaseWorkPIV):

    def _crop_im(self, ixvec, iyvec, im):
        niwo2 = self.niwo2
        subim = im[iyvec - niwo2:iyvec + niwo2,
                   ixvec - niwo2:ixvec + niwo2]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def _crop_subimages(self, ixvec, iyvec, im0, im1):
        return (self._crop_im(ixvec, iyvec, im0),
                self._crop_im(ixvec, iyvec, im1))

WorkPIV = FirstWorkPIV


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
