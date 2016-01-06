"""Base classes for PIV
=======================


A piv computation is often organized in steps. Each step can also
be organized in sub-steps.




"""

import numpy as np
import scipy as sc

from correl import calcul_correl_norm_scipy, CorrelWithFFT


class NoPeakError(Exception):
    """No peak"""


class RegionOfInterest(object):
    """Masks..."""


class BaseStep(object):
    def __init__(self, params=None, data=None):
        self.params = params

    def prepare(self):
        if hasattr(self, '_substeps'):
            for substep in self._substeps:
                substep.prepare()


class BasePIVStep(BaseStep):
    def __init__(self, params=None, data=None, series_images=None,
                 n_interrogation_window=None, step=None,
                 region_of_interest=None):
        self.n_interrogation_window = n_interrogation_window

        if step is None:
            step = 0.5
        self.step = step
        self.region_of_interest = region_of_interest
        self.series_images = series_images

        niw = self.n_interrogation_window
        self.niwo2 = niw/2

        self.outputs = {}

        # self._calcul_correl_norm = calcul_correl_norm_scipy

        correl = CorrelWithFFT(niw, niw)
        self._calcul_correl_norm = correl.calcul_correl_norm

    def prepare(self):
        super(BasePIVStep, self).prepare()

        serie = self.series_images.get_serie_from_index(0)

        im0 = serie.get_array_from_index(0)

        if self.region_of_interest is not None:
            im0 = self.region_of_interest.select(im0)

        len_y, len_x = im0.shape
        niw = self.n_interrogation_window
        step = int(np.round(self.step*niw))

        self.inds_x_vec = np.arange(0, len_x, step, dtype=int)
        self.inds_y_vec = np.arange(0, len_y, step, dtype=int)

    def compute_outputs(self, compute_all=False):
        for index, serie in enumerate(self.series_images):
            im0, im1 = serie.get_arrays()
            results = self._calcul_1_field(im0, im1)
            if index not in self.outputs or compute_all:
                self.outputs[index] = results

    def _calcul_1_field(self, im0, im1):
        niwo2 = self.niwo2
        tmp = [(niwo2, niwo2), (niwo2, niwo2)]
        im0pad = np.pad(im0 - im0.min(), tmp, 'constant')
        im1pad = np.pad(im1 - im1.min(), tmp, 'constant')
        return self._loop_vectors(im0pad, im1pad)

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
                correl = self._calcul_correl_norm(im0crop, im1crop)
                correls.append(correl)
                try:
                    deltax, deltay = self._find_peak(correl)
                    deltaxs[iy, ix] = deltax
                    deltays[iy, ix] = deltay
                except NoPeakError:
                    pass

        results = {
            'correls': correls, 'deltaxs': deltaxs, 'deltays': deltays}
        return results

    def _crop_subimages(self, ixvec, iyvec, im0, im1):
        raise NotImplementedError

    def _find_peak(self, correl):
        iy, ix = np.unravel_index(correl.argmax(), correl.shape)
        ix, iy = self._find_peak_subpixel(
            correl, ix, iy)
        return ix - self.niwo2, iy - self.niwo2

    def _find_peak_subpixel(self, correl, ix, iy):
        return ix, iy

    def subpix_interp(correl_map):
        """Subpixel interpolation (buggy)

        Parameters
        ----------

        correl_map: numpy.ndarray

          Normalized correlation
        """
        ny = correl_map.shape[0]
        nx = correl_map.shape[1]
        Y = np.dot(np.reshape(
            np.linspace(1, ny, ny), (ny, 1)), np.ones((1, nx)))  # grille
        X = np.dot(np.ones((ny, 1)),
                   np.reshape(np.linspace(1, nx, nx), (1, nx)))  # grille
        correl_map = np.reshape(correl_map, (nx*ny, 1), order='F')
        X = np.reshape(X, (nx*ny, 1), order='F')
        Y = np.reshape(Y, (nx*ny, 1), order='F')
        X = np.double(X)
        Y = np.double(Y)
        M = np.reshape(np.concatenate((X**2, Y**2, X, Y, X**0)),
                       (nx*ny, 5), order='F')
        coef = np.dot(np.linalg.pinv(M), sc.log(A))
        Sx = 1/np.sqrt(-2*coef[0])
        Sy = 1/np.sqrt(-2*coef[1])
        X0 = coef[2]*Sx**2
        Y0 = coef[3]*Sy**2
        deplx = X0-(nx+1)/2  # displacement x
        deply = Y0-(ny+1)/2  # displacement y

        return deplx, deply


class FirstPIVStep(BasePIVStep):

    def _crop_im(self, ixvec, iyvec, im):
        niwo2 = self.niwo2
        subim = im[iyvec - niwo2:iyvec + niwo2,
                   ixvec - niwo2:ixvec + niwo2]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def _crop_subimages(self, ixvec, iyvec, im0, im1):
        return (self._crop_im(ixvec, iyvec, im0),
                self._crop_im(ixvec, iyvec, im1))


class PIVStepFromDisplacement(BasePIVStep):
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


class Filter(BaseStep):
    pass


if __name__ == '__main__':
    import os
    from copy import copy

    from fluiddyn.util.serieofarrays import \
        SerieOfArraysFromFiles, SeriesOfArrays
    import display

    HOME = os.environ['HOME']
    base_path = os.path.join(HOME, 'Dev/howtopiv')

    # path = 'samples/Karman'
    # base_name = 'PIVlab_Karman'

    # def give_indslices_from_indserie(iserie):
    #     indslices = copy(serie_arrays._index_slices_all_files)
    #     indslices[0] = [iserie+1, iserie+3]
    #     return indslices

    path = 'samples/Oseen'
    base_name = 'PIVlab_Oseen_z'

    def give_indslices_from_indserie(iserie):
        indslices = copy(serie_arrays._index_slices_all_files)
        indslices[0] = [2*iserie+1, 2*iserie+3, 1]
        return indslices

    serie_arrays = SerieOfArraysFromFiles(
        os.path.join(base_path, path), base_name=base_name)
    series = SeriesOfArrays(serie_arrays, give_indslices_from_indserie,
                            ind_stop=None)

    o = FirstPIVStep(
        series_images=series, n_interrogation_window=64, step=0.6)
    o.prepare()
    o.compute_outputs()

    for index, serie in enumerate(series):
        print(serie.get_name_files())
        im0, im1 = serie.get_arrays()
        results = o.outputs[index]
        correls = results['correls']
        deltaxs = results['deltaxs']
        deltays = results['deltays']

        display.display(im0, im1, o.inds_x_vec, o.inds_y_vec, deltaxs, deltays)
