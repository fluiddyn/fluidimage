"""Base classes for PIV
=======================


A piv computation is often organized in steps. Each step can also
be organized in sub-steps.




"""


from __future__ import print_function

from copy import deepcopy

# import matplotlib.pyplot as plt
import numpy as np

from fluidimage.calcul.correl import CorrelWithFFT
from piv_results import HeavyPIVResults


class NoPeakError(Exception):
    """No peak"""


class BaseStep(object):
    def __init__(self, params=None):
        self.params = params


class BaseWork(object):
    def __init__(self, params=None):
        self.params = params


class PIVSerie(BaseStep):
    def __init__(self, series_images=None, params=None,
                 n_interrogation_window=None, overlap=None):

        self.outputs = {}
        self.series_images = series_images

        self.work = FirstPIVWork(
            params=params,
            n_interrogation_window=n_interrogation_window, overlap=overlap)

        serie = self.series_images.get_serie_from_index(0)
        im = serie.get_array_from_index(0)

        self.work.prepare(im)

    def compute_outputs(self, compute_all=False):
        for index, couple in enumerate(self.series_images):
            results = self.work.calcul_1_field(couple)
            if index not in self.outputs or compute_all:
                self.outputs[index] = results


class BasePIVWork(BaseWork):
    def __init__(self, params=None,
                 n_interrogation_window=None, overlap=None):

        self.n_interrogation_window = n_interrogation_window

        if overlap is None:
            overlap = 0.5
        self.overlap = overlap

        niw = self.n_interrogation_window
        self.niwo2 = niw/2

        correl = CorrelWithFFT(niw, niw)
        self._calcul_correl_norm = correl.calcul_correl_norm

        # subpix initialization
        self.n_subpix_zoom = 2
        xs = np.arange(2*self.n_subpix_zoom+1, dtype=float)
        ys = np.arange(2*self.n_subpix_zoom+1, dtype=float)
        X, Y = np.meshgrid(xs, ys)
        nx, ny = X.shape
        X = X.ravel()
        Y = Y.ravel()
        M = np.reshape(np.concatenate(
            (X**2, Y**2, X, Y, np.ones(nx*ny))), (5, nx*ny)).T
        self.Minv_subpix = np.linalg.pinv(M)

    def prepare(self, im):

        len_y, len_x = im.shape
        niw = self.n_interrogation_window
        overlap = int(np.round(self.overlap*niw))

        self.inds_x_vec = np.arange(0, len_x, overlap, dtype=int)
        self.inds_y_vec = np.arange(0, len_y, overlap, dtype=int)

    def calcul_1_field(self, couple):
        im0, im1 = couple.get_arrays()
        niwo2 = self.niwo2
        tmp = [(niwo2, niwo2), (niwo2, niwo2)]
        im0pad = np.pad(im0 - im0.min(), tmp, 'constant')
        im1pad = np.pad(im1 - im1.min(), tmp, 'constant')

        deltaxs, deltays, correls = self._loop_vectors(im0pad, im1pad)

        result = HeavyPIVResults(
            deltaxs, deltays, correls, deepcopy(couple), self)

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
                correl = self._calcul_correl_norm(im0crop, im1crop)
                correls.append(correl)
                try:
                    deltax, deltay = self._find_peak(correl)
                    deltaxs[iy, ix] = deltax
                    deltays[iy, ix] = deltay
                except NoPeakError as e:
                    print(e)
                    deltaxs[iy, ix] = np.nan
                    deltays[iy, ix] = np.nan

        return deltaxs, deltays, correls

    def _crop_subimages(self, ixvec, iyvec, im0, im1):
        raise NotImplementedError

    def _find_peak(self, correl):
        iy, ix = np.unravel_index(correl.argmax(), correl.shape)
        ix, iy = self._find_peak_subpixel(
            correl, ix, iy, 'centroid')  # method : 2Dgaussian or centroid
        return ix - self.niwo2, iy - self.niwo2

    def _find_peak_subpixel_old(self, correl, ix, iy):
        return ix, iy

    def _find_peak_subpixel(self, correl, ix, iy, method):
        """Find peak using linalg.solve (buggy?)

        Parameters
        ----------

        correl_map: numpy.ndarray

          Normalized correlation
        """
        n = self.n_subpix_zoom

        ny, nx = correl.shape

        if iy-n < 0 or iy+n+1 > ny or \
           ix-n < 0 or ix+n+1 > nx:
            raise NoPeakError

        if method == '2Dgaussian':

            # crop: possibly buggy!
            correl = correl[iy-n:iy+n+1,
                            ix-n:ix+n+1]

            ny, nx = correl.shape

            assert nx == ny == 2*n + 1

            correl_map = correl.ravel()
            correl_map[correl_map == 0.] = 1e-6

            coef = np.dot(self.Minv_subpix, np.log(correl_map))

            sigmax = 1/np.sqrt(-2*coef[0])
            sigmay = 1/np.sqrt(-2*coef[1])
            X0 = coef[2]*sigmax**2
            Y0 = coef[3]*sigmay**2

            tmp = 2*n + 1
            if X0 > tmp or Y0 > tmp:
                raise NoPeakError

            deplx = X0 - nx/2  # displacement x
            deply = Y0 - ny/2  # displacement y

        elif method == 'centroid':

            correl = correl[iy-1:iy+2, ix-1:ix+2]
            ny, nx = correl.shape

            X, Y = np.meshgrid(range(3), range(3))
            X0 = np.sum(X * correl) / np.sum(correl)
            Y0 = np.sum(Y * correl) / np.sum(correl)

            if X0 > 2 or Y0 > 2:
                raise NoPeakError

            deplx = X0 - nx/2  # displacement x
            deply = Y0 - ny/2  # displacement y

        return deplx + ix, deply + iy


class FirstPIVWork(BasePIVWork):

    def _crop_im(self, ixvec, iyvec, im):
        niwo2 = self.niwo2
        subim = im[iyvec - niwo2:iyvec + niwo2,
                   ixvec - niwo2:ixvec + niwo2]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def _crop_subimages(self, ixvec, iyvec, im0, im1):
        return (self._crop_im(ixvec, iyvec, im0),
                self._crop_im(ixvec, iyvec, im1))


class PIVWorkFromDisplacement(BasePIVWork):
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
    from copy import copy

    from fluiddyn.util.serieofarrays import \
        SerieOfArraysFromFiles, SeriesOfArrays

    path = '../image_samples/Oseen/Oseen_center*'

    def give_indslices_from_indserie(iserie):
        indslices = copy(serie_arrays._index_slices_all_files)
        indslices[0] = [iserie+1, iserie+3]
        return indslices

    # path = '../image_samples/Karman'

    # def give_indslices_from_indserie(iserie):
    #     indslices = copy(serie_arrays._index_slices_all_files)
    #     indslices[0] = [2*iserie+1, 2*iserie+3, 1]
    #     return indslices

    serie_arrays = SerieOfArraysFromFiles(path)
    series = SeriesOfArrays(serie_arrays, give_indslices_from_indserie,
                            ind_stop=None)

    o = PIVSerie(
        series_images=series, n_interrogation_window=48, overlap=0.5)
    o.compute_outputs()

    for results in o.outputs.values():
        print(results.couple.get_name_files())
        results.display()

    # nx = ny = 16

    # xs = np.arange(nx, dtype=float)
    # ys = np.arange(ny, dtype=float)
    # X, Y = np.meshgrid(xs, ys)

    # x0 = y0 = 5.
    # sigma = 1.

    # correl = np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2))

    # plt.ion()
    # # plt.imshow(correl)

    # o._find_peak_linalg(correl)
