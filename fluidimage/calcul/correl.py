"""

correlation computed with fft are much faster !

"""

from __future__ import division, print_function

import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import correlate
from numpy.fft import fft2, ifft2

from .fft import FFTW2DReal2Complex, CUFFT2DReal2Complex


class NoPeakError(Exception):
    """No peak"""


# if 'OMP_NUM_THREADS' in os.environ:
#     nthreads = int(os.environ['OMP_NUM_THREADS'])
# else:
#     pass

# It seems that it is better to used nthreads = 1 for the fft with very small
# dimension used for PIV
nthreads = 1


class CorrelBase(object):
    def __init__(self, im0_shape, im1_shape):
        self.inds0 = tuple(np.array(im0_shape)//2 - 1)

        self.subpix = SubPix()

    def compute_displacement_from_indices(self, indices):
        return np.array(self.inds0) - np.array(indices)

    def compute_displacement_from_correl(self, correl, method='centroid'):
        iy, ix = np.unravel_index(correl.argmax(), correl.shape)
        indices = self.subpix.compute(
            correl, ix, iy, method)
        return self.compute_displacement_from_indices(indices)


class CorrelScipySignal(CorrelBase):
    def __init__(self, im0_shape, im1_shape=None, mode='same'):

        if im1_shape is None:
            im1_shape = im0_shape

        super(CorrelScipySignal, self).__init__(im0_shape, im1_shape)

        modes = ['valid', 'same']
        if mode not in modes:
            raise ValueError('mode should be in ' + modes)
        self.mode = mode
        if mode == 'same':
            ny, nx = im0_shape
            if nx % 2 == 0:
                ind0x = nx // 2 - 1
            else:
                ind0x = nx // 2
            if ny % 2 == 0:
                ind0y = ny // 2 - 1
            else:
                ind0y = ny // 2

        else:
            ny, nx = np.array(im0_shape) - np.array(im1_shape)
            ind0x = nx // 2
            ind0y = ny // 2

        self.inds0 = tuple([ind0y, ind0x])

    def __call__(self, im0, im1):
        norm = np.sum(im1**2)
        if self.mode == 'valid':
            correl = correlate2d(im0, im1, mode='valid')
        elif self.mode == 'same':
            correl = correlate2d(im0, im1, mode='same', fillvalue=im1.min())
        else:
            assert False, 'Bad value for self.mode'

        return correl/norm


class CorrelScipyNdimage(CorrelBase):
    def __init__(self, im0_shape, im1_shape=None):
        super(CorrelScipyNdimage, self).__init__(im0_shape, im1_shape)
        self.inds0 = tuple(np.array(im0_shape)//2)

    def __call__(self, im0, im1):
        norm = np.sum(im1**2)
        return correlate(im0, im1, mode='constant', cval=im1.min())/norm


class CorrelFFTNumpy(CorrelBase):
    def __init__(self, im0_shape, im1_shape):
        super(CorrelFFTNumpy, self).__init__(im0_shape, im1_shape)
        if im0_shape != im1_shape:
            raise ValueError('The input images have to have the same shape.')

    def __call__(self, im0, im1):
        norm = np.sum(im1**2)
        corr = ifft2(fft2(im0).conj() * fft2(im1)).real / norm
        return np.fft.fftshift(corr[::-1, ::-1])


class CorrelFFTW(CorrelBase):
    FFTClass = FFTW2DReal2Complex

    def __init__(self, im0_shape, im1_shape=None):
        super(CorrelFFTW, self).__init__(im0_shape, im1_shape)

        if im1_shape is None:
            im1_shape = im0_shape

        if im0_shape != im1_shape:
            raise ValueError('The input images have to have the same shape.')

        n0, n1 = im1_shape
        self.op = self.FFTClass(n1, n0)

    def __call__(self, im0, im1):
        norm = np.sum(im1**2)
        op = self.op
        corr = op.ifft(op.fft(im0).conj() * op.fft(im1)) / norm
        return np.fft.fftshift(corr[::-1, ::-1])


class CorrelCuFFT(CorrelBase):
    FFTClass = CUFFT2DReal2Complex


class SubPix(object):
    methods = ['2d_gaussian', 'centroid']

    def __init__(self):
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

    def compute(self, correl, ix, iy, method='centroid'):
        """Find peak using linalg.solve (buggy?)

        Parameters
        ----------

        correl_map: numpy.ndarray

          Normalized correlation
        """
        if method not in self.methods:
            raise ValueError('method has to be in {}'.format(self.methods))

        n = self.n_subpix_zoom

        ny, nx = correl.shape

        if iy-n < 0 or iy+n+1 > ny or \
           ix-n < 0 or ix+n+1 > nx:
            raise NoPeakError

        if method == '2d_gaussian':

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

        return deply + iy + 0.5, deplx + ix + 0.5
