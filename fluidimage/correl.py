"""

correlation computed with fft are much faster !

"""

from __future__ import division, print_function

import numpy as np
from scipy.signal import convolve2d
import pyfftw

# if 'OMP_NUM_THREADS' in os.environ:
#     nthreads = int(os.environ['OMP_NUM_THREADS'])
# else:
#     pass

# It seems that it is better to used nthreads = 1 for the fft with very small
# dimension used for PIV
nthreads = 1


def calcul_correl_norm_scipy(im0, im1):
    correl = convolve2d(
        im1, np.rot90(im0, 2), 'same', fillvalue=im1.min())
    correl_min = correl.min()
    return (correl - correl_min) / (correl.max() - correl_min)


class FFTW2DReal2Complex(object):
    """ A class to use fftw """
    type_real = 'float32'
    type_complex = 'complex64'

    def __init__(self, nx, ny):

        if nx % 2 != 0 or ny % 2 != 0:
            raise ValueError('nx and ny should be even')
        shapeX = [ny, nx]
        shapeK = [ny, nx//2 + 1]

        self.shapeX = shapeX
        self.shapeK = shapeK

        self.arrayX = pyfftw.n_byte_align_empty(shapeX, 16, self.type_real)
        self.arrayK = pyfftw.n_byte_align_empty(shapeK, 16, self.type_complex)

        self.fftplan = pyfftw.FFTW(input_array=self.arrayX,
                                   output_array=self.arrayK,
                                   axes=(0, 1),
                                   direction='FFTW_FORWARD',
                                   threads=nthreads)
        self.ifftplan = pyfftw.FFTW(input_array=self.arrayK,
                                    output_array=self.arrayX,
                                    axes=(0, 1),
                                    direction='FFTW_BACKWARD',
                                    threads=nthreads)

        self.coef_norm = nx*ny

    def fft(self, ff):
        self.arrayX[:] = ff
        self.fftplan(normalise_idft=False)
        return self.arrayK/self.coef_norm

    def ifft(self, ff_fft):
        self.arrayK[:] = ff_fft
        self.ifftplan(normalise_idft=False)
        return self.arrayX.copy()

    def compute_energy_from_Fourier(self, ff_fft):
        return (np.sum(abs(ff_fft[:, 0])**2 + abs(ff_fft[:, -1])**2) +
                2*np.sum(abs(ff_fft[:, 1:-1])**2))/2

    def compute_energy_from_spatial(self, ff):
        return np.mean(abs(ff)**2)/2

    def project_fft_on_realX(self, ff_fft):
        return self.fft(self.ifft(ff_fft))


class FFTW2DReal2ComplexFloat64(FFTW2DReal2Complex):
    """ A class to use fftw """
    type_real = 'float64'
    type_complex = 'complex128'


class CorrelWithFFT(object):
    def __init__(self, nx, ny):
        self.op = FFTW2DReal2Complex(nx, ny)

    def calcul_correl_norm(self, im0, im1):
        fft = self.op.fft
        ifft = self.op.ifft
        correl = np.fft.fftshift(ifft(fft(im0).conj() * fft(im1)))
        correl_min = correl.min()
        return (correl - correl_min) / (correl.max() - correl_min)
